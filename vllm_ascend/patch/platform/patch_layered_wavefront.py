"""Let layered prefill fill the PP pipeline with a stage wavefront.

A layered-prefill layer group runs on exactly one PP stage.  The upstream
policy keeps a single prompt in the layer pipeline, so at PP=N only one stage
has prefill work on any step and the other N-1 idle.

``layered_prefill_config.max_concurrent_layered_prefills`` (default 1, clamped
to ``pipeline_parallel_size``) lets up to that many prompts hold layer groups
at once.  Admission is staggered: a new prompt always enters on stage 0, so it
is admitted only once stage 0 has drained.  Prompts admitted together would
stay on the same stage in lockstep and add nothing.

With ``fuse_mixed_batch`` a decode that joins a span at group 0 rides every
group of that span and samples once at its end.  Two overlapping spans must
not recruit the same decode: it could finish neither, never leaves
``running``, and the engine stalls.  At concurrency 1 spans never overlap, so
that guard only engages when the wavefront is on.

The scheduler methods live on vLLM's base ``Scheduler``; subclasses installed
through ``scheduler_cls`` (and the balance scheduler) inherit the patch.
"""

import functools
import inspect

import vllm.v1.core.sched.scheduler as _sched_mod
from vllm.distributed.utils import get_pp_indices
from vllm.logger import logger

from vllm_ascend.ascend_config import LayeredPrefillExtensions


def _base_scheduler_cls():
    # patch_balance_schedule rebinds the module name to a subclass; patch the
    # class it derives from so every scheduler_cls sees the same behaviour.
    for cls in _sched_mod.Scheduler.__mro__:
        if cls.__module__ == _sched_mod.__name__ and cls.__name__ == "Scheduler":
            return cls
    return _sched_mod.Scheduler


@functools.lru_cache(maxsize=64)
def layer_group_owners(num_hidden_layers: int, num_groups: int, pp_size: int) -> tuple[int | None, ...]:
    """PP rank owning each layer group, None for a group split across stages."""
    from vllm.v1.core.layered_prefill import make_pp_aligned_layer_group_ranges

    stages = [get_pp_indices(num_hidden_layers, rank, pp_size) for rank in range(pp_size)]
    owners: list[int | None] = []
    for layer_range in make_pp_aligned_layer_group_ranges(num_hidden_layers, num_groups, pp_size):
        owners.append(
            next(
                (
                    rank
                    for rank, (start, end) in enumerate(stages)
                    if start <= layer_range.start and layer_range.end <= end
                ),
                None,
            )
        )
    return tuple(owners)


def owner_of_group(policy, num_groups: int, group_id: int) -> int | None:
    if num_groups <= 0 or not 0 <= group_id < num_groups:
        return None
    return layer_group_owners(
        int(policy.num_hidden_layers),
        int(num_groups),
        int(policy.pipeline_parallel_size),
    )[group_id]


def group_owner(policy, request) -> int | None:
    """PP rank that will execute the request's next layer group."""
    return owner_of_group(
        policy,
        int(getattr(request, "layered_prefill_num_groups", 0) or 0),
        int(getattr(request, "layered_prefill_group_id", 0) or 0),
    )


def in_flight_group_owner(policy, request) -> int | None:
    """PP rank executing the group already handed to the worker.

    The scheduler advances the group cursor when it hands a group to the
    worker, so the in-flight group is one behind that cursor.
    """
    return owner_of_group(
        policy,
        int(getattr(request, "layered_prefill_num_groups", 0) or 0),
        int(getattr(request, "layered_prefill_group_id", 0) or 0) - 1,
    )


def _layered_wavefront_depth(self) -> int:
    depth = getattr(self, "_layered_wavefront_depth_cache", None)
    if depth is None:
        policy = self.layered_prefill_policy
        configured = LayeredPrefillExtensions.from_vllm_config(self.vllm_config).max_concurrent_layered_prefills
        depth = max(1, min(configured, int(policy.pipeline_parallel_size)))
        self._layered_wavefront_depth_cache = depth
        if policy.enabled:
            logger.info(
                "Layered prefill wavefront: max_concurrent_layered_prefills=%s pp=%s -> depth=%s",
                configured,
                policy.pipeline_parallel_size,
                depth,
            )
    return depth


def _layered_wavefront_state(self):
    """Ready prompts, PP stages already busy, and the slots in use.

    ``ready`` holds prompts whose previous group has come back, sorted
    most-advanced first: draining the deepest stage keeps the pipeline
    ordered and frees that stage for the prompt behind it.
    """
    policy = self.layered_prefill_policy
    ready = []
    busy_owners: set[int] = set()
    active = 0
    for request in self.running:
        if not (self._is_layered_prefill_pending(request) and self._is_layered_request_eligible(request)):
            continue
        active += 1
        if request.num_in_flight_tokens > 0:
            owner = in_flight_group_owner(policy, request)
            if owner is not None:
                busy_owners.add(owner)
            continue
        ready.append(request)
    ready.sort(key=lambda r: r.layered_prefill_group_id, reverse=True)
    return ready, busy_owners, active


def _next_layered_admission_candidate(self):
    """First waiting prompt the policy may admit, same walk as upstream."""
    for queue in (self.waiting, self.skipped_waiting):
        for request in queue:
            if not self._is_layered_request_eligible(request):
                continue
            if not request.layered_prefill_enabled:
                self.layered_prefill_policy.initialize_request(request)
            if self._is_layered_prefill_pending(request):
                return request
    return None


def _get_layered_wavefront_candidate(self):
    """Pick a layer group that keeps one prompt busy per PP stage."""
    policy = self.layered_prefill_policy
    ready, busy_owners, active = self._layered_wavefront_state()
    for request in ready:
        owner = group_owner(policy, request)
        if owner is None or owner not in busy_owners:
            return request
    if ready:
        return ready[0]
    if active >= self._layered_wavefront_depth():
        return None
    if policy.config.require_pd_mixed and self._has_unsampled_final_prefill():
        return None
    candidate = self._next_layered_admission_candidate()
    if candidate is None:
        return None
    entry_owner = group_owner(policy, candidate)
    if entry_owner is not None and entry_owner in busy_owners:
        return None
    return candidate


def _patch_scheduler(scheduler_cls) -> bool:
    required = (
        "_get_layered_prefill_candidate",
        "_attach_fused_mixed_decodes",
        "_is_layered_decode_request",
    )
    if not all(hasattr(scheduler_cls, name) for name in required):
        logger.debug("vLLM scheduler has no layered prefill; wavefront patch skipped")
        return False
    if getattr(scheduler_cls, "_layered_wavefront_patched", False):
        return True

    original_get_candidate = scheduler_cls._get_layered_prefill_candidate
    original_attach_fused = scheduler_cls._attach_fused_mixed_decodes
    original_is_decode = inspect.getattr_static(scheduler_cls, "_is_layered_decode_request").__func__

    def _get_layered_prefill_candidate(self):
        if self._layered_wavefront_depth() > 1:
            return self._get_layered_wavefront_candidate()
        return original_get_candidate(self)

    def _is_layered_decode_request(request) -> bool:
        if getattr(request, "_layered_rides_other_span", False):
            return False
        return original_is_decode(request)

    def _attach_fused_mixed_decodes(self, scheduler_output, candidate):
        plan = scheduler_output.layered_prefill_plan
        if self._layered_wavefront_depth() <= 1 or plan is None or plan.group_id != 0:
            return original_attach_fused(self, scheduler_output, candidate)
        # Group 0 recruits riders from ``running``.  A decode that already
        # holds a slot is mid-span on another prompt; hide it from that walk.
        riding = [
            request
            for request in self.running
            if request is not candidate and getattr(request, "layered_fused_decode_slot", False)
        ]
        for request in riding:
            request._layered_rides_other_span = True
        try:
            return original_attach_fused(self, scheduler_output, candidate)
        finally:
            for request in riding:
                request._layered_rides_other_span = False

    scheduler_cls._layered_wavefront_depth = _layered_wavefront_depth
    scheduler_cls._layered_wavefront_state = _layered_wavefront_state
    scheduler_cls._next_layered_admission_candidate = _next_layered_admission_candidate
    scheduler_cls._get_layered_wavefront_candidate = _get_layered_wavefront_candidate
    scheduler_cls._get_layered_prefill_candidate = _get_layered_prefill_candidate
    scheduler_cls._is_layered_decode_request = staticmethod(_is_layered_decode_request)
    scheduler_cls._attach_fused_mixed_decodes = _attach_fused_mixed_decodes
    scheduler_cls._layered_wavefront_patched = True
    return True


_patch_scheduler(_base_scheduler_cls())
