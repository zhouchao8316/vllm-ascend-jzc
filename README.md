<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
    <img alt="vllm-ascend" src="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM Ascend Plugin
</h3>

<div align="center">

[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/vllm-project/vllm-ascend)

</div>

<p align="center">
| <a href="https://www.hiascend.com/en/"><b>About Ascend</b></a> | <a href="https://docs.vllm.ai/projects/ascend/en/latest/"><b>Documentation</b></a> | <a href="https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/"><b>Support Matrix</b></a> | <a href="https://slack.vllm.ai"><b>#SIG-Ascend</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support"><b>Users Forum</b></a> | <a href="https://tinyurl.com/vllm-ascend-meeting"><b>Weekly Meeting</b></a> |
</p>

<p align="center">
<a ><b>English</b></a> | <a href="README.zh.md"><b>中文</b></a>
</p>

---

## This fork: Model Runner V2 + Layered Prefill

Based on [`gjc0824/vllm-ascend`](https://github.com/gjc0824/vllm-ascend) `layer_prefill`. Layered prefill on Model Runner V2; the original branch rejected that combination.

1. **V2 platform gate**: drop the V1-only check. Layered remains default-off.
2. **D/P sub-batches**: a separate `AscendInputBuffers` for Prefill; D then P in one step (`attn_metadata` is instance state); merge sampling.
3. **MoE layer offset**: P runs `[group_start, group_end)`; write the offset to `forward_context.moe_layer_index`.
4. **Hybrid KV**: `num_prefill_lookahead is None` is treated as 0. Log the CUDAGraph mode used by D and P.
5. **Prefix cache and async scheduling** are allowed on V2 (off in the numbers below).
6. **ACL 507011**: `FULL_DECODE_ONLY` pads dummy FIA rows. DSA-CP must use the real `num_reqs` and zero padded `seq_lens` / block tables; otherwise MTE reads out of range.
7. **PP>1**: one recv/send per scheduler step; D/P rows in one `IntermediateTensors` payload; sampled tokens share one PPHandler slot (intermediate P groups excluded). Layer groups must align with PP stages.

Code: [`vllm_ascend/worker/v2/model_runner.py`](vllm_ascend/worker/v2/model_runner.py), [`layered_prefill.py`](vllm_ascend/worker/v2/layered_prefill.py), [`attn_utils.py`](vllm_ascend/worker/v2/attn_utils.py), [`dsa_cp.py`](vllm_ascend/attention/context_parallel/dsa_cp.py).

### DSV4-Flash, TP=8, PP=1 (2026-09-20)

- Host: 8× Ascend 910B3
- Model: DeepSeek-V4 Flash W8A8 (`--quantization ascend`, `--tokenizer-mode deepseek_v4`)
- Parallel: TP=8, EP, DP=1, PP=1, `VLLM_USE_V2_MODEL_RUNNER=1`
- Graph: `cudagraph_mode=FULL_DECODE_ONLY`
- Bench: aisbench GSM8K stream, in=4096 / out=256 / n=16 / concurrency=8; prefix cache off, async off
- Setups: chunked (MBT=256, 512) vs layered G=8 (`allowed_num_groups=[8]`, `max_num_batched_tokens=16392`); `max_num_seqs=8`. On a 4k prompt, MBT=512 and G=8 are both ~8 prefill steps; MBT=256 is ~16. G=8 is a single earlier run, reused for the 512 compare.

| Setup | Requests | TTFT avg (ms) | TPOT avg (ms) | E2EL avg (ms) | Output tok/s |
|---|:---:|---:|---:|---:|---:|
| chunked MBT=256 | 16/16 | 11726.4 | 114.2 | 40844.7 | 48.6 |
| chunked MBT=512 | 16/16 | 6848.3 | 85.7 | 28694.7 | 69.8 |
| layered G=8 | 16/16 (reused) | 2488.9 | 56.2 | 16832.6 | 117.1 |

Vs chunked MBT=512 (same step count): G=8 TTFT ≈ 2.8×, TPOT −29.5 ms, E2EL and throughput ≈ 1.7×. MBT=256 uses twice as many steps and is not an iso-step baseline. One pass; no ACL 507011.

### Pipeline parallel

`PP>1` is accepted with V2 layered. Prefill groups stay eager; Decode may use `FULL_DECODE_ONLY`. Still rejected: DP>1, DBO, SP, EPLB, PCP/DCP (`context_parallel_size>1`), KV offload.

Qwen3-30B-A3B, V2, TP=1:

| Config | Test | Date | Result |
|---|---|---|---|
| PP=2 | LLM smoke (P-only, mixed P+D, frontier) | 2026-09-17, 09-18 | pass |
| PP=4 | LLM smoke, including 4-hop frontier | 2026-09-17 | pass |
| PP=4 | aisbench in=4096 / out=256 / n=32 / c=8 | 2026-09-17 | chunked, layered G=1, layered G=4 completed |

| Setup | TTFT avg (ms) | TPOT avg (ms) | Output tok/s |
|---|---:|---:|---:|
| chunked | 1060.0 | 33.8 | 211.1 |
| layered G=1 | 1322.1 | 42.4 | 166.6 |
| layered G=4 | 1319.3 | 39.8 | 177.0 |

Layered G=4: `mix_frac=1.0`, 124 layered steps. V1 has restricted-text checks for PP=2/4 and TP=2+PP=2+EP=2 (eager; Decode graph + Prefill eager).

2026-09-20, cards 4–7, Qwen3-30B-A3B V2 PP=2 TP=2, `cudagraph_mode=FULL_DECODE_ONLY`, `max_num_seqs=1`, four greedy prompts: layered G=2 matched the FDO baseline token-for-token, and matched the earlier eager PP=2 TP=2 baseline. Graph capture succeeded; no ACL 507011. Mixed P+D under FDO was not in this pass.

The DSV4-Flash TP8+EP table is PP=1. DSV4 with PP=2/4 was not run. Not covered: bubble time, cancel/preempt/finish, logits/KV. On PP=4, Prefill tokens can differ between graph and eager (also without layered). With `max_num_seqs>1`, a mid-sequence token may differ from layered-off; a single request at TP=2 matches token-for-token.

---
*Latest News* 🔥

- [2026/08] We released the new official version [v0.23.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.23.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.23.0/) to start using vLLM Ascend Plugin on Ascend.
- [2026/05] We released the new official version [v0.18.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.18.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.18.0/) to start using vLLM Ascend Plugin on Ascend.
- [2026/02] We released the new official version [v0.13.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.13.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.13.0/) to start using vLLM Ascend Plugin on Ascend.

<details>
<summary>More</summary>

- [2025/12] We released the new official version [v0.11.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.11.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.11.0/) to start using vLLM Ascend Plugin on Ascend.
- [2025/09] We released the new official version [v0.9.1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.9.1)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.9.1/tutorials/large_scale_ep.html) to start deploying large-scale Expert Parallelism (EP) on Ascend.
- [2025/08] We hosted the [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/7n8OYNrCC_I9SJaybHA_-Q) with vLLM and Tencent! Please find the [meetup slides](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF).
- [2025/06] [User stories](https://docs.vllm.ai/projects/ascend/en/latest/community/user_stories/index.html) page is now live! It kicks off with LLaMA-Factory/verl/TRL/GPUStack to demonstrate how vLLM Ascend assists Ascend users in enhancing their experience across fine-tuning, evaluation, reinforcement learning (RL), and deployment scenarios.
- [2025/06] [Contributors](https://docs.vllm.ai/projects/ascend/en/latest/community/contributors.html) page is now live! All contributions deserve to be recorded, thanks for all contributors.
- [2025/05] We've released the first official version [v0.7.3](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.3)! We collaborated with the vLLM community to publish a blog post sharing our practice: [Introducing vLLM Hardware Plugin, Best Practice from Ascend NPU](https://blog.vllm.ai/2025/05/12/hardware-plugin.html).
- [2025/03] We hosted the [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/VtxO9WXa5fC-mKqlxNUJUQ) with vLLM team! Please find the [meetup slides](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF).
- [2025/02] vLLM community officially created [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) repo for running vLLM seamlessly on the Ascend NPU.
- [2024/12] We are working with the vLLM community to support [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162).

</details>

---

## Overview

vLLM Ascend (`vllm-ascend`) is a community maintained hardware plugin for running vLLM seamlessly on the Ascend NPU.

It is the recommended approach for supporting the Ascend backend within the vLLM community. It adheres to the principles outlined in the [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162), providing a hardware-pluggable interface that decouples the integration of the Ascend NPU with vLLM.

By using vLLM Ascend plugin, popular open-source models, including Transformer-like, Mixture-of-Experts (MoE), Embedding, Multi-modal LLMs can run seamlessly on the Ascend NPU.

For detailed information on supported models and features, please refer to the [support matrix](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/).

## Prerequisites

- Hardware: Atlas 800I A2 Inference series, Atlas A2 Training series, Atlas 800I A3 Inference series, Atlas A3 Training series, Atlas 300I Duo (Experimental)
- OS: Linux
- Software:
    - Python >= 3.10, < 3.13
    - CANN == 9.1.0 (For Ascend HDK version, please refer to the [Release Notes](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/releasenote/9.1.0/release-notes.md))
    - PyTorch == 2.10.0, TorchNPU == 2.10.0.post4
    - vLLM (the same version as vllm-ascend)

## Accessing Ascend NPU

If you need to access Ascend NPU computing resources for development or testing, please visit the [HiDevLab - Online Development](https://hidevlab.huawei.com/online-develop-intro) page on the Huawei HiDevLab platform to apply for and use them.

## Getting Started

Please use the following recommended versions to get started quickly:

| Version    | Release type | Doc                                  |
|------------|--------------|--------------------------------------|
| v0.23.0 | Latest stable version | See [QuickStart](https://docs.vllm.ai/projects/ascend/en/v0.23.0/quick_start.html) and [Installation](https://docs.vllm.ai/projects/ascend/en/v0.23.0/installation.html) for more details |

## Branch

vllm-ascend has a main branch and a dev branch.

- **main**: main branch, corresponds to the vLLM main branch, and is continuously monitored for quality through Ascend CI.
- **releases/vX.Y.Z**: development branch, created alongside new releases of vLLM. For example, `releases/v0.13.0` is the dev branch for vLLM `v0.13.0` version.

Below are the maintained branches:

| Branch           | Status       | Note                                 |
|------------------|--------------|--------------------------------------|
| main             | Maintained   | CI commitment for vLLM main branch and vLLM v0.27.1 tag |
| releases/v0.13.0 | Maintained   | Only bug fixes are allowed, and no new release tags anymore. |
| releases/v0.18.0 | Maintained   | CI commitment for vLLM 0.18.0 version |
| releases/v0.23.0 | Maintained   | CI commitment for vLLM 0.23.0 version |
| rfc/<feature-name> | Maintained   | [Feature branches](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html#feature-branches) for collaboration |

Please refer to [Versioning policy](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html) for more details.

## Contributing

See [CONTRIBUTING](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/contribution/index.html) for more details, which is a step-by-step guide to help you set up the development environment, build and test.

We welcome and value any contributions and collaborations:

- Please let us know if you encounter a bug by [filing an issue](https://github.com/vllm-project/vllm-ascend/issues)
- Please use [User forum](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support) for usage questions and help.

## Weekly Meeting

- vLLM Ascend Weekly Meeting: <https://tinyurl.com/vllm-ascend-meeting>
- Wednesday, 15:00 - 16:00 (UTC+8, [Convert to your timezone](https://dateful.com/convert/gmt8?t=15))

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.
