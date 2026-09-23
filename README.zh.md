<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
    <img alt="vllm-ascend" src="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM Ascend Plugin
</h3>

<p align="center">
| <a href="https://www.hiascend.com/en/"><b>关于昇腾</b></a> | <a href="https://docs.vllm.ai/projects/ascend/en/latest/"><b>官方文档</b></a> | <a href="https://slack.vllm.ai"><b>#sig-ascend</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support"><b>用户论坛</b></a> | <a href="https://tinyurl.com/vllm-ascend-meeting"><b>社区例会</b></a> |
</p>

<p align="center">
<a href="README.md"><b>English</b></a> | <a><b>中文</b></a>
</p>

---

## 本仓库：Model Runner V2 + Layered Prefill

基于 [`gjc0824/vllm-ascend`](https://github.com/gjc0824/vllm-ascend) `layer_prefill`。在 Model Runner V2 上启用 layered prefill；原分支拒绝该组合。

1. **V2 门禁**：去掉仅允许 V1 runner 的检查。layered 默认关闭。
2. **D/P 子批**：Prefill 使用独立 `AscendInputBuffers`；同一步先 D 后 P（`attn_metadata` 为实例状态）；合并 sampling。
3. **MoE 层偏移**：P 执行 `[group_start, group_end)`，将偏移写入 `forward_context.moe_layer_index`。
4. **hybrid KV**：`num_prefill_lookahead is None` 视为 0；记录 D/P 实际使用的 CUDAGraph 模式。
5. **V2 允许 prefix cache 与 async scheduling**（下表实验均关闭）。
6. **ACL 507011**：`FULL_DECODE_ONLY` 会为 FIA 填充 dummy 行。DSA-CP 须使用真实 `num_reqs`，并将 padding 行的 `seq_lens` / block table 置零，否则 MTE 越界。
7. **PP>1**：每个 scheduler step 一收一发；D/P 行放入同一 `IntermediateTensors`；采样共用一个 PPHandler（中间 P group 不采样）。layer group 须与 PP stage 对齐。

代码：[`vllm_ascend/worker/v2/model_runner.py`](vllm_ascend/worker/v2/model_runner.py)、[`layered_prefill.py`](vllm_ascend/worker/v2/layered_prefill.py)、[`attn_utils.py`](vllm_ascend/worker/v2/attn_utils.py)、[`dsa_cp.py`](vllm_ascend/attention/context_parallel/dsa_cp.py)。

### DSV4-Flash，TP=8，PP=1（2026-09-20）

- 机型：8× Ascend 910B3
- 模型：DeepSeek-V4 Flash W8A8（`--quantization ascend`，`--tokenizer-mode deepseek_v4`）
- 并行：TP=8，EP，DP=1，PP=1，`VLLM_USE_V2_MODEL_RUNNER=1`
- 图模式：`cudagraph_mode=FULL_DECODE_ONLY`
- 压测：aisbench GSM8K stream，in=4096 / out=256 / n=16 / concurrency=8；prefix cache 关，async 关
- 对照：chunked（MBT=256、512）vs layered G=8（`allowed_num_groups=[8]`，`max_num_batched_tokens=16392`）；`max_num_seqs=8`。4k prompt 上 MBT=512 与 G=8 均为约 8 步 prefill；MBT=256 约 16 步。G=8 为单次测量，512 对照复用该结果。

| 臂 | 请求 | TTFT avg (ms) | TPOT avg (ms) | E2EL avg (ms) | 输出吞吐 (tok/s) |
|---|:---:|---:|---:|---:|---:|
| chunked MBT=256 | 16/16 | 11726.4 | 114.2 | 40844.7 | 48.6 |
| chunked MBT=512 | 16/16 | 6848.3 | 85.7 | 28694.7 | 69.8 |
| layered G=8 | 16/16（复用） | 2488.9 | 56.2 | 16832.6 | 117.1 |

相对 chunked MBT=512（等步）：G=8 TTFT ≈ 2.8×，TPOT −29.5 ms，E2EL 与吞吐 ≈ 1.7×。MBT=256 步数多一倍，不作等步对照。单次 pass；无 ACL 507011。

16k 输入（query=16384，out=512，n=24）的 chunked / layered / `fuse_mixed_batch` 结果见 [DSV4_TP8_16K_TOKEN_LAYER_FUSE.md](DSV4_TP8_16K_TOKEN_LAYER_FUSE.md)。

### Pipeline parallel

V2 layered 允许 `PP>1`。Prefill group 保持 eager；Decode 可用 `FULL_DECODE_ONLY`。仍拒绝：DP>1、DBO、SP、EPLB、PCP/DCP（`context_parallel_size>1`）、KV offload。

Qwen3-30B-A3B，V2，TP=1：

| 配置 | 测试 | 日期 | 结果 |
|---|---|---|---|
| PP=2 | LLM smoke（P-only、P+D 混合、frontier） | 2026-09-17、09-18 | 通过 |
| PP=4 | LLM smoke，含 4-hop frontier | 2026-09-17 | 通过 |
| PP=4 | aisbench in=4096 / out=256 / n=32 / c=8 | 2026-09-17 | chunked、layered G=1、layered G=4 完成 |

| 臂 | TTFT avg (ms) | TPOT avg (ms) | 输出吞吐 (tok/s) |
|---|---:|---:|---:|
| chunked | 1060.0 | 33.8 | 211.1 |
| layered G=1 | 1322.1 | 42.4 | 166.6 |
| layered G=4 | 1319.3 | 39.8 | 177.0 |

layered G=4：`mix_frac=1.0`，124 个 layered step。V1 对 PP=2/4 与 TP=2+PP=2+EP=2 有受限文本核对（eager；Decode graph + Prefill eager）。

2026-09-20，卡 4–7，Qwen3-30B-A3B V2 PP=2 TP=2，`cudagraph_mode=FULL_DECODE_ONLY`，`max_num_seqs=1`，4 条 greedy：layered G=2 与同模式 FDO baseline 逐 token 一致，也与此前 eager PP=2 TP=2 baseline 一致。graph capture 成功，无 ACL 507011。本次未覆盖 FDO 下的 P+D 混合。

上表 DSV4-Flash TP8+EP 为 PP=1。DSV4 的 PP=2/4 未跑。未覆盖：bubble 时间、cancel/preempt/finish、logits/KV。PP=4 上 graph 与 eager 的 Prefill token 可能不一致（关闭 layered 时同样存在）。`max_num_seqs>1` 时，中段 token 可能与 layered-off 不一致；单请求 TP=2 可逐 token 对齐。

---
*最新消息* 🔥

- [2026/08] 我们发布了新的正式版本 [v0.23.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.23.0)! 请按照[官方指南](https://docs.vllm.ai/projects/ascend/en/v0.23.0/)开始在 Ascend 上部署 vLLM Ascend Plugin。
- [2026/05] 我们发布了新的正式版本 [v0.18.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.18.0)! 请按照[官方指南](https://docs.vllm.ai/projects/ascend/en/v0.18.0/)开始在Ascend上部署vLLM Ascend Plugin。
- [2026/02] 我们发布了新的正式版本 [v0.13.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.13.0)! 请按照[官方指南](https://docs.vllm.ai/projects/ascend/en/v0.13.0/)开始在Ascend上部署vLLM Ascend Plugin。

<details>
<summary>更多内容</summary>

- [2025/12] 我们发布了新的正式版本 [v0.11.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.11.0)! 请按照[官方指南](https://docs.vllm.ai/projects/ascend/en/v0.11.0/)开始在Ascend上部署vLLM Ascend Plugin。
- [2025/09] 我们发布了新的正式版本 [v0.9.1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.9.1)! 请按照[官方指南](https://docs.vllm.ai/projects/ascend/en/v0.9.1/tutorials/large_scale_ep.html)开始在Ascend上部署大型专家并行 (EP)。
- [2025/08] 我们与vLLM和腾讯合作举办了[vLLM北京Meetup](https://mp.weixin.qq.com/s/7n8OYNrCC_I9SJaybHA_-Q)，！请查阅[活动幻灯片](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF)。
- [2025/06] [用户案例](https://docs.vllm.ai/projects/ascend/en/latest/community/user_stories/index.html)现已上线！展示了LLaMA-Factory/verl/TRL/GPUStack等用户案例，展示了vLLM Ascend如何帮助昇腾用户在模型微调、评估、强化学习 (RL) 以及部署等场景中提升体验。
- [2025/06] [贡献者](https://docs.vllm.ai/projects/ascend/en/latest/community/contributors.html)页面现已上线！所有的贡献都值得被记录，感谢所有的贡献者。
- [2025/05] 我们发布了首个正式版本 [v0.7.3](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.3)！我们与 vLLM 社区合作发布了一篇博客文章，分享了我们的实践：[Introducing vLLM Hardware Plugin, Best Practice from Ascend NPU](https://blog.vllm.ai/2025/05/12/hardware-plugin.html)。
- [2025/03] 我们和vLLM团队举办了[vLLM Beijing Meetup](https://mp.weixin.qq.com/s/CGDuMoB301Uytnrkc2oyjg)! 请查阅[活动幻灯片](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF).
- [2025/02] vLLM社区正式创建了[vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)仓库，让vLLM可以无缝运行在Ascend NPU。
- [2024/12] 我们正在与 vLLM 社区合作，以支持 [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162).

</details>

---

## 总览

vLLM 昇腾插件 (`vllm-ascend`) 是一个由社区维护的让vLLM在Ascend NPU无缝运行的后端插件。

此插件是 vLLM 社区中支持昇腾后端的推荐方式。它遵循[[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162)所述原则：通过解耦的方式提供了vLLM对Ascend NPU的支持。

使用 vLLM 昇腾插件，可以让类Transformer、混合专家(MOE)、嵌入、多模态等流行的大语言模型在 Ascend NPU 上无缝运行。

支持的模型详细信息，请参考[模型支持列表](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_models.html)。

## 准备

- 硬件：Atlas 800I A2 Inference系列、Atlas A2 Training系列、Atlas 800I A3 Inference系列、Atlas A3 Training系列、Atlas 300I Duo（实验性支持）
- 操作系统：Linux
- 软件：
    - Python >= 3.10, < 3.13
    - CANN == 9.1.0 (Ascend HDK 版本详见 [版本说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/releasenote/9.1.0/release-notes.md))
    - PyTorch == 2.10.0, TorchNPU == 2.10.0.post4
    - vLLM (与vllm-ascend版本一致)

## 访问昇腾NPU

如果您需要访问昇腾NPU算力资源进行开发或测试，请进入华为HiDevLab平台的[HiDevLab-在线开发](https://hidevlab.huawei.com/online-develop-intro) 页面申请并使用算力。

## 开始使用

推荐您使用以下版本快速开始使用：

| Version    | Release type | Doc                                  |
|------------|--------------|--------------------------------------|
| v0.23.0 | 最新正式/稳定版本 | 请查看[快速开始](https://docs.vllm.ai/projects/ascend/en/v0.23.0/quick_start.html)和[安装指南](https://docs.vllm.ai/projects/ascend/en/v0.23.0/installation.html)了解更多 |

## 分支策略

vllm-ascend有主干分支和开发分支。

- **main**: 主干分支，与vLLM的主干分支对应，并通过昇腾CI持续进行质量看护。
- **releases/vX.Y.Z**: 开发分支，随vLLM部分新版本发布而创建，比如`releases/v0.13.0`是vllm-ascend针对vLLM `v0.13.0` 版本的开发分支。

下面是维护中的分支：

| 分支              | 状态         | 备注                  |
|------------------|--------------|----------------------|
| main             | Maintained   | 基于vLLM main分支和vLLM最新版本（v0.27.1）CI看护   |
| releases/v0.13.0 | Maintained   | 只允许Bug修复，不会再发布新版本 |
| releases/v0.18.0 | Maintained   | 基于vLLM v0.18.0版本CI看护 |
| releases/v0.23.0 | Maintained   | 基于vLLM v0.23.0版本CI看护 |
| rfc/<feature-name> | Maintained   | 为协作创建的[特性分支](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html#feature-branches) |

请参阅[版本策略](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html)了解更多详细信息。

## 贡献

请参考[CONTRIBUTING](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/contribution/index.html)文档了解更多关于开发环境搭建、功能测试以及 PR 提交规范的信息。

我们欢迎并重视任何形式的贡献与合作：

- 请通过[Issue](https://github.com/vllm-project/vllm-ascend/issues)来告知我们您遇到的任何Bug。
- 请通过[用户论坛](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support)来交流使用问题和寻求帮助。

## 社区例会

- vLLM Ascend 每周社区例会: <https://tinyurl.com/vllm-ascend-meeting>
- 每周三下午，15:00 - 16:00 (UTC+8, [查看您的时区](https://dateful.com/convert/gmt8?t=15))

## 许可证

Apache 许可证 2.0，如 [LICENSE](./LICENSE) 文件中所示。
