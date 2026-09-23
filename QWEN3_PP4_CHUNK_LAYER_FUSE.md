# Qwen3-30B-A3B，PP=4：chunked / layered / fuse_mixed_batch

2026-09-21 与 2026-09-23。PP=4、TP=1，比较三种 prefill：chunked、layered，以及 layered 打开 `fuse_mixed_batch`。

chunked 按 token 切，一个 step 跑当前 stage 的全部层。layered 按层切，一个 step 只跑一组层；mixed 步里 decode 先单独跑完再跑这一组 prefill。`fuse_mixed_batch` 仍按层切，mixed 步里 decode 和 prefill 进同一次 forward。

PP=4 的两张表是 chunked 与 layered 打开 `fuse_mixed_batch`。正号表示大于同一对照的 chunked。这两趟都是单次。

## PP=4，in=32768，G=8

模型 48 层，8 组，每组 6 层。每个 PP stage 12 层，放 2 组，组不跨 stage。chunked 的 `max_num_batched_tokens=4096`。out=512，n=24，concurrency=8。24/24。

| 配置 | TTFT avg (ms) | E2EL avg (ms) | ITL median (ms) | 输出吞吐 (tok/s) |
|---|---:|---:|---:|---:|
| chunked MBT=4096 | 5626.7 | 44690.9 | 61.9 | 91.45 |
| layered G=8, fuse_mixed_batch | 16536.7 | 70970.9 | 61.9 | 57.67 |

## PP=4，in=4096，G=4

四组 `[0,12) [12,24) [24,36) [36,48)`，一组一个 PP stage。chunked 的 `max_num_batched_tokens=8192`，一个 prefill step 覆盖全部 48 层。out=256，n=32，concurrency=8。32/32。

| 配置 | TTFT avg (ms) | TTFT P90 (ms) | TPOT avg (ms) | E2EL avg (ms) | E2EL P90 (ms) | 输出吞吐 (tok/s) |
|---|---:|---:|---:|---:|---:|---:|
| chunked MBT=8192 | 1108.1 | 2485.0 | 32.1 | 9303.6 | 11029.4 | 219.9 |
| layered G=4, fuse_mixed_batch | 1306.3 | 3387.4 | 37.8 | 10954.6 | 13239.3 | 186.4 |

## 配置

- 机型：8× Ascend 910B3。PP=4 用卡 0–3
- 模型：Qwen3-30B-A3B，48 层。V2，`VLLM_USE_V2_MODEL_RUNNER=1`
- 图模式：`FULL_DECODE_ONLY`。均匀 decode 进图。Prefill 和 mixed 步保持 eager
- prefix cache 关，async 关
- chunked 用表里的 `max_num_batched_tokens`。layered 关掉 chunked prefill
- `fuse_mixed_batch` 在对应 layered 行上打开。`max_concurrent_layered_prefills` 默认关闭，PP=4 两趟都没开

## 三种 prefill 在 PP 上怎么切

chunked 沿 token 切。每个 step 跑当前 stage 的层，prefill 最多带 `max_num_batched_tokens` 个 token。

layered 沿层切。48 层分成若干组，一组不跨 stage，一个 step 只跑一组。激活按 stage 顺序往下传。一条 prefill 走完所有组才出第一个 token。mixed 步里，已经在 decode 的请求先走 FULL decode 图并在这一步采样，然后这条 prefill 再 eager 跑当前组。

`fuse_mixed_batch` 用同样的层组。差别在 mixed 步：decode token 和 prefill token 放进同一次 eager forward。非最后一个 rank 把这次输出整段发给下一 rank，不再拆成 decode 行和 prefill 行。这条 decode 从第 0 组骑到最后一组，只在最后一组采样。最后一组在最后一个 PP rank。没有 prefill 的纯 decode 仍走 FULL 图。

一条 prefill 还没走完时，下一条 layered prefill 不会被放进来。`max_concurrent_layered_prefills` 默认关闭，上面两趟都没开。

layered 和 `fuse_mixed_batch` 都按整段 prompt 估峰值激活，KV 比 chunked 小。表里比的是整条实现。

PP=2、TP=2、4 条 prompt 的输出对照通过，0 处不一致。另一次 PP=2、TP=1 启动时退出：空闲显存 36.93 GiB，低于 51.81 GiB。PP=4、TP=2 的输出对照没有跑完。
