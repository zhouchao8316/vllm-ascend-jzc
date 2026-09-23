# DSV4-Flash，TP=8：chunked / layered / fuse_mixed_batch

2026-09-22 至 2026-09-23。同一输入下比较三种 prefill：chunked、layered，以及 layered 打开 `fuse_mixed_batch`。切分数取 4 和 8。表中数字是该配置全部完整 pass 的算术平均。

## chunk4

`max_num_batched_tokens=4096`，`allowed_num_groups=[4]`。chunked、layered、layered（`fuse_mixed_batch`）各 3 次 pass，均为 24/24，失败 0。

| 配置 | TTFT avg (ms) | TTFT P90 (ms) | TPOT avg (ms) | TPOT P90 (ms) | E2EL avg (ms) | E2EL P90 (ms) | 输出吞吐 (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| chunked MBT=4096 | 4550.4 | 10504.6 | 62.2 | 66.6 | 36303.4 | 44402.8 | 112.1 |
| layered G=4 | 4390.8 | 10102.2 | 63.6 | 67.8 | 36901.6 | 44405.3 | 110.3 |
| layered G=4, fuse_mixed_batch | 4496.0 | 9406.7 | 62.3 | 66.3 | 36336.8 | 42616.1 | 112.4 |

相对 chunked MBT=4096：layered G=4 的 E2EL +1.6%，吞吐 −1.6%，TTFT −3.5%。`fuse_mixed_batch` 的 E2EL +0.1%，吞吐 +0.2%，TTFT −1.2%，TTFT P90 −10.4%（9406.7 对 10504.6）。chunk4 上 `fuse_mixed_batch` 没有拉开 E2EL，尾部首 token 更短。

## chunk8

chunked 的 `max_num_batched_tokens=2048`，layered 为 `allowed_num_groups=[8]`。chunked 与 layered 各 3 次 pass；`fuse_mixed_batch` 1 次 pass。进入本表的请求都是 24/24，失败 0。

| 配置 | TTFT avg (ms) | TTFT P90 (ms) | TPOT avg (ms) | TPOT P90 (ms) | E2EL avg (ms) | E2EL P90 (ms) | 输出吞吐 (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| chunked MBT=2048 | 5356.1 | 13016.9 | 66.4 | 71.0 | 39277.6 | 49258.8 | 103.2 |
| layered G=8 | 4586.6 | 11188.6 | 63.6 | 67.5 | 37102.9 | 45591.4 | 109.1 |
| layered G=8, fuse_mixed_batch | 4639.8 | 9684.9 | 62.1 | 66.3 | 36366.2 | 42780.7 | 112.3 |

相对 chunked MBT=2048：layered G=8 的 E2EL −5.5%，吞吐 +5.7%，TTFT −14.4%。`fuse_mixed_batch` 的 E2EL −7.4%，吞吐 +8.8%，TTFT −13.4%，TTFT P90 −25.6%（9684.9 对 13016.9）。`fuse_mixed_batch` 这一行是单次测量。

chunked 的平均被 09-22 21:23 那一次 pass 拉高（E2EL 40731.8，吞吐 99.4）。另外两次 E2EL 为 38567.7 和 38533.4，平均 38550.6。layered G=8 的 37102.9 相对这个平均仍是 −3.8%。

## 配置

- 机型：8× Ascend 910B3，容器 `jzc-gjc`
- 模型：DeepSeek-V4 Flash W8A8（`--quantization ascend`，`--tokenizer-mode deepseek_v4`）
- 并行：TP=8，EP，DP=1，PP=1，`VLLM_USE_V2_MODEL_RUNNER=1`
- 图模式：`cudagraph_mode=FULL_DECODE_ONLY`。Prefill group 保持 eager；Decode 可用 `FULL_DECODE_ONLY`
- 长度：数据集 `INPUT_LEN=16380`。chat template 固定多 4 个 token，scheduler 里 `query=16384`
- 压测：aisbench stream，out=512，n=24，concurrency=8。prefix cache 关，async 关。`max_model_len=20480`，`gpu_memory_utilization=0.95`，`max_num_seqs=8`，`require_pd_mixed=false`
- 对照：chunked（MBT=4096、2048）vs layered G=4、G=8（关 chunked prefill，`max_num_batched_tokens=20480`，`allowed_num_groups=[4]` 或 `[8]`）。`fuse_mixed_batch` 在对应 layered 配置上打开

## layered 与 fuse_mixed_batch

模型 43 层。chunked 沿序列切：每个 step 跑全部层，但 prefill 最多带 `max_num_batched_tokens` 个 token。layered 沿层切：prompt 不再按 token 切开，43 层分成若干组，一个 step 只跑其中一组。G=4 是 `[0,11) [11,22) [22,33) [33,43)`。G=8 不能整除 43 层，各组层数是 6、6、6、5、5、5、5、5。一条 prefill 要走完所有组才产生第一个 token。

layered 在 mixed 步里把 Decode 和 Prefill 拆开跑。已经在 decode 的请求先走 FULL decode 图，这一步就采样；同一条 prefill 再 eager 跑当前这一组层。纯 decode 步没有 prefill，只走这张图。

`fuse_mixed_batch` 仍按同样的层组切。差别只在 mixed 步：Decode token 和 Prefill token 放进同一次 eager forward，一起跑当前这一组，不再先跑 decode 图。这条 decode 要跟着这条 prefill 从第 0 组骑到最后一组，每个组都重放同一个 decode token，只在最后一组采样。中途才到达的 decode 会错过前面的层，因此等到下一条从第 0 组开始的 prefill。没有 prefill 的纯 decode 步与 layered 相同，仍走 FULL 图。

layered 与 `fuse_mixed_batch` 都关 chunked prefill，profile 按整段 prompt 估峰值激活，KV 比 chunked 小。表中对比的是整条实现，不是同一 KV 容量下的算子。

输入长度能被 MBT 整除，并不保证 prefill 恰好分成 4 段或 8 段。mixed 步里的 decode 占 token 预算，prefill 可能多出一段。

没有完整 24/24 的 pass 不进入平均。2026-09-22 22:18 之后，chunk4 的后两次 pass 和当晚全部 `fuse_mixed_batch` 在启动检查时退出：空闲显存低于 0.95 要求的 57.91 GiB，服务没有起来。layered G=8（`fuse_mixed_batch`）只有一次 pass，表中就是这一次。

## 实验数据

每次 pass 都是 24/24，失败 0。平均表由下面这些数算出。

### chunk4

| 配置 | 时间 | TTFT avg | TTFT P90 | TPOT avg | TPOT P90 | E2EL avg | E2EL P90 | 吞吐 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| chunked MBT=4096 | 09-22 22:04 | 4543.5 | 10491.5 | 61.4 | 65.8 | 35922.9 | 43998.0 | 113.3 |
| chunked MBT=4096 | 09-23 09:21 | 4551.5 | 10510.9 | 62.1 | 66.4 | 36263.6 | 44294.9 | 112.3 |
| chunked MBT=4096 | 09-23 10:30 | 4556.1 | 10511.4 | 63.0 | 67.6 | 36723.7 | 44915.4 | 110.8 |
| layered G=4 | 09-22 22:18 | 4335.8 | 10012.2 | 63.0 | 66.8 | 36537.3 | 43997.3 | 111.4 |
| layered G=4 | 09-23 09:35 | 4484.3 | 10263.1 | 64.5 | 69.2 | 37431.6 | 45271.9 | 108.8 |
| layered G=4 | 09-23 10:02 | 4352.3 | 10031.3 | 63.4 | 67.4 | 36735.9 | 43946.6 | 110.8 |
| layered G=4, fuse_mixed_batch | 09-23 09:48 | 4514.1 | 9417.7 | 63.0 | 67.2 | 36695.2 | 42622.8 | 111.2 |
| layered G=4, fuse_mixed_batch | 09-23 10:17 | 4488.5 | 9409.3 | 61.7 | 65.6 | 36014.2 | 42528.8 | 113.4 |
| layered G=4, fuse_mixed_batch | 09-23 10:45 | 4485.3 | 9393.2 | 62.3 | 66.1 | 36300.9 | 42696.6 | 112.5 |

09-23 10:45 这次 `fuse_mixed_batch` 在 24/24 记完之后，worker 退出，引擎报 `Executor failed`。日志里没有 out of memory。失败数是 0，这一次计入平均。

### chunk8

| 配置 | 时间 | TTFT avg | TTFT P90 | TPOT avg | TPOT P90 | E2EL avg | E2EL P90 | 吞吐 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| chunked MBT=2048 | 09-22 20:39 | 5204.6 | 12679.9 | 65.3 | 69.8 | 38567.7 | 48316.4 | 105.0 |
| chunked MBT=2048 | 09-22 21:23 | 5701.4 | 13836.3 | 68.6 | 73.3 | 40731.8 | 51269.3 | 99.4 |
| chunked MBT=2048 | 09-22 21:37 | 5162.3 | 12534.6 | 65.3 | 69.8 | 38533.4 | 48190.6 | 105.1 |
| layered G=8 | 09-22 20:54 | 4535.8 | 11114.1 | 63.1 | 66.9 | 36775.0 | 45142.0 | 110.1 |
| layered G=8 | 09-22 21:09 | 4578.8 | 11224.2 | 64.1 | 67.9 | 37341.1 | 45818.3 | 108.4 |
| layered G=8 | 09-22 21:52 | 4645.3 | 11227.4 | 63.7 | 67.8 | 37192.6 | 45814.0 | 108.8 |
| layered G=8, fuse_mixed_batch | 09-23 11:02 | 4639.8 | 9684.9 | 62.1 | 66.3 | 36366.2 | 42780.7 | 112.3 |
