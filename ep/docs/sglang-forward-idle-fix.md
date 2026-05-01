# UCCL EP — SGLang `launch_server` `forward_idle` Investigation

## Status

| Component | Status | Notes |
|-----------|--------|-------|
| Bug A: host-side asserts reject empty input | **FIXED** in this branch | All `_ptr != 0` and `num_tokens > 0` checks loosened in `uccl_ep.cc`, mirroring DeepEP |
| Eager unit tests for empty input | **PASSING** | `test_intranode.py` and `test_low_latency.py` `run_empty_input_rank_test` (edges + single-producer scenarios) |
| `launch_server` with `--deepep-mode normal` | **WORKING** | SGLang auto-disables CUDA graphs in this mode; verified end-to-end with curl on H200 |
| `launch_server` with `--deepep-mode auto` (CUDA graphs ON) | **HANGS** | Kernel-level desync. Bug B below. |
| `launch_server` with `--deepep-mode auto --disable-cuda-graph` | **STILL HANGS** | Confirms bug is NOT CUDA-graph specific. Eager LL kernels also fail under asymmetric load. |

## Branch State

Branch `fix/sglang-forward-idle-empty-input` on `github.com/shawlleyw/uccl`. 21+ commits.

| File | What changed | Purpose |
|------|--------------|---------|
| `ep/src/uccl_ep.cc` | Removed `_ptr != 0` asserts in 6 host wrappers; loosened `num_tokens > 0` to `>= 0`; added `cudaGetLastError()` after intranode kernel launches | Bug A fix |
| `ep/src/internode_ll.cu` | Added diagnostic `printf` calls in dispatch SEND/RECV and combine RECV spin loops | Diagnostic only — should be reverted before merging the host-side fix to main |
| `ep/bench/test_intranode.py` | Added `run_empty_input_rank_test()` with `edges` + `single-producer` scenarios | Regression test for Bug A (eager mode) |
| `ep/bench/test_low_latency.py` | Same + `run_cuda_graph_asymmetric_test()` (CUDA-graph wrapper that doesn't trigger Bug B) | Regression tests; CUDA-graph repro is partial |
| `ep/docs/sglang-forward-idle-fix.md` | This document | Investigation log |

---

# Bug A: Host-side asserts reject empty input

## Symptom

`launch_server` with heterogeneous DP traffic raised:
```
RuntimeError: DeepEP error: CPU recv timeout
```
After ~100s. Root cause: empty rank's `EP_HOST_ASSERT(topk_idx_ptr != 0)` (or one of the other pointer-null asserts) fired because PyTorch returns `data_ptr() == 0` for zero-byte CUDA allocations on this allocator. The empty rank's kernel never launched. Other ranks waited at `barrier_block` for the empty rank, eventually timing out.

## Verification (empirical)

Tested on H200 inside Modal:
```
torch.empty((0, 4),    dtype=bool,        device='cuda').data_ptr() = 0
torch.empty((0, 4),    dtype=int64,       device='cuda').data_ptr() = 0
torch.empty((0, 4096), dtype=bfloat16,    device='cuda').data_ptr() = 0
torch.empty((0, 4),    dtype=float32,     device='cuda').data_ptr() = 0
```
The CUDA caching allocator returns 0 for every zero-byte allocation regardless of dtype.

## Fix applied

Removed all `_ptr != 0` checks in 6 host wrappers in `ep/src/uccl_ep.cc`, mirroring DeepEP's design (DeepEP takes `at::Tensor` at the C++ boundary and asserts on tensor *properties*, not raw pointers — properties survive zero-row tensors). Property/shape asserts (`num_tokens >= 0`, `num_experts > 0`, `hidden > 0`, shape divisibility) preserved.

Affected wrappers (line numbers approximate, will shift slightly across commits):
- `get_dispatch_layout` (~`:521`) — input layout for dispatch
- `intranode_prepare` (~`:640`) — intranode notify_dispatch wrapper
- `intranode_dispatch` (~`:732`) — intranode dispatch wrapper
- `intranode_combine` (~`:823`) — intranode combine wrapper
- `low_latency_dispatch` (~`:1209`) — LL dispatch wrapper
- `low_latency_combine` (~`:1311`) — LL combine wrapper

Internode-normal asserts at `uccl_ep.cc:882` and `:985` left unchanged (out of scope).

`cudaGetLastError()` checks added after each of the 5 intranode kernel launch sites in `uccl_ep.cc`, mirroring the existing pattern in the LL host wrappers. This converts silent kernel-launch failures into logged errors.

## Regression tests

Both eager unit tests **PASS** on H200, confirming Bug A is fixed.

| Test | File | Function |
|------|------|----------|
| Intranode empty-input | `ep/bench/test_intranode.py:269` | `run_empty_input_rank_test()` calls `_run_empty_input_scenario` with two configs: `empty_ranks={0, num_ranks-1}` (edges), `empty_ranks={1, ..., num_ranks-1}` (single-producer) |
| LL empty-input | `ep/bench/test_low_latency.py:140` | Same two scenarios via `_run_empty_input_scenario_ll` |
| LL CUDA-graph asymmetric | `ep/bench/test_low_latency.py:313` | `run_cuda_graph_asymmetric_test()` wraps LL dispatch+combine in captured `torch.cuda.graph` and replays with asymmetric `topk_idx`. Passes — meaning CUDA-graph alone with asymmetric content does NOT reproduce Bug B in isolation. |

---

# Bug B: Kernel-level rank desync under asymmetric load

## Status

NOT fixed. Mis-diagnosed multiple times during investigation. The current best understanding is that the LL kernel's 2-buffer toggle design races when ranks drift in iteration count, and the implicit synchronization via spin-wait + `next_clean` is insufficient under `launch_server`'s heterogeneous-load environment.

`--deepep-mode normal` is a working production workaround because SGLang auto-disables CUDA graphs in that mode (forcing eager NORMAL kernels which we verified are empty-safe). Performance is degraded vs `auto` because no decode CUDA graphs.

## Smoking-gun evidence

With Python-level prints in `qwen3_moe.py` at MoE block boundaries and kernel-level `printf` in UCCL's LL recv-wait spin loops, with `CUDA_LAUNCH_BLOCKING=1`:

```
rank 0:  layer 44, 45, 46, 47, layer 0   ← finished 48 layers, started NEXT forward
rank 1:  layer 44, 45, 46, 47, layer 0   ← same
rank 3:  layer 44, 45, 46, 47, layer 0   ← same
rank 2:  layer 30, 31, 32, 33, layer 34  ← STUCK on layer 34 of FIRST forward
```

Kernel `printf` showed `rank=2` spinning in `combine_recv_ipc` waiting for `rdma_recv_flag[expert] != 0` from `src=0`, with cycle counter exceeding 100 seconds (~150 BILLION cycles).

Qwen3-30B-A3B has 48 transformer blocks. **Ranks 0/1/3 raced 14+ layers ahead of rank 2.** Rank 2 had real work (the warmup request); ranks 0/1/3 were `forward_idle` (no tokens).

py-spy at the same time showed:
```
rank 0:  apply (quantization/fp8.py:491)   ← Fp8LinearMethod (regular linear)
                                              forward_prepare attention QKV
rank 1/2/3: apply (quantization/fp8.py:1177) ← Fp8MoEMethod (MoE expert grouped GEMM)
```
Different layer phase = different layer index. Not aligned.

## Why this should be impossible

LL `dispatch` and `combine` are all-to-all collectives. Every rank must participate in every collective. If rank 2 hadn't reached layer 35's dispatch, ranks 0/1/3 should have hung waiting for it. Instead they advanced 14 layers, meaning the all-to-all is NOT actually synchronizing — fast ranks are completing it without rank 2's participation, by reading stale buffer data that satisfies the spin condition.

## What we ruled out

- **CUDA graphs as the trigger.** Reproduced with `--disable-cuda-graph`.
- **Empty input being the only trigger.** Any rank significantly faster than peers will race. Empty input is just the most common cause in production.
- **`mask_buffer_ptr` port being the right fix.** Per user direction: timeout-and-mask treats the symptom (hang) but not the underlying state corruption.
- **`clean_low_latency_buffer` missing barriers being the root cause.** UCCL has the cross-rank barriers commented out at `internode_ll.cu:27, 37`, but `clean_low_latency_buffer` is only called by sglang at NORMAL→LL mode transitions ([`deepep.py:226-244`](file:///home/shaoyuw/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L226-L244)), not between iterations. Adding the barrier here would help at startup but wouldn't prevent ongoing iteration drift.
- **Missing combine flag reset between iterations.** The combine flag IS reset because the dispatch recv count buffer and combine recv flag buffer share the SAME memory ([`ep_config.hpp:158-161`](file:///home/shaoyuw/uccl/ep/include/ep_config.hpp#L158-L161) asserts `dispatch_rdma_recv_count_buffer == combine_rdma_recv_flag_buffer`). Dispatch's `next_clean` cleanup zeros both at once.

## Best current hypothesis

The LL kernel's `next_clean` cleanup runs at KERNEL ENTRY (UCCL `internode_ll.cu:298-300`) — it zeros the buffer slots that the *next* iteration will use:

```cpp
if (sm_id == 0) {
  for (int i = lane_id; i < num_next_clean_int; i += WARP_SIZE) {
    next_clean[i] = 0;
    next_clean_second[i] = 0;
  }
  __syncwarp();
  for (int i = lane_id; i < num_experts; i += WARP_SIZE)
    atomic_add_release_global(atomic_finish_counter_per_expert + i,
                              FINISHED_SUM_TAG);
}
```

With the 2-buffer toggle (`buffers[low_latency_buffer_idx ^= 1]`):

```
Rank A iter 5 (uses B): kernel start → cleans buffer A → writes to B → reads B
Rank A iter 6 (uses A): kernel start → cleans buffer B → writes to A → reads A
                                       ↑ but rank B may still be in iter 5 using B
```

If rank A is fast enough to enter iter 6 while rank B is still in iter 5's combine reading buffer B, **rank A's iter 6 entry-cleanup zeros the slots that rank B is currently spinning on**. Rank B's wait then never terminates.

The implicit synchronization via spin-wait works WITHIN a single iteration (sender writes sentinel → receiver reads it), but does not prevent a fast rank's out-of-order entry to the NEXT iteration that races with the previous iteration's reader on the buffer being cleaned.

## Why DeepEP doesn't have this problem (uncertain)

Both DeepEP and UCCL have identical `next_clean`-at-entry cleanup pattern. We did NOT find a meaningful structural difference that would prevent this race in DeepEP. Hypotheses we did not fully verify:

1. **DeepEP's `mask_buffer_ptr` + `NUM_TIMEOUT_CYCLES` absorbs the race in practice.** When a stale read leads to wrong data and the eventual wait hits the timeout, the rank gets masked, allowing the system to proceed. UCCL has neither, so the race becomes a permanent hang. (User says this is not the right structural fix, but it may be why DeepEP appears fine.)
2. **DeepEP's CUDA graph capture lock-steps timing.** With all 4 ranks executing the same captured graph, kernel launch timing is more predictable than ad-hoc Python-driven launches, so drift is naturally smaller.
3. **There's still kernel-level synchronization we didn't identify.** Possibly in how DeepEP uses `atomic_clean_flag` or per-warp synchronization within `combine`. Worth a fresh look in the next session.

---

# Possible Fixes (none implemented in this branch)

Listed in increasing scope. **mask+timeout is NOT the right direction per user.**

## Option 1: Sequence numbers in the sentinel value

Encode iteration parity (or a small counter) into the sentinel. Receiver waits for the EXPECTED value, not just non-zero. Stale data from a previous iteration won't satisfy the spin condition.

- **Pros:** Minimal kernel change. No additional buffers. Same memory layout. Preserves LL latency.
- **Cons:** Needs careful encoding so the value space doesn't conflict with `-num_tokens_sent - 1` semantics. May require a new `epoch` counter parameter passed to the kernel from host.
- **Where:**
  - Dispatch count-send: `internode_ll.cu:436, 443` (current writes `-num_tokens_sent - 1`)
  - Dispatch recv-wait: `internode_ll.cu:510-530`
  - Combine flag-send: `internode_ll.cu:1062-1075`
  - Combine recv-wait: `internode_ll.cu:1095-1112`
- **Smallest possible fix. Try this first.**

## Option 2: Per-iteration cross-rank barrier inside the kernel

Add an explicit barrier at the END of dispatch and at the END of combine, similar to DeepEP's `barrier()` function at [`DeepEP/csrc/kernels/internode_ll.cu:23-70`](file:///home/shaoyuw/DeepEP/csrc/kernels/internode_ll.cu#L23-L70). Use `sync_buffer_ptr` (one int per rank) for a counter-based barrier via IPC peer pointers.

- **Pros:** Real synchronization. Eliminates drift entirely. Trivial correctness argument.
- **Cons:** Defeats the LL "low-latency" benefit. Performance cost per layer = inter-rank round-trip.
- **Where:** Add `sync_buffer_ptr` member to UCCL `Buffer` class, port DeepEP's `barrier()` function, call it at kernel boundaries.

## Option 3: More buffers (3+ instead of 2)

Increase the buffer rotation depth so the cleanup never races with the previous user. With N buffers, iter K's cleanup targets buffer for iter K+1; the buffer being cleaned was last used in iter K-N+1, by which time all peers should be done with it.

- **Pros:** Preserves LL semantics. No per-iteration sync overhead.
- **Cons:** N× memory. Bounds drift to N-1 iterations but doesn't eliminate it.
- **Where:** `LowLatencyLayout` (`ep_config.hpp:168-329`) — change `LowLatencyBuffer buffers[2]` to a larger array, update toggle from `^= 1` to `% N`. All call sites in `uccl_ep.cc` that compute the index.

## Option 4: Make `clean_low_latency_buffer` re-runnable + call between iterations

Add the cross-rank barrier to `clean_low_latency_buffer` (uncomment and implement what's stubbed at `internode_ll.cu:27, 37`), then have sglang call it between every LL forward pass.

- **Pros:** Reuses existing function.
- **Cons:** Per-forward-pass barrier overhead. May still race within a single forward pass since each forward has multiple LL calls (one per MoE layer × 48 layers for Qwen3-30B).

## Recommended: Option 1 first, then Option 3 if needed

Sequence-numbered sentinels are the smallest, lowest-overhead change. If they prove insufficient, escalate to more buffers (Option 3) or in-kernel barriers (Option 2).

---

# Reproduction Commands (Modal H200 4-GPU)

The Modal container ID changes per session; replace `ta-XXXXX` with the current one (`modal container list` to find it). The local working directory is `~/uccl` on the developer machine.

## Setup (once per fresh container)

```bash
# 1. Push the fix branch from local workstation to the remote
cd ~/uccl
git push origin fix/sglang-forward-idle-empty-input

# 2. Open shell into the Modal container
modal shell ta-XXXXX

# 3. Apply the sglang compatibility patch (separate UnboundLocalError bug in sglang's paras_a100 branch)
cd /sgl-workspace/sglang
git apply /data/patches/sglang_use_deep_gemm_fix.patch

# 4. Pull the UCCL EP fix branch into the container
cd /sgl-workspace/uccl
git fetch origin fix/sglang-forward-idle-empty-input
git checkout fix/sglang-forward-idle-empty-input
cd ep
python3 setup.py install   # ~30-60s, kernel rebuild
```

## Verify Bug A is fixed (eager unit tests)

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
bash /sgl-workspace/uccl_bench.sh
```

Expected output includes:
```
[empty-input-rank/edges]            forcing ranks [0, 3]    ... passed
[empty-input-rank/single-producer]  forcing ranks [1, 2, 3] ... passed
[empty-input-rank-LL/edges]         forcing ranks [0, 3]    ... passed
[empty-input-rank-LL/single-producer] forcing ranks [1, 2, 3] ... passed
[ll-cudagraph-asym/edges]           empty_ranks=[0, 3]      ... passed
[ll-cudagraph-asym/single-producer] empty_ranks=[1, 2, 3]   ... passed
```

## Working configuration: `--deepep-mode normal` (Bug A fixed, Bug B avoided)

```bash
cat > /tmp/launch_30b_normal.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}
export SGLANG_DG_CACHE_DIR=/data/deep_gemm_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export NVSHMEM_QP_DEPTH=2048
unset SGLANG_DEEPEP_BF16_DISPATCH || true

cat > /tmp/deepep_config.json <<JSON
{
  "normal_dispatch": {"num_sms": 24, "num_max_nvl_chunked_send_tokens": 16,
                      "num_max_nvl_chunked_recv_tokens": 512,
                      "num_max_rdma_chunked_send_tokens": 16,
                      "num_max_rdma_chunked_recv_tokens": 512},
  "normal_combine":  {"num_sms": 24, "num_max_nvl_chunked_send_tokens": 16,
                      "num_max_nvl_chunked_recv_tokens": 512,
                      "num_max_rdma_chunked_send_tokens": 16,
                      "num_max_rdma_chunked_recv_tokens": 512}
}
JSON

cd /sgl-workspace/sglang
exec python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-FP8 \
  --trust-remote-code --load-format dummy --quantization fp8 \
  --tp-size 4 --ep-size 4 --dp-size 4 \
  --mem-fraction-static 0.85 \
  --moe-dense-tp-size 1 --chunked-prefill-size 32768 \
  --cuda-graph-bs 256 --page-size 256 \
  --attention-backend fa3 \
  --enable-dp-attention --enable-dp-lm-head \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --deepep-config /tmp/deepep_config.json \
  --ep-num-redundant-experts 0 --ep-dispatch-algorithm dynamic \
  --port 30002 --host 127.0.0.1
EOF
chmod +x /tmp/launch_30b_normal.sh

nohup bash /tmp/launch_30b_normal.sh > /tmp/sglang_server.log 2>&1 &
tail -F /tmp/sglang_server.log
# Wait ~60-90s for "Uvicorn running"

# Test heterogeneous prompts (different sizes trigger forward_idle on idle ranks)
for i in 1 2 3 4; do
  echo "=== Request $i ==="
  PROMPT=$(printf 'Hi %.0s' $(seq 1 $((i*5))))
  curl -sS -m 30 -X POST http://127.0.0.1:30002/generate \
    -H 'Content-Type: application/json' \
    -d "{\"text\": \"$PROMPT\", \"sampling_params\": {\"max_new_tokens\": 5, \"temperature\": 0}}" \
    | head -c 250
  echo
done
```

All 4 requests should return JSON within seconds. Output text is gibberish because of `--load-format dummy` (random weights) — that's expected; the test is whether the server responds at all.

SGLang internally sets `disable_cuda_graph=True` when `--deepep-mode normal` is set ([`server_args.py:1428-1432`](file:///home/shaoyuw/sglang/python/sglang/srt/server_args.py#L1428-L1432)).

## Reproducing Bug B (the unfixed kernel-level desync)

Same script as above but with these key changes:
- REMOVE `--cuda-graph-bs 256 --page-size 256 --moe-dense-tp-size 1 --chunked-prefill-size 32768`
- REMOVE `--deepep-config /tmp/deepep_config.json`
- CHANGE `--deepep-mode normal` to `--deepep-mode auto`
- ADD `--disable-cuda-graph` to confirm it's not graphs

```bash
cat > /tmp/launch_30b_auto_nograph.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}
export SGLANG_DG_CACHE_DIR=/data/deep_gemm_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export NVSHMEM_QP_DEPTH=2048
unset SGLANG_DEEPEP_BF16_DISPATCH || true

cd /sgl-workspace/sglang
exec python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-FP8 \
  --trust-remote-code --load-format dummy --quantization fp8 \
  --tp-size 4 --ep-size 4 --dp-size 4 \
  --mem-fraction-static 0.85 \
  --enable-dp-attention --enable-dp-lm-head \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --disable-cuda-graph \
  --ep-num-redundant-experts 0 --ep-dispatch-algorithm dynamic \
  --port 30002 --host 127.0.0.1
EOF
chmod +x /tmp/launch_30b_auto_nograph.sh

# Launch with CUDA_LAUNCH_BLOCKING=1 so py-spy frames are accurate at the hang
CUDA_LAUNCH_BLOCKING=1 nohup bash /tmp/launch_30b_auto_nograph.sh \
    > /tmp/sglang_server.log 2>&1 &

# Server reaches "Uvicorn running" then immediately processes 4 warmup batches
# ("The capital city of France is" sample) and HANGS.
# nvidia-smi will show 100% GPU util / 0% memory util on all 4 ranks.
```

## Diagnose the hang

```bash
# Check GPU state
nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory --format=csv,noheader
# Expected when hung: all GPUs at 100% / 0%

# Find scheduler PIDs
PIDS=$(for p in /proc/[0-9]*; do
  c=$(cat $p/comm 2>/dev/null)
  [[ "$c" == sglang::sched* ]] && basename $p
done)
echo "PIDS=$PIDS"

# py-spy each scheduler — see which Python frame each is at
for pid in $PIDS; do
  echo "=== PID $pid ==="
  py-spy dump --pid $pid 2>&1 | head -25
done

# Layer-position evidence (this branch already has [SGL] prints)
grep '\[SGL\] r=' /tmp/sglang_server.log | sort -k2 -t'=' | tail -20

# Kernel-level evidence (this branch already has [DBG ...] prints)
grep 'STILL waiting' /tmp/sglang_server.log | tail -20
grep 'dispatch_send_ipc\|dispatch_send_ibgda' /tmp/sglang_server.log | tail -20
```

## Diagnostic prints already injected (revert before merging)

| Location | Print prefix | Information |
|----------|--------------|-------------|
| `ep/src/internode_ll.cu` LL dispatch recv-wait IPC loop | `[DBG dispatch_recv_ipc]` | rank, src_rank, responsible_expert_idx, cycles waited |
| `ep/src/internode_ll.cu` LL combine recv-wait IPC loop | `[DBG combine_recv_ipc]` | rank, responsible_expert_idx, src_rank, cycles waited |
| `ep/src/internode_ll.cu` LL dispatch count-send IPC + IBGDA paths | `[DBG dispatch_send_ipc/ibgda]` | rank, dst_rank, dst_expert_local, num_tokens_sent, sentinel, dst_p2p_ptr |

To revert before merging the host-side fix:
```bash
git -C ~/uccl log --oneline | grep '^[0-9a-f]* debug(ep):'
# git revert <each-debug-commit-sha>
```

The Python-level `[SGL]` print in `qwen3_moe.py` was injected ad-hoc into the container's sglang and is NOT on this branch. To re-inject in a new session:
```bash
sed -i '/^        final_hidden_states = self.experts($/i\        import time as _t; _r = __import__("torch").distributed.get_rank() if __import__("torch").distributed.is_initialized() else -1; print(f"[SGL] r={_r} layer={self.layer_id} t={_t.time():.4f} ENTER_EXPERTS", flush=True)' \
    /sgl-workspace/sglang/python/sglang/srt/models/qwen3_moe.py
```

## Useful Modal commands

```bash
# Find the active Modal container
modal container list | head -5

# Open a shell
modal shell ta-XXXXX

# Inside the container, find sglang processes
for p in /proc/[0-9]*; do
  c=$(cat $p/comm 2>/dev/null)
  [[ "$c" == sglang* ]] && echo "$(basename $p): $c"
done
# Expected: sglang::data_pa, sglang::detoken, sglang::schedul × 4

# Kill all sglang processes cleanly between test runs
pkill -9 -f 'sglang' 2>/dev/null
pkill -9 -f 'inductor.compile' 2>/dev/null
sleep 5

# Check for leaked zmq ports (will block next launch)
ss -tlnp 2>/dev/null | grep -E ':30[0-9]{3}'
# If something is held, kill that PID. Or change --port to avoid leaked port.

# DeepGEMM JIT cache — avoids 10-20 min recompile
ls -la /data/deep_gemm_cache/
# Set SGLANG_DG_CACHE_DIR=/data/deep_gemm_cache in launch script
```

---

# Mode Resolution Trace (reference)

How `--deepep-mode auto` decides between NORMAL and LL:

**Cross-rank coordination** ([`scheduler.py:2183-2240`](file:///home/shaoyuw/sglang/python/sglang/srt/managers/scheduler.py#L2183-L2240)):

```python
is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False
local_info = torch.tensor([num_tokens, can_cuda_graph, num_tokens_for_logprob,
                           is_extend_in_batch, ...], device=device)
torch.distributed.all_gather_into_tensor(global_info.flatten(), local_info, group=group)
is_extend_in_batch = global_info[:, 0, 3].tolist()
local_batch.is_extend_in_batch = any(is_extend_in_batch)
# OR-reduce — any prefilling rank → all ranks see True → all use NORMAL for this step
```

**Mode resolution** ([`utils.py:99-106`](file:///home/shaoyuw/sglang/python/sglang/srt/layers/moe/utils.py#L99-L106)):

```python
def resolve(self, is_extend_in_batch: bool) -> DeepEPMode:
    if self != DeepEPMode.AUTO:
        return self
    if is_extend_in_batch:
        return DeepEPMode.NORMAL
    else:
        return DeepEPMode.LOW_LATENCY
```

**CUDA graph capture** ([`cuda_graph_runner.py:711, 730`](file:///home/shaoyuw/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py#L711-L730)):
captures graphs in DECODE mode with `is_extend_in_batch=False` → AUTO resolves to LOW_LATENCY → captured graph contains LL kernels.

**Auto-disable for normal mode** ([`server_args.py:1428-1432`](file:///home/shaoyuw/sglang/python/sglang/srt/server_args.py#L1428-L1432)):

```python
def _handle_a2a_moe(self):
    if self.moe_a2a_backend == "deepep":
        if self.deepep_mode == "normal":
            logger.warning("Cuda graph is disabled because deepep_mode=`normal`")
            self.disable_cuda_graph = True
```

So `--deepep-mode normal` forces eager-only execution, sidestepping the LL kernel entirely.

---

# Files Modified

```
ep/src/uccl_ep.cc                       host-side fix (Bug A)
ep/src/internode_ll.cu                  diagnostic printf (revert before merge)
ep/bench/test_intranode.py              regression tests (eager mode)
ep/bench/test_low_latency.py            regression tests (eager + CUDA-graph)
ep/docs/sglang-forward-idle-fix.md      this document
```

# Files Deliberately Not Modified

- `ep/src/internode.cu` — internode-normal mode; out of scope
- `ep/src/intranode.cu`, `ep/src/layout.cu` — kernel files; eager-mode empty safety verified, no changes needed for Bug A
- `ep/src/proxy.cpp`, `rdma.cpp`, `fifo.cpp`, `uccl_proxy.cpp` — CPU proxy layer; not modified, but may be involved in Bug B (the IBGDA path goes through these)
- `ep/include/ep_config.hpp` — `LowLatencyBuffer` and `LowLatencyLayout`; would need changes here for Option 3 (more buffers)
- `~/sglang/**` — out of scope, except for the diagnostic `[SGL]` print injected ad-hoc into the container's `qwen3_moe.py`. NOT on this branch.
- `~/DeepEP/**` — used as reference only

---

# Open Questions for Next Session

1. **Why does DeepEP not exhibit the same drift problem?** Both kernels have the same `next_clean`-at-entry pattern with 2-buffer toggle. If the design is fundamentally racy, DeepEP should also fail. Possibilities listed above need ruling in or out.

2. **Does the bug also reproduce with `--deepep-mode low_latency` (explicit, not auto)?** Should be yes (same LL kernel path). Quick experiment to confirm.

3. **What is the exact GPU instruction the kernel is spinning on?** `cuda-gdb attach <pid>` on a stuck scheduler would give the kernel-side instruction pointer and confirm the spin location. Useful for verifying buffer addressing.

4. **Can the bug be reproduced in a unit test that uses CUDA graphs WITH a multi-iteration drift pattern?** Current `run_cuda_graph_asymmetric_test` doesn't reproduce. A more aggressive test that captures multiple graphs at different batch sizes and replays interleaved (mimicking sglang's pattern) might.

5. **Does Option 1 (sequence-numbered sentinels) work?** Smallest possible fix. Worth attempting first.

# Recommended Path Forward

1. **Land the host-side fix.** The current branch's content (minus diagnostic `printf`s and the partial `run_cuda_graph_asymmetric_test`) is a clean PR. Bug A is real, the fix is correct, regression tests cover it.

2. **Open a separate issue/PR for Bug B** with a link to this document. Note `--deepep-mode normal` workaround for users hitting the production deadlock.

3. **In a fresh debugging session, attempt Option 1 (sequence numbers) first** as the smallest-effort fix. If it doesn't resolve the drift, escalate to Option 3 (more buffers) or Option 2 (in-kernel barrier). Investigate Open Question 1 (DeepEP behavior) in parallel — there may be a synchronization mechanism we missed in the source-only investigation.
