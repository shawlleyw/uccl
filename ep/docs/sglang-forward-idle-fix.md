# UCCL EP — SGLang `launch_server` `forward_idle` Investigation

## Status

| Component | Status | Notes |
|-----------|--------|-------|
| Bug A: host-side asserts reject empty input | **FIXED** | All `_ptr != 0` and `num_tokens > 0` checks loosened in `uccl_ep.cc`, mirroring DeepEP |
| Bug B: rank desync from missing clean barrier | **FIXED** | Added cross-rank IPC barrier (before+after) inside `clean_low_latency_buffer`. Mirrors DeepEP's `nvshmemx_barrier_all_block` pattern using UCCL's existing `barrier_block`. |
| Eager unit tests for empty input | **PASSING** | `test_intranode.py`, `test_low_latency.py` `run_empty_input_rank_test` |
| `launch_server` with `--deepep-mode normal` | **WORKING** | (was working before; still works) |
| `launch_server` with `--deepep-mode auto --disable-cuda-graph` | **WORKING** | Verified end-to-end on Modal H200; 8 heterogeneous curls all 200, no kernel hangs |
| `launch_server` with `--deepep-mode auto` (CUDA graphs ON) | **WORKING** | Verified end-to-end on Modal H200; production-grade config |

## Branch State

Branch `fix/sglang-forward-idle-empty-input` on `github.com/shawlleyw/uccl`.

| File | What changed | Purpose |
|------|--------------|---------|
| `ep/src/uccl_ep.cc` | Removed `_ptr != 0` asserts in 6 host wrappers; loosened `num_tokens > 0` to `>= 0`; added `cudaGetLastError()` after intranode kernel launches; pass `barrier_signal_ptrs_gpu`/`nvl_rank`/`num_nvl_ranks` into `clean_low_latency_buffer` | Bug A + Bug B fix |
| `ep/src/internode_ll.cu` | `clean_low_latency_buffer` kernel now calls `barrier_block<kNumRanks>` BEFORE and AFTER the per-rank zero-fill; host wrapper templated via `SWITCH_RANKS` | Bug B fix |
| `ep/include/internode_ll.cuh` | Updated `clean_low_latency_buffer` declaration | Bug B fix |
| `ep/bench/test_intranode.py` | Added `run_empty_input_rank_test()` with `edges` + `single-producer` scenarios | Regression test for Bug A (eager mode) |
| `ep/bench/test_low_latency.py` | Same + `run_cuda_graph_asymmetric_test()` | Regression tests |
| `ep/docs/sglang-forward-idle-fix.md` | This document | Investigation log + fix writeup |

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

# Bug B: Rank desync from missing cross-rank barrier in `clean_low_latency_buffer`

## Status

**FIXED.** Verified end-to-end on Modal H200 across all three relevant launch_server configurations:

| Config | Result |
|---|---|
| `--deepep-mode auto --disable-cuda-graph` + `CUDA_LAUNCH_BLOCKING=1` | 4 heterogeneous curls (5/10/15/20 token prompts) all 200, no hangs |
| `--deepep-mode auto --disable-cuda-graph` (no LAUNCH_BLOCKING) | 4 heterogeneous curls all 200, no hangs |
| `--deepep-mode auto` (CUDA graphs ON, no LAUNCH_BLOCKING) | 8 heterogeneous curls all 200, no hangs |

LL unit tests (`test_low_latency.py`) still pass after the fix.

## Root cause (corrected from prior hypotheses)

The original investigation explicitly **dismissed** "missing barrier in `clean_low_latency_buffer`" as the cause, on the grounds that clean is only called at NORMAL→LL mode transitions, not between iterations. That dismissal was wrong: the **first** LL iteration after the transition is exactly where the deadlock fires, and the missing barrier IS the cause. There is no ongoing per-iteration drift to worry about — the corruption happens once, at startup, and from then on the system is wedged.

The race:

```
Rank A: clean()  zeros own buffer  →  dispatch SEND  writes -1 via IPC into rank B's slot S
Rank B:                      clean()  zeros own buffer (also wipes A's write)  →  dispatch SEND  →  dispatch RECV reads slot S = 0  →  spins forever
```

Each rank's `clean_low_latency_buffer` zeroes only its **own** signaling buffer (no cross-rank effect on its own). But peers write into our signaling buffer via IPC P2P pointers during dispatch SEND. Without a barrier between everyone-finished-cleaning and anyone-starting-dispatch, a fast rank's first dispatch SEND can land in a slow rank's buffer and then be wiped by the slow rank's later clean. The slow rank's RECV then sees zero and spins.

DeepEP's `clean_low_latency_buffer` calls `nvshmemx_barrier_all_block()` before AND after the zero-fill ([`DeepEP/csrc/kernels/internode_ll.cu#L83-L101`](file:///home/shaoyuw/DeepEP/csrc/kernels/internode_ll.cu#L83-L101)) precisely to prevent this. UCCL had both calls **commented out**.

## Smoking-gun evidence (kernel `printf`)

After adding diagnostics:

- **All 4 ranks complete dispatch SEND** (every `responsible_expert_idx ∈ [0, num_experts)` runs the `st_release_sys_global` to peer slots).
- **2 of 4 ranks complete dispatch RECV** (saw the sentinel values written by peers).
- **2 of 4 ranks remain stuck in dispatch RECV spin** for 500+ billion cycles, waiting for slots that peers already wrote to.
- **The unstuck ranks proceeded through expert compute and combine SEND**, then stuck in combine RECV waiting for the lagging ranks' combine flags (which never come because those ranks are still wedged in dispatch).
- The same pattern reproduces non-deterministically: which 2 of 4 ranks get stuck depends on launch timing.

This pattern is consistent only with the slow rank's clean wiping the fast rank's IPC dispatch write to that slow rank's buffer.

## Fix

`ep/src/internode_ll.cu` — `clean_low_latency_buffer` kernel now takes `barrier_signal_ptrs` and the rank, and calls `barrier_block<kNumRanks>` before and after the zero-fill. Host wrapper picks the right `kNumRanks` template via `SWITCH_RANKS`.

`ep/src/uccl_ep.cc` — `Buffer::clean_low_latency_buffer` passes the existing `barrier_signal_ptrs_gpu` (set up in `Buffer::sync()`) along with `nvl_rank` / `num_nvl_ranks`. Asserts that `barrier_signal_ptrs_gpu` is non-null.

`ep/include/internode_ll.cuh` — declaration updated.

This mirrors DeepEP's pre/post-clean barrier but uses UCCL's existing IPC-based `barrier_block` instead of NVSHMEM. Intra-node only — multi-node clean barrier across nodes would need a separate mechanism (e.g., NCCL host-side barrier or a CPU-proxy barrier), but is out of scope here.

Performance impact: clean is called only at NORMAL→LL mode transitions (not per layer), so two extra cross-rank atomics per transition is negligible. LL benchmark numbers are unchanged.

## Why prior hypotheses were wrong

- **"Mid-stream `next_clean` race in the 2-buffer toggle"**: the two LL buffers are used in strict alternation (`dispatch ↔ buffer 0`, `combine ↔ buffer 1`); with all 4 ranks doing the same call sequence, both ranks always target the same buffer index for the same call, so no cross-buffer race exists during steady state. The race only happens at the boundary where buffers transition from "untouched / NORMAL-mode garbage" to "in use".
- **"`mask_buffer_ptr` is what saves DeepEP"**: DeepEP's mask absorbs late timeouts after a hang, but the structural reason DeepEP doesn't hang in the first place is the pre/post barrier in clean. We confirmed this by porting only the clean barrier (no mask, no timeout) and the bug went away.
- **"Sequence-numbered sentinel"**: would also work, but is a much larger change (touches every LL kernel) for the same correctness guarantee that the clean barrier provides for free.

---

# Reproduction & Verification Commands (Modal H200 4-GPU)

These commands originally reproduced Bug B; with the cross-rank clean barrier in place they now serve as the verification suite. Replace `ta-XXXXX` with your current container ID (`modal container list`). Local working directory is `~/uccl` on the developer machine.

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

## Verifying Bug B fix (`--deepep-mode auto`)

This is the configuration that previously hung; now it should complete the warmup and respond to curl. Use the same launch script for both `--disable-cuda-graph` and CUDA-graph-enabled paths (just toggle the flag).

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

# Detached launch (no LAUNCH_BLOCKING needed for the fix; included previously
# only because py-spy frames were less accurate without it during diagnosis).
setsid nohup bash /tmp/launch_30b_auto_nograph.sh > /tmp/sglang_server.log 2>&1 < /dev/null & disown

# Wait ~2-3 minutes for "The server is fired up and ready to roll!" then send
# heterogeneous prompts that previously triggered the hang:
for i in 1 2 3 4; do
  PROMPT=$(printf 'Hi %.0s' $(seq 1 $((i*5))))
  curl -sS -m 30 -X POST http://127.0.0.1:30002/generate \
    -H 'Content-Type: application/json' \
    -d "{\"text\": \"$PROMPT\", \"sampling_params\": {\"max_new_tokens\": 5, \"temperature\": 0}}" \
    | head -c 200
  echo
done

# To also verify the CUDA-graph path, remove the --disable-cuda-graph line and
# relaunch. Output text is gibberish under --load-format dummy; the test is
# whether the server responds (HTTP 200) instead of hanging.
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

## Re-injecting diagnostic prints (only if a new bug needs investigation)

The diagnostic `[DBG ...]` prints used during this investigation have all been reverted from the branch (commits `b27da4c`, `6000e0a`, `3471717`, `365ea1b` revert the four `debug(ep): ...` commits). The kernel is clean.

If you need to instrument the kernel again, the useful insertion points are:

| Location | Suggested print prefix | Information |
|----------|------------------------|-------------|
| `ep/src/internode_ll.cu` LL dispatch recv-wait IPC loop | `[DBG dispatch_recv_ipc]` | rank, src_rank, responsible_expert_idx, cycles waited |
| `ep/src/internode_ll.cu` LL combine recv-wait IPC loop | `[DBG combine_recv_ipc]` | rank, responsible_expert_idx, src_rank, cycles waited |
| `ep/src/internode_ll.cu` LL dispatch count-send IPC + IBGDA paths | `[DBG dispatch_send_ipc/ibgda]` | rank, dst_rank, dst_expert_local, num_tokens_sent, sentinel, dst_p2p_ptr |
| `ep/src/internode_ll.cu` combine flag-send IPC path | `[DBG combine_send_ipc]` | rank, dst_rank, global_expert, dst_p2p_ptr, ll_buf |
| `ep/src/internode_ll.cu` kernel entry of dispatch + combine | `[DBG (dispatch|combine)_kernel_enter]` | rank, num_tokens, ll_buf, phases |

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
ep/src/uccl_ep.cc                       host-side fix (Bug A) + clean barrier wiring (Bug B)
ep/src/internode_ll.cu                  clean_low_latency_buffer cross-rank barrier (Bug B)
ep/include/internode_ll.cuh             updated declaration (Bug B)
ep/bench/test_intranode.py              regression tests (eager mode)
ep/bench/test_low_latency.py            regression tests (eager + CUDA-graph)
ep/docs/sglang-forward-idle-fix.md      this document
```

# Files Deliberately Not Modified

- `ep/src/internode.cu` — internode-normal mode; out of scope
- `ep/src/intranode.cu`, `ep/src/layout.cu` — eager-mode empty safety already verified
- `ep/src/proxy.cpp`, `rdma.cpp`, `fifo.cpp`, `uccl_proxy.cpp` — CPU proxy layer; the cross-node IBGDA path passes through these but the bug we hit was the intra-node IPC race, which the in-kernel barrier resolves directly
- `ep/include/ep_config.hpp` — `LowLatencyBuffer` / `LowLatencyLayout` unchanged
- `~/sglang/**` — only the ad-hoc `[SGL]` print injected into the container's `qwen3_moe.py` for diagnostic; NOT on this branch
- `~/DeepEP/**` — reference only

---

# Open Questions / Follow-ups

1. **Multi-node clean barrier.** The fix uses UCCL's intra-node `barrier_block` (IPC-based). For multi-node LL deployments, the same race could occur across nodes — peers on a different node would write via IBGDA into our signaling buffer and could be wiped by our late clean. A multi-node clean barrier would need either NCCL host-side `ncclBarrier` or a CPU-proxy-mediated barrier. Out of scope here; raise as a follow-up if multi-node `launch_server` is exercised.

2. **CUDA-graph asymmetric reproducer.** `run_cuda_graph_asymmetric_test` in `test_low_latency.py` was written during the investigation but does not by itself trigger Bug B (the bug needs the NORMAL→LL transition that sglang performs between forwards). Consider extending the unit test to call `clean_low_latency_buffer` and then immediately replay an asymmetric graph, to lock the regression in.

3. **CUDA-GDB instruction pointer.** Not needed now that the bug is fixed; would only be useful if a related-but-distinct deadlock surfaces.

3. **In a fresh debugging session, attempt Option 1 (sequence numbers) first** as the smallest-effort fix. If it doesn't resolve the drift, escalate to Option 3 (more buffers) or Option 2 (in-kernel barrier). Investigate Open Question 1 (DeepEP behavior) in parallel — there may be a synchronization mechanism we missed in the source-only investigation.
