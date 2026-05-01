# UCCL EP — SGLang `launch_server` `forward_idle` Fix

## Status

This document describes a **partial fix** for a deadlock that affects
`sglang.launch_server --moe-a2a-backend deepep` on the UCCL EP fork. The fix
addresses the host-side cliff that caused the most-visible symptom. There is
also a **separate, unfixed kernel-level bug** described in the
[Remaining Bug](#remaining-bug-ll-kernel-under-cuda-graph-replay) section
below; the workaround is documented but not yet eliminated at the source.

## Symptom

When SGLang runs `launch_server` with `--moe-a2a-backend deepep` and traffic
is heterogeneous across DP ranks, the cluster enters a deadlocked state:

- `nvidia-smi` reports 100% GPU utilization but 0% memory bandwidth on every rank.
- The CPU polling loop in `ep/src/uccl_ep.cc:697` eventually raises
  `RuntimeError: DeepEP error: CPU recv timeout`.
- The watchdog at `scheduler_runtime_checker_mixin.py` may fire first
  (`Watchdog timeout (self.watchdog_timeout=300)`) and kill the schedulers.

`bench_one_batch` does not reproduce the hang because it broadcasts the same
batch size to every DP rank — every rank always has `num_tokens > 0` and the
asymmetric-empty case never occurs.

## Two Layered Bugs

The full deadlock is the result of two distinct bugs that were initially
conflated. Separating them is essential for understanding what this fix does
and does not address.

### Bug A: Host-side asserts reject zero-token input

**Status: fixed by this branch.**

SGLang's `launch_server` schedules data-parallel forward passes. When a DP
rank has no sequences for the current step, `forward_idle()` is invoked with
empty input tensors. The MoE layer calls
`buffer.get_dispatch_layout(topk_ids, …)` with `topk_ids` of shape
`(0, num_topk)`. PyTorch's allocator returns `data_ptr() == 0` for zero-byte
buffers on this CUDA build (verified empirically on H200, see Background).

UCCL EP's host wrappers used `EP_HOST_ASSERT(_ptr != 0)` checks in many
places. Those checks fired on the empty rank, raised a C++ exception, and
prevented its kernel from launching. Other ranks then waited forever in
`barrier_block` for the empty rank to participate. After
`UCCL_EP_CPU_TIMEOUT_SECS` (~100 s) the CPU polling loop raised the timeout.

This branch removes those `_ptr != 0` checks in the affected wrappers,
mirroring DeepEP's design: DeepEP takes `at::Tensor` objects at the C++
boundary and asserts on tensor *properties* (`.dim()`, `.size()`, `.dtype()`,
`.is_contiguous()`) which all hold for zero-row tensors. UCCL EP takes raw
`uintptr_t` and only had pointer-non-null as a substitute, which fails on
allocators that return `0` for zero-byte buffers.

### Bug B: UCCL LL kernel deadlocks under CUDA graph replay with empty input

**Status: NOT fixed by this branch. Workaround documented below.**

After Bug A is fixed, `launch_server` still deadlocks when run with
`--deepep-mode auto` (the default) on traffic that produces empty ranks.
Stack inspection via `py-spy dump --pid <scheduler>` (with
`CUDA_LAUNCH_BLOCKING=1` so Python frames reflect actual kernel state)
shows all 4 DP schedulers stuck identically:

```
replay (torch/cuda/graphs.py:117)               # cudaGraphLaunch
replay (cuda_graph_runner.py:885)
_forward_raw (model_runner.py:2330)
forward (model_runner.py:2297)
```

The captured CUDA graph is hung. Because the captured graph for decode is
captured with `is_extend_in_batch=False`, the AUTO mode resolves it to
LOW_LATENCY kernels (see [Mode Resolution Trace](#mode-resolution-trace)).
The captured LL graph contains UCCL EP's LL `dispatch` and `combine` kernels.

DeepEP's equivalent LL kernels under the same configuration work — that
combination has been used for production benchmarks. UCCL's LL kernels work
in **eager** mode for empty input (verified by the regression tests in this
branch). The bug is specifically at the intersection of LL + CUDA-graph
replay + asymmetric empty input.

Root-causing this bug requires kernel-level diff against DeepEP and is left
as future work (see [Next Steps](#next-steps)). Until it is fixed, users
must work around it.

## Workaround for Bug B: `--deepep-mode normal`

SGLang automatically disables CUDA graphs when `--deepep-mode normal` is set.
The relevant code in
[`server_args.py:1428-1432`](file:///home/shaoyuw/sglang/python/sglang/srt/server_args.py#L1428-L1432):

```python
def _handle_a2a_moe(self):
    if self.moe_a2a_backend == "deepep":
        if self.deepep_mode == "normal":
            logger.warning("Cuda graph is disabled because deepep_mode=`normal`")
            self.disable_cuda_graph = True
```

With CUDA graphs disabled, decode runs through the eager NORMAL kernels.
UCCL EP's NORMAL kernels are empty-safe (verified by T1 audit and the
intranode regression test). Combined with the host-side fix in Bug A, this
configuration produces a working `launch_server` on a single-node 4-GPU H200.

The trade-off: decode performance is worse without CUDA graphs (no
amortization of kernel launch overhead). For single-node test deployments
this is acceptable. For production multi-node serving, Bug B must be fixed
in the kernel.

UCCL's own bench scripts at `ep/bench/sglang/` use this approach. The
`launch_uep` helper in `common_launch.sh` sets `--deepep-mode normal` for
all production-style runs (`Qwen3-30B_uep.sh`, `Qwen3-235B_uep.sh`).

## Mode Resolution Trace

For reference, the auto-mode resolution that triggers Bug B:

**SGLang enforces cross-rank mode consistency.** All DP ranks always pick the
same DeepEP mode for a given forward step. The mechanism is an OR-reduce on
`is_extend_in_batch` across all ranks
([`scheduler.py:2183-2240`](file:///home/shaoyuw/sglang/python/sglang/srt/managers/scheduler.py#L2183-L2240)):

```python
# Each rank computes its local is_extend
is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False

# All-gather across all DP ranks
torch.distributed.all_gather_into_tensor(global_info.flatten(), local_info, group=group)
is_extend_in_batch = global_info[:, 0, 3].tolist()  # [F, F, T, T]

# OR-reduce: any prefilling rank → all ranks see True
local_batch.is_extend_in_batch = any(is_extend_in_batch)
```

Then mode resolution at
[`utils.py:99-106`](file:///home/shaoyuw/sglang/python/sglang/srt/layers/moe/utils.py#L99-L106):

```python
def resolve(self, is_extend_in_batch: bool) -> DeepEPMode:
    if self != DeepEPMode.AUTO:
        return self                       # explicit normal or low_latency: stays
    if is_extend_in_batch:
        return DeepEPMode.NORMAL          # any rank prefilling → all use NORMAL
    else:
        return DeepEPMode.LOW_LATENCY     # all ranks decoding/idle → all use LL
```

The CUDA graph runner captures graphs in DECODE forward mode only
([`cuda_graph_runner.py:263`](file:///home/shaoyuw/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py#L263))
and passes `is_extend_in_batch=False` to the DeepEP adapter
([`cuda_graph_runner.py:711, 730`](file:///home/shaoyuw/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py#L711-L730)).
With AUTO mode that resolves to LOW_LATENCY, so the captured graph contains
LL kernels. With NORMAL mode that resolves to NORMAL — but `disable_cuda_graph
= True` was already set, so no graph is captured at all.

## Background: Empirical PyTorch Behavior on This Cluster

Tested on H200 inside the Modal container:

```
torch.empty((0, 4),    dtype=bool,        device='cuda').data_ptr() = 0
torch.empty((0, 4),    dtype=int64,       device='cuda').data_ptr() = 0
torch.empty((0, 4096), dtype=bfloat16,    device='cuda').data_ptr() = 0
torch.empty((0, 4),    dtype=float32,     device='cuda').data_ptr() = 0
```

The CUDA caching allocator on this build returns `0` for every zero-byte
allocation regardless of dtype. The behavior is allocator-dependent in
general — PyTorch documentation does not guarantee non-null `data_ptr()` for
empty tensors — but on this cluster it is reliably `0`. This is the trigger
for Bug A's pointer-null asserts.

## Fix Applied (Bug A only)

All changes are in `ep/src/uccl_ep.cc`. The pointer-non-null checks in four
host wrappers are removed entirely, mirroring DeepEP. Property-based asserts
(`num_tokens >= 0`, `num_experts > 0`, `hidden > 0`, shape divisibility) are
preserved.

| Wrapper | Original asserts | Final state |
|---------|------------------|-------------|
| `get_dispatch_layout` (~`:521`) | `topk_idx_ptr != 0`, `num_tokens_per_rank_ptr != 0`, `num_tokens_per_expert_ptr != 0`, `is_token_in_rank_ptr != 0`, `num_experts > 0` | Only `num_tokens >= 0` and `num_experts > 0` remain |
| `intranode_prepare` (~`:640`) | `num_tokens > 0`, `num_experts > 0`, plus several `_ptr != 0` checks | Only `num_tokens >= 0` and `num_experts > 0` remain |
| `intranode_dispatch` (~`:732`) | `_ptr != 0` checks for `x_ptr`, `is_token_in_rank_ptr`, `recv_x_ptr`, etc., plus `num_tokens > 0 && hidden > 0 && num_recv_tokens >= 0` | `num_tokens >= 0 && hidden > 0 && num_recv_tokens >= 0` plus shape divisibility |
| `intranode_combine` (~`:823`) | `_ptr != 0` checks for `x_ptr`, `src_idx_ptr`, `rank_prefix_matrix_ptr`, `channel_prefix_matrix_ptr`, `send_head_ptr`, `recv_x_ptr`, plus shape divisibility | Only shape divisibility remains |
| `low_latency_dispatch` (~`:1209`) | `_ptr != 0` checks for `x_ptr`, `topk_idx_ptr`, `packed_recv_x_ptr`, `packed_recv_count_ptr`, `packed_recv_src_info_ptr`, `packed_recv_layout_range_ptr`, `packed_recv_x_scales_ptr` (when `use_fp8`) | All removed; property/shape asserts preserved |
| `low_latency_combine` (~`:1311`) | `_ptr != 0` checks for `x_ptr`, `topk_idx_ptr`, `topk_weights_ptr`, `src_info_ptr`, `layout_range_ptr`, `out_ptr` | All removed; shape/dim asserts preserved |

The internode-normal asserts at `:882` and `:985` were left unchanged. Those
code paths are out of scope for this fix.

`cudaGetLastError()` checks were also added after each of the 5 intranode
kernel launches in `uccl_ep.cc`, mirroring the existing pattern in the LL
host wrappers. This converts silent kernel-launch failures (e.g., from
invalid grid configurations on edge-case input) into logged errors.

## Verification

### What was verified

| Path | Method | Result |
|------|--------|--------|
| Eager intranode dispatch + combine, empty input on some ranks | Unit test (`test_intranode.py:269` two scenarios) on 4×H200 | PASSED |
| Eager LL dispatch + combine, empty input on some ranks | Unit test (`test_low_latency.py:140` two scenarios) on 4×H200 | PASSED |
| `launch_server` with `--deepep-mode normal` (CUDA graphs auto-disabled) + heterogeneous curl requests | End-to-end on 4×H200 | PASSED — server stays healthy across multiple requests |
| Existing intranode/LL correctness sweeps | Existing unit tests | PASSED (no regressions) |

### What was NOT verified

| Path | Why not |
|------|---------|
| LL dispatch + combine under CUDA graph replay, with empty input | Unit tests do not capture CUDA graphs; this path is Bug B and is unfixed |
| `launch_server` with `--deepep-mode auto` (CUDA graphs enabled, decode uses LL graph) | Hangs deterministically on heterogeneous traffic — this is Bug B |
| `launch_server` with `--deepep-mode low_latency` | Same code path as auto-decode; expected to hit Bug B |
| Multi-node deployment | Cluster is single-node; multi-node may surface additional CPU-proxy interactions |

### Regression tests added

| Test | File | Scenarios |
|------|------|-----------|
| `run_empty_input_rank_test` | `ep/bench/test_intranode.py:269` | (a) `edges`: ranks `{0, num_ranks-1}` empty. (b) `single-producer`: ranks `{1, …, num_ranks-1}` empty. |
| `run_empty_input_rank_test` | `ep/bench/test_low_latency.py:140` | Same two scenarios. |

Both tests use deterministic round-robin routing on non-empty ranks (avoids
RNG divergence) and construct empty ranks with `(0, hidden)` `x`,
`(0, num_topk)` int64 `topk_idx`, and `(0, num_topk)` `topk_weights`. The
int64 zero-row tensor is what triggers `data_ptr() == 0`, exercising the
removed pointer-null checks.

**Limitation: these tests only cover eager mode.** They do not exercise
CUDA graphs and therefore cannot catch Bug B. A regression test for Bug B
would need to manually capture and replay a CUDA graph that includes the LL
kernels with empty input — that work is part of fixing Bug B.

To run on a Hopper cluster:

```
torchrun --standalone --nproc_per_node=8 ep/bench/test_intranode.py
torchrun --standalone --nproc_per_node=8 ep/bench/test_low_latency.py
```

## Remaining Bug: LL Kernel Under CUDA Graph Replay

This bug is unresolved in this branch. Documenting what is known so the next
person can pick it up.

### What we know

1. UCCL EP's LL `dispatch` and `combine` kernels work correctly in eager
   mode for empty input — the unit tests prove this.
2. They work correctly under CUDA graph capture+replay for non-empty input
   (this is the standard case used by every prior bench).
3. They deadlock under CUDA graph capture+replay when one or more ranks
   have empty input on replay. Stack signature: all ranks stuck in
   `cudaGraphLaunch` (replay), GPU at 100% util / 0% memory.
4. DeepEP's LL kernels do not deadlock under the same conditions.

### What we do not know yet

- Which specific operation in the captured graph hangs. A `cuda-gdb attach`
  on a stuck process would show the kernel-side instruction pointer.
- Whether the bug is in the CPU proxy path (`uccl::nvshmemi_ibgda_*`), the
  IPC direct path (`st_release_sys_global`), or shared between them.
- Whether the `low_latency_buffer_idx` host-side toggle interacts badly
  with CUDA graphs (the captured graph would always reference the buffer
  index value at capture time).

### Recommended next steps to fix Bug B

1. **Reproduce in a controlled unit test.** Extend `test_low_latency.py`
   with a scenario that wraps the dispatch+combine pair in `torch.cuda.graph()`
   with the empty-rank pattern. This isolates the bug from sglang's
   complexity.

2. **Run with `cuda-gdb`.** Attach to a stuck scheduler, get the kernel
   instruction pointer, identify the spinning instruction. Likely candidates:
   a `while (ld_acquire_sys_global(...) == 0)` spin in the LL receiver
   path that never sees the sentinel value because the captured graph
   doesn't propagate the sentinel write correctly.

3. **Compare LL kernel host wrappers between UCCL and DeepEP.** Look for:
   - Pre-launch `cudaMemset`/`cudaMemsetAsync` calls — if these are not
     captured into the graph, state from a previous call leaks into the
     next replay
   - Buffer index handling — UCCL uses `low_latency_buffer_idx ^= 1`
     host-side and passes the value as a kernel arg. DeepEP may handle this
     differently (e.g., per-buffer captured graphs)
   - The `clean_low_latency_buffer` mechanism — when is it called and how
     does it interact with graph capture

4. **Check the CPU proxy.** If UCCL's CPU proxy path is involved, the
   FIFO-relayed atomic-add for the count-send sentinel may not be visible
   to the receiver inside a captured graph because the proxy thread is
   external to the graph.

## Files Modified

```
ep/src/uccl_ep.cc                    host-side fix (Bug A)
ep/bench/test_intranode.py           regression test (eager mode only)
ep/bench/test_low_latency.py         regression test (eager mode only)
ep/docs/sglang-forward-idle-fix.md   this document
```

Branch: `fix/sglang-forward-idle-empty-input` on
`github.com/shawlleyw/uccl`.

## Files Deliberately Not Modified

- `ep/src/internode.cu` — internode-normal mode; out of scope
- `ep/src/intranode.cu`, `ep/src/internode_ll.cu`, `ep/src/layout.cu` —
  kernel files; eager-mode empty safety verified in unit tests, no changes
  needed for Bug A
- `ep/src/proxy.cpp`, `ep/src/rdma.cpp`, `ep/src/fifo.cpp`,
  `ep/src/uccl_proxy.cpp` — CPU proxy layer; not modified, but suspected
  involvement in Bug B (see [Recommended next steps](#recommended-next-steps-to-fix-bug-b))
- `~/sglang/**` — out of scope; SGLang's behavior is correct
- `~/DeepEP/**` — used as reference only

## Next Steps

1. Validate this branch on a Hopper cluster with `--deepep-mode normal` and
   real (non-dummy) weights. The current validation used `--load-format
   dummy` for speed; a real-weights run gives the final smoke test.

2. Decide on Bug B fix priority. If single-node `launch_server` with
   `--deepep-mode normal` is acceptable for the immediate use case, Bug B
   can be deferred. If `--deepep-mode auto` (with CUDA graphs) is required
   for production decode performance, Bug B must be fixed. The fix recipe
   in [Recommended next steps](#recommended-next-steps-to-fix-bug-b) is the
   starting point.

3. Consider upstreaming this branch to the UCCL EP main repository.

## Caveats and Honest Assessment

- The "kernel-level empty safety" audit (T1 in the original work plan)
  produced a verdict of "no kernel changes needed" based on source reading
  alone. That verdict was correct for **eager mode**, which is what the
  audit examined. It was **not** correct as a general statement: Bug B
  exists and is in those same kernels under different execution conditions
  (CUDA graph replay). A more rigorous audit would have explicitly noted
  that CUDA-graph behavior was out of scope and untested. This document
  corrects that omission.
- The regression tests added in this branch will catch any future
  re-introduction of Bug A (host-side asserts on empty input). They will
  not catch Bug B because they do not capture CUDA graphs.
- The `--deepep-mode normal` workaround works because SGLang itself
  disables CUDA graphs in that mode. It is not a long-term fix; production
  performance demands `auto` (or `low_latency`) for decode.
