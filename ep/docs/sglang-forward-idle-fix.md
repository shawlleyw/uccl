# UCCL EP — SGLang `launch_server` `forward_idle` Fix

## Symptom

When SGLang runs with `--moe-a2a-backend deepep` in `launch_server` mode, some GPU ranks
enter a state where `nvidia-smi` shows 100% GPU utilization but 0% memory bandwidth. The
process eventually crashes with:

```
RuntimeError: DeepEP error: CPU recv timeout
```

The timeout originates at `ep/src/uccl_ep.cc:697`, where the CPU polling loop waits for
a response from a peer rank that never arrives. The hang is permanent once triggered.

Running `bench_one_batch` with the same model and EP configuration works without issue.
The failure is specific to `launch_server` with real heterogeneous request traffic.

## Root Cause

SGLang's `launch_server` uses data-parallel (DP) scheduling. Different DP ranks may
receive different numbers of sequences per forward step. When a rank receives zero
sequences, it calls `buffer.get_dispatch_layout(topk_ids, ...)` with a zero-element
`topk_ids` tensor. PyTorch returns `data_ptr() == 0` (null pointer) for zero-element
tensors in some versions.

Three host-side asserts in `uccl_ep.cc` fire immediately for the empty rank:

- `:521` `EP_HOST_ASSERT(topk_idx_ptr != 0)` — in `get_dispatch_layout`
- `:640` `EP_HOST_ASSERT(num_tokens > 0)` — in `intranode_prepare`
- `:732` `EP_HOST_ASSERT(num_tokens > 0 && hidden > 0 && ...)` — in `intranode_dispatch`

When any of these asserts fires, the empty rank's kernel never launches. The other ranks
proceed normally and enter `barrier_block`, waiting for the empty rank to participate in
the collective. The empty rank never joins. The CPU polling loop at `:697` spins until
`UCCL_EP_CPU_TIMEOUT_SECS` (~100 seconds) elapses, then raises the timeout error.

The root cause is that the host-side guards were written assuming `num_tokens > 0` is
always true, which holds for `bench_one_batch` but not for `launch_server` with
heterogeneous DP batches.

## Why `bench_one_batch` Worked

`bench_one_batch` broadcasts the same batch to all DP ranks simultaneously. Every rank
always has `num_tokens > 0`, so the three asserts never fire. The asymmetric case (some
ranks empty, others not) only occurs in `launch_server` with real heterogeneous request
traffic, where different clients connect to different DP ranks and the per-step token
counts diverge.

## Kernel-Level Empty Safety (Verified)

Before loosening the host-side guards, T1 verified that all 7 kernel mechanisms handle
`num_tokens == 0` correctly. The verification was read-only (no source files modified).

| VP | Mechanism | Verdict |
|----|-----------|---------|
| 1 | Intranode dispatch: sender sentinel writes unconditional | VERIFIED EMPTY-SAFE |
| 2 | Intranode dispatch: receiver decodes sentinel, loop guard exits | VERIFIED EMPTY-SAFE |
| 3 | LL dispatch: atomic counter math reaches `FINISHED_SUM_TAG * 2` | VERIFIED EMPTY-SAFE |
| 4 | LL dispatch: count-send value is `-num_tokens_sent - 1` | VERIFIED EMPTY-SAFE |
| 5 | LL dispatch: receiver decodes sentinel, skips token-copy loop | VERIFIED EMPTY-SAFE |
| 6 | LL combine: finishing-flag send unconditional; proxy converts value to 1 | VERIFIED EMPTY-SAFE |
| 7 | LL combine: receive-side wait spins on `== 0`, exits when proxy delivers 1 | VERIFIED EMPTY-SAFE |

Key findings:

**Intranode dispatch (VP1-VP2).** The sender's `st_relaxed_sys_global` calls that write
the `-v - 1` sentinel encoding are gated only by `lane_id == 0 && send_warp_id_in_rank == 0`,
not by any `num_tokens > 0` predicate. They execute before the per-token send loop. When
`num_tokens == 0`, the prefix matrix entries are all zero, so the sentinel value is `-1`.
The receiver's spin-wait exits on `-1` (non-zero), decodes to `num_tokens_to_recv = 0`,
and the copy loop at line 460 runs zero iterations.

**LL dispatch (VP3-VP5).** When `num_tokens == 0`, the per-token counting loop contributes
`0` to the atomic counter. The cleanup warp unconditionally adds `+FINISHED_SUM_TAG`, and
the count-reduction warp adds `+FINISHED_SUM_TAG - 0`. The final counter value is
`2 * FINISHED_SUM_TAG`, which matches the wait predicate exactly. The count-send value
`-num_tokens_sent - 1 = -1` is non-zero, so the receiver's spin-wait exits and decodes
to `num_recv_tokens = 0`.

**LL combine finishing flag (VP6-VP7).** The previously uncertain path: the IBGDA combine
branch at `internode_ll.cu:1071` passes `num_tokens_to_send` as the value, which is `0`
for an empty-input rank. However, the CPU proxy at `rdma.cpp:2230, 2518, 2559, 3212`
unconditionally rewrites `value = 1` whenever `is_combine` is set on the command. The
IPC path at line 1063 writes the literal `1` directly. Both paths guarantee a non-zero
value reaches the receiver's flag slot, so the receiver's `== 0` spin-wait exits cleanly.

The T1 conclusion: kernels are empty-safe end-to-end. Only the host-side guards needed
loosening. T5 (kernel-side patches) was not needed.

## Fix Applied

Three changes were made across two commits (T2/T3 and T4), all in `ep/src/uccl_ep.cc`.

| Change | File | Before | After | Rationale |
|--------|------|--------|-------|-----------|
| T2a | `uccl_ep.cc:640` | `EP_HOST_ASSERT(num_tokens > 0)` | `EP_HOST_ASSERT(num_tokens >= 0)` | Empty input is valid; negative values still rejected |
| T2b | `uccl_ep.cc:732` | `EP_HOST_ASSERT(num_tokens > 0 && hidden > 0 && ...)` | `EP_HOST_ASSERT(num_tokens >= 0 && hidden > 0 && ...)` | Same rationale; `hidden > 0` preserved |
| T3 | `uccl_ep.cc:521` | `EP_HOST_ASSERT(topk_idx_ptr != 0)` | `EP_HOST_ASSERT((topk_idx_ptr != 0) \|\| (num_tokens == 0))` + `EP_HOST_ASSERT(num_tokens >= 0)` | PyTorch may return `data_ptr() == 0` for zero-element tensors; the layout kernel's loop is bounded by `num_tokens`, so no dereference occurs when `num_tokens == 0` |
| T4 | `uccl_ep.cc` | No launch-error checks on intranode kernel launches | `cudaGetLastError()` after each of 5 intranode kernel launches | Mirrors the LL host wrapper pattern; converts silent kernel-launch failures into logged errors |

The internode-normal asserts at `:882` and `:985` (`EP_HOST_ASSERT(num_tokens > 0 && ...)`)
were deliberately left unchanged. Those code paths are out of scope for this fix.

## Deliberately Deferred: `mask_buffer_ptr` Port

DeepEP includes a hang-recovery system controlled by `enable_shrink=True` on the Buffer
constructor. When enabled, a `mask_buffer_ptr` is allocated and `is_rank_masked()` can
return `true` for ranks that are unreachable, allowing the collective to proceed without
them. This is a separate mechanism from empty-input handling.

SGLang creates the UCCL EP buffer without `enable_shrink=True`. As a result,
`mask_buffer_ptr = nullptr` and `is_rank_masked()` always returns `false` in production.
This system is not what makes DeepEP handle empty input; it's a separate concern for
tolerating unreachable ranks.

Porting `mask_buffer_ptr` support to UCCL EP would require:

1. Kernel signature changes in `internode_ll.cu` to accept and check the mask pointer
2. Buffer class changes in `uccl_ep.cc` to allocate and expose the mask buffer
3. Python wrapper changes in `deep_ep_wrapper/` to surface `enable_shrink` to callers
4. SGLang calling `clean_mask_buffer()` between forward passes

This is deferred as a separate hardening plan. The current fix (T2-T4) addresses the
immediate `launch_server` deadlock without requiring any of the above.

## Verification

Verification for this fix is code-level only. The development machine (A100) cannot
compile or run UCCL EP, which requires Hopper (sm_90) or an AMD-supported architecture.

Evidence files are in `.sisyphus/evidence/`:

- `task-1-sentinel-verification.md` (or `q1-sentinel-verification.md`) — T1 kernel audit
- `task-2-asserts.txt` — T2 assert changes at lines 640 and 732
- `task-3-disjunction.txt` — T3 disjunction at line 521
- `task-4-launch-checks.txt` — T4 `cudaGetLastError` additions (5 calls)

Runtime cluster validation is the user's responsibility on a Hopper cluster after
applying this patch. Recommended test: run SGLang `launch_server` with
`--moe-a2a-backend deepep` and trigger `forward_idle` by having heterogeneous DP batch
sizes across ranks (e.g., send requests to only one DP rank while others are idle).

## Files Modified

- `ep/src/uccl_ep.cc` — 3 commits on branch `fix/sglang-forward-idle-empty-input`:
  - `79b894c fix(ep): allow num_tokens=0 in intranode_prepare and intranode_dispatch host wrappers` (T2)
  - `fd10252 fix(ep): permit topk_idx=nullptr when num_tokens=0 in get_dispatch_layout host wrapper` (T3)
  - `1a1f683 chore(ep): mirror LL host wrapper's cudaGetLastError pattern in intranode launches` (T4)

## Files Deliberately NOT Modified

- `ep/src/internode.cu` — internode-normal mode; out of scope for this fix
- `ep/src/internode_ll.cu` — LL kernels verified empty-safe by T1 (VP3-VP7); no changes needed
- `ep/src/intranode.cu` — intranode kernels verified empty-safe by T1 (VP1-VP2); no changes needed
- `ep/src/layout.cu` — `get_dispatch_layout` kernel loop is bounded by `num_tokens`; no changes needed
- `ep/src/proxy.cpp`, `ep/src/rdma.cpp`, `ep/src/fifo.cpp`, `ep/src/uccl_proxy.cpp` — CPU proxy layer; read-only for VP6/VP7 verification
- `ep/src/uccl_ep.cc:882`, `ep/src/uccl_ep.cc:985` — internode-normal asserts preserved; out of scope
- `~/sglang/**` — out of scope per user decision; SGLang calls the UCCL EP API correctly
- `~/DeepEP/**` — used as reference only for parity checks during T1 audit

## Next Steps

1. Apply this patch to a Hopper cluster and run `launch_server` with heterogeneous DP
   batches to confirm the fix eliminates the `forward_idle` deadlock.

2. If hang-recovery for unreachable ranks is needed in production, create a separate plan
   for porting the `mask_buffer_ptr` / `enable_shrink` system (see "Deliberately Deferred"
   section above).

3. Consider upstreaming this fix to the UCCL EP main branch via a pull request, with the
   T1 kernel-safety audit as supporting evidence.
