# neo 0.14.4 patch — O(1) `ObjectList._contains` (fixes O(N²) P1 `get_data`)

**What:** replaces `neo/core/objectlist.py`'s `_contains` (and adds bookkeeping in `_handle_append` +
cache invalidation in both `__setitem__`s) so identity-membership is **O(1)** instead of an O(N) rebuild
of `[id(x) for x in self._items]` on every append.

**Why:** on neo 0.13+, a `Segment`'s children live in `ObjectList`/`SpikeTrainList`. During pyNN's
cross-rank `gather_blocks` → `neo.core.container.merge`, spiketrains are appended one-by-one; each append
calls `_handle_append` → `_contains`, which was O(N) → **O(N²)** over the ~37,500 neurons a MOZAIK P1
presentation records. This ~halved P1 simulation throughput vs the neo 0.12 golden (test3 blank
`get_data` ~158 s vs ~76 s; total test3 nt12 ~2× 16:55). neo 0.12 used plain lists (O(1) append), so the
regression appeared with the neo14 migration. Root-cause profile + evidence:
`../../slurm_neo14_pytest/26-07-13_neo14_slowdown_ROOTCAUSE.md`.

**How it works:** `_contains` caches a `set` of `id()`s tagged by `(id(self._items), len(self._items))`.
It rebuilds (O(N)) only when the list is replaced, its length changes unexpectedly, or after unpickling
(the stored `id()` no longer matches). `_handle_append` adds the incoming object's id and bumps the tracked
length, so the append hot path stays O(1). `__setitem__` (same-length in-place replace) invalidates the
cache. Correctness (duplicate-rejection) is preserved.

**Verified:** append micro-bench O(N²)→O(1) (~420× at N=40k); test3 blank `get_data` 158 s→~76 s (= golden);
faithful test3 nt12 913 s ≈ golden 1015 s; **30/30 `tests/full_model` reference tests pass** (spikes+voltages).

**Files here:**
- `objectlist.py`, `spiketrainlist.py` — the full patched neo 0.14.4 files (dropped into the SIF).
- `objectlist.py.diff`, `spiketrainlist.py.diff` — unified diffs vs pristine neo 0.14.4 (for review / upstreaming).

**Applied by:** `mozaik/mozaik-opt-neo14_2026-07_neopatch.def` (`%post` copies these over the pip-installed
neo) → SIF `mozaik-sif/mozaik-opt-neopatch_2026-07-13.sif`. For iteration, bind-mount them over the SIF's
`neo/core/*.py`. **Candidate upstream contribution to python-neo — STOP before any PR.**
