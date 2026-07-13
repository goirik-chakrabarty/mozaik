# quantities 0.16.4 perf patch — memoize bare-name unit-registry lookups

**What / why.** In the neo14 P1 `get_data` cross-rank merge (and in connector construction),
`quantities.dimensionality.Dimensionality.__hash__` calls `hash(unit_registry['dimensionless'])`
on **every** hash. `UnitRegistry.__getitem__` re-runs a full `regex-normalize + ast.parse +
compile + eval` on each call. cProfile of the patched-neo sim (perf1, nt12, rank-0) after the neo
`_contains` O(1) fix:

| function | ncalls | cumtime |
|---|---|---|
| `dimensionality.py:59 __hash__` | 4.55 M | 184 s |
| `registry.py:72 __getitem__` (outer) | 4.88 M | 168 s |
| `registry.py:29 __getitem__` (inner ast.parse/eval) | 4.88 M | 142 s |

`print_callers` shows **all 4.55 M** outer lookups come from `__hash__`, and the label is always
`'dimensionless'` (a bare name). This is pure redundant re-parsing of a constant.

**The fix.** Memoize `UnitRegistry.__getitem__` results **only for bare-name labels**
(`^[A-Za-z_][A-Za-z0-9_]*$`). A bare name parses to a single `ast.Name` and `eval`s to the ONE
pre-registered singleton object — identity-stable across calls — so returning a cached reference is
byte-for-byte behavior-identical to re-parsing. Compound expressions (`'m/s'`, `'g/cc'`) `eval` to a
**fresh** object each call, so caching them would change object identity; they deliberately fall
through to the original path. `LookupError` also propagates uncached, so a unit registered later
still resolves. The registry is an append-only singleton (inner `__setitem__` forbids redefinition),
so a cached bare name never becomes stale.

**Correctness.** `test_registry_equiv.py` (hash-independent fingerprints: dimensionality strings,
magnitudes, identity-stability, cross-equality, neo spiketrain roundtrip) is **byte-identical**
patched vs unpatched, including repeated (cache-exercising) and compound lookups. Gated on the
30/30 `tests/full_model` reference tests (spikes + voltages) with this patch bind-mounted over the
neopatch SIF.

**Files.** `registry.py` (patched) · `registry.orig.py` (pristine 0.16.4) · `registry.py.diff`
(reviewable unified diff) · `test_registry_equiv.py` (equivalence smoke, in the staging dir).

**Same pattern as** `../neo-0.14.4/` (the `_contains` O(1) fix) — an algorithmic/caching change in a
vendored dep, correctness-neutral, candidate for upstream contribution. Do NOT open a PR without
sign-off.
