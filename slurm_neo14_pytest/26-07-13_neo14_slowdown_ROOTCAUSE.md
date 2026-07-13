# neo14 P1 slowdown — investigation status (2026-07-13)  [CORRECTED]

## Symptom (CONFIRMED, faithful conditions)
P1 test3 nt12 blank get_data ~158s (neo14, job 14841732) vs golden ~76s
(mozaik-opt.sif = neo 0.12, old pyNN, job 14614503). NEST sim identical (72s video).
S500 ~1.5x slow. Overhead is in get_data/serialization/gather, NOT compute.
NOTE: golden vs current differ in BOTH pyNN (old vs new) AND neo (0.12 vs 0.14.4).

## What was RULED OUT (micro-benchmarks; correcting an earlier wrong conclusion)
- neo 0.14 `SpikeTrainList.append` IS O(N^2) (117->470->962 us/append; 38.5s @ 40k)
  BUT the NEST path does NOT use it: pyNN `_get_spiketimes` returns a TUPLE, so
  `_get_current_segment` takes the bulk `SpikeTrainList.from_spike_time_array` branch
  (pyNN/recording/__init__.py:337), not the per-append branch (:315). So append O(N^2)
  is real but off the NEST hot path. [earlier "append is the cause" = RETRACTED]
- `from_spike_time_array` is lazy: instant to build; materializing (list(stl)) = 5.3s
  @ 37500 but CACHES (2nd call 0.002s). pickle(stl)=1.5s == pickle(plain list)=1.5s.
  => real neo spiketrain overhead ~7s (one-time materialize + pickle), NOT the +80s.

## Still OPEN — where the +80s actually is
Not yet pinned. Candidates: new pyNN `_get_all_signals` Vm path ("JACOMMENT: very
expensive", pyNN/nest/recording.py), AnalogSignal ObjectList handling at scale, or
mozaik's per-presentation gather. DEFINITIVE next step = cProfile the REAL get_data in a
short sim (not micro-benchmarks): wrap models/__init__.py:180-189 or run
`python -m cProfile -o prof run.py ...` on a 1-2 stimulus test3 and read the top cumulative.

## Evidence artifacts (this dir)
- 26-07-13_neo_objectlist_bench.py  (append O(N^2) + lazy-list findings)
- golden: ../slurm_scale/mozaik-scale-nt12-14614503_12.err (neo12, ~76s blanks, 16:55)
- neo14 faithful: repro-test3-14841732_0.err (158s blanks)
- neo12-override import fail: neo12ovr-test3-14841899.err (code needs neo 0.13+)
