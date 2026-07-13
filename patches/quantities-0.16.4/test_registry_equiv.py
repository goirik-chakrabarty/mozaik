"""Hash-independent behavior-equivalence smoke for the memoized quantities registry.
Prints only process-stable facts (values, dimensionality strings, identity-stability,
cross-equality). Output MUST be byte-identical between patched and unpatched."""
import quantities as pq
from quantities.registry import unit_registry
import neo, numpy as np

labels = ['dimensionless', 's', 'ms', 'mV', 'Hz', 'kg', 'm/s', 'g/cc', 'V', 'A', 'uS', 'nA', 'in',
          '', 'percent', 'g/cc', 's', 'dimensionless']  # repeats exercise the cache
print("== registry lookups ==")
for lab in labels:
    try:
        v = unit_registry[lab]
        same = unit_registry[lab] is v
        # equality against a fresh parse of the same normalized text (semantics, not identity)
        print(f"{lab!r:14} dim={str(v.dimensionality):12} mag={float(v.magnitude):g} "
              f"id_stable={same} eq_pq_s={bool(v==pq.s)}")
    except Exception as e:
        print(f"{lab!r:14} ERROR {type(e).__name__}: {e}")

print("== consistency: lookup == module unit ==")
for name in ['s','ms','mV','Hz','kg','V','A']:
    print(f"{name:6} unit_registry==pq.{name}: {bool(unit_registry[name]==getattr(pq,name))}")

print("== neo spiketrain merge roundtrip ==")
sts = [neo.SpikeTrain(np.array([1.0*i, 2.0*i]), t_stop=100.0, units='s') for i in range(1, 6)]
seg = neo.Segment()
for st in sts: seg.spiketrains.append(st)
print("n_spiketrains =", len(seg.spiketrains), "units =", str(seg.spiketrains[0].units.dimensionality),
      "rescale_ms_ok =", str(sts[0].rescale('ms').units.dimensionality))
print("DONE")
