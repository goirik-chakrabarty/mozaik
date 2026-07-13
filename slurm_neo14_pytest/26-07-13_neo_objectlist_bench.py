import time, numpy as np, neo, quantities as pq
print("neo", neo.__version__)

def bench_append(N, kind):
    seg = neo.Segment()
    if kind == "spiketrain":
        objs = [neo.SpikeTrain(np.array([1.0, 2.0, 3.0]), t_stop=10.0, units="s") for _ in range(N)]
        target = seg.spiketrains
    else:  # analogsignal (Vm-like: 500 samples x 1 channel)
        arr = np.random.rand(500, 1).astype("float64")
        objs = [neo.AnalogSignal(arr, units="mV", sampling_rate=1 * pq.kHz) for _ in range(N)]
        target = seg.analogsignals
    t0 = time.perf_counter()
    for o in objs:
        target.append(o)
    dt = time.perf_counter() - t0
    print(f"  append {N:>6} {kind:<12}: {dt:7.3f} s  ({dt/N*1e6:7.2f} us/append)  "
          f"container={type(target).__module__}.{type(target).__name__}")
    return dt

print("=== container type for Segment.spiketrains ===")
print("  ", type(neo.Segment().spiketrains))
for N in (5000, 20000, 40000):   # 40000 ~ the ~37500 neurons a blank get_data pulls
    print(f"--- N={N} ---")
    bench_append(N, "spiketrain")
bench_append(2000, "analogsignal")   # Vm: fewer recorded, heavier objects
