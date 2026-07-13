"""Verify the load-bearing assumption of the parallel-warm design:
   raw-reading a layer file into the OS page cache makes the subsequent
   mx.load + mx.eval of that file fast (memory-speed, not disk-speed).

If true: a parallel raw-reader pool warming upcoming layers -> mx.eval becomes
nearly free -> effective read BW jumps from ~2.5 to ~8+ GB/s.
"""
import os, sys, time, glob
from concurrent.futures import ThreadPoolExecutor
import mlx.core as mx

PACK = "/Users/ssathananthan/Project/Expirments/qwen2-test2-32b/packed"
layers = sorted(glob.glob(os.path.join(PACK, "layer_*.safetensors")))

def raw_warm(path, chunk=8 << 20):
    # NO F_NOCACHE: we WANT to populate the page cache
    fd = os.open(path, os.O_RDONLY)
    try:
        total = 0
        while True:
            b = os.read(fd, chunk)
            if not b:
                break
            total += len(b)
        return total
    finally:
        os.close(fd)

def mlx_eval(path):
    d = mx.load(path)
    vals = list(d.values())
    mx.eval(vals)
    return sum(int(v.nbytes) for v in vals)

# 1) COLD mx.load+eval baseline (layer 40)
t0 = time.time(); n = mlx_eval(layers[40]); dt = time.time()-t0
print(f"COLD  mx.load+eval        : {n/1e9:.2f} GB  {dt:.3f}s  {n/1e9/dt:.2f} GB/s")

# 2) WARM single: raw-read layer 41 then mx.load+eval it
t0 = time.time(); raw_warm(layers[41]); dtw = time.time()-t0
t0 = time.time(); n = mlx_eval(layers[41]); dt = time.time()-t0
print(f"warm-read layer           : {dtw:.3f}s")
print(f"WARM  mx.load+eval (cached): {n/1e9:.2f} GB  {dt:.3f}s  {n/1e9/dt:.2f} GB/s")

# 3) End-to-end: parallel raw-warm 6 layers (8 threads) THEN serial mx.eval them.
#    This mimics the proposed engine: warmer pool fills page cache fast,
#    compute thread mx.evals from warm cache. Measure total effective BW.
warm_set = layers[42:48]
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(raw_warm, warm_set))
warm_dt = time.time() - t0
t0 = time.time()
tot = sum(mlx_eval(p) for p in warm_set)
eval_dt = time.time() - t0
print(f"\nPIPELINE on 6 layers ({tot/1e9:.2f} GB):")
print(f"  parallel raw-warm (8 thr): {warm_dt:.3f}s  ({tot/1e9/warm_dt:.2f} GB/s)")
print(f"  serial mx.eval (warm)    : {eval_dt:.3f}s  ({tot/1e9/eval_dt:.2f} GB/s)")
print(f"  >> if overlapped, the warm BW ({tot/1e9/warm_dt:.2f} GB/s) is the new ceiling")
