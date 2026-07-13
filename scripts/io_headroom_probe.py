"""I/O headroom probe for NunSpark streaming.

Question the whole 'Advanced I/O' design forks on:
  Does the SSD have bandwidth the current SINGLE-threaded reader leaves unused?

Two independent measurements:
  A) RAW reads with F_NOCACHE (true cold disk reads, repeatable on same files)
     at 1/2/4/8 threads -> physical disk ceiling vs single-stream.
  B) MLX path (mx.load + mx.eval) on COLD disjoint layer sets at 1/2/4 threads
     -> can we exploit headroom THROUGH mlx, or does mlx serialize?
"""
import os, sys, time, fcntl, glob
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = 48  # macOS fcntl: disable data caching for this fd

PACK = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/ssathananthan/Project/Expirments/qwen2-test2-32b/packed"
layers = sorted(glob.glob(os.path.join(PACK, "layer_*.safetensors")))
assert len(layers) >= 32, f"need >=32 layers, found {len(layers)}"


def read_raw(path, nocache=True, chunk=8 << 20):
    fd = os.open(path, os.O_RDONLY)
    try:
        if nocache:
            fcntl.fcntl(fd, F_NOCACHE, 1)
        total = 0
        while True:
            b = os.read(fd, chunk)
            if not b:
                break
            total += len(b)
        return total
    finally:
        os.close(fd)


def run(files, nthreads, fn):
    t0 = time.time()
    if nthreads == 1:
        totals = [fn(f) for f in files]
    else:
        with ThreadPoolExecutor(max_workers=nthreads) as ex:
            totals = list(ex.map(fn, files))
    dt = time.time() - t0
    gb = sum(totals) / 1e9
    return gb, dt, gb / dt


print(f"pack: {PACK}\nlayers: {len(layers)}  (~{os.path.getsize(layers[0])/1e6:.0f} MB each)\n")

# ---- B) MLX path on COLD disjoint sets (run FIRST while pages are cold) ----
import mlx.core as mx

def read_mlx(path):
    d = mx.load(path)
    vals = list(d.values())
    mx.eval(vals)
    return sum(int(v.nbytes) for v in vals)

print("=== B) MLX path (mx.load + mx.eval), COLD disjoint layers ===")
print(f"{'threads':>8} {'GB':>7} {'sec':>7} {'GB/s':>7}")
mlx_sets = {1: layers[0:6], 2: layers[6:12], 4: layers[12:18]}
mlx_res = {}
for nt in (1, 2, 4):
    gb, dt, bw = run(mlx_sets[nt], nt, read_mlx)
    mlx_res[nt] = bw
    print(f"{nt:>8} {gb:>7.2f} {dt:>7.2f} {bw:>7.2f}")

# ---- A) RAW F_NOCACHE on the SAME set (always true disk reads) ----
print("\n=== A) RAW reads, F_NOCACHE (true disk ceiling), same 8 layers ===")
print(f"{'threads':>8} {'GB':>7} {'sec':>7} {'GB/s':>7}")
raw_files = layers[24:32]
raw_res = {}
for nt in (1, 2, 4, 8):
    gb, dt, bw = run(raw_files, nt, read_raw)
    raw_res[nt] = bw
    print(f"{nt:>8} {gb:>7.2f} {dt:>7.2f} {bw:>7.2f}")

# ---- verdict ----
print("\n=== VERDICT ===")
raw_gain = raw_res[8] / raw_res[1]
mlx_gain = mlx_res[4] / mlx_res[1]
print(f"raw  8-thread / 1-thread = {raw_gain:.2f}x   (physical disk headroom)")
print(f"mlx  4-thread / 1-thread = {mlx_gain:.2f}x   (exploitable through mlx)")
if raw_gain < 1.15:
    print(">> SSD already saturated by one reader. Parallel I/O CANNOT help.")
    print(">> Only lever left: read FEWER bytes (quant / MoE / acceptance).")
elif mlx_gain >= 1.3:
    print(">> Headroom exists AND mlx can exploit it -> multi-thread PieceCache wins.")
else:
    print(">> Headroom exists but mlx serializes -> must warm page cache OFF the")
    print(">> mlx path (parallel raw pre-reads), then mx.load hits warm pages.")
