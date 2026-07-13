"""Phase-2 (zero-copy) feasibility probe.

Two load-bearing questions for spec §14 (Approach B):
  A) Can we read raw safetensors bytes and rebuild mx.array bit-IDENTICAL to
     mx.load — incl. quantized uint32 weights + float16 scales/biases? (the gate)
  B) Does parallel F_NOCACHE read-into-buffer + array-build beat MLX's cold read
     (~2.86 GB/s), approaching the ~8.7 GB/s raw ceiling?
"""
import os, sys, json, time, fcntl, glob
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mlx.core as mx

F_NOCACHE = 48
PACK = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/ssathananthan/Project/Expirments/qwen2-test2-32b/packed"
layers = sorted(glob.glob(os.path.join(PACK, "layer_*.safetensors")))

# safetensors dtype string -> numpy dtype (bit-exact; no arithmetic)
ST_TO_NP = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U64": np.uint64, "U32": np.uint32, "U16": np.uint16, "U8": np.uint8,
    "BOOL": np.bool_,
    # BF16 has no native numpy dtype: read as uint16, bitcast to mx.bfloat16.
}


def read_bytes(path, nocache):
    fd = os.open(path, os.O_RDONLY)
    try:
        if nocache:
            fcntl.fcntl(fd, F_NOCACHE, 1)
        chunks = []
        while True:
            b = os.read(fd, 1 << 23)
            if not b:
                break
            chunks.append(b)
        return b"".join(chunks)
    finally:
        os.close(fd)


def parse_header(buf: bytes):
    n = int.from_bytes(buf[:8], "little")
    header = json.loads(buf[8:8 + n])
    return n, header


def build_arrays(buf: bytes):
    """Reconstruct {name: mx.array} from a full safetensors file buffer."""
    n, header = parse_header(buf)
    base = 8 + n
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt, shape, (beg, end) = meta["dtype"], meta["shape"], meta["data_offsets"]
        raw = buf[base + beg: base + end]
        if dt == "BF16":
            a = mx.array(np.frombuffer(raw, np.uint16).reshape(shape))
            a = a.view(mx.bfloat16)  # bitcast (verify on a bf16 pack)
        else:
            a = mx.array(np.frombuffer(raw, ST_TO_NP[dt]).reshape(shape))
        out[name] = a
    return out


# ---- A) bit-identical gate on a real 4-bit layer ----
f = layers[0]
buf = read_bytes(f, nocache=False)
rec = build_arrays(buf)
ref = mx.load(f)
print(f"=== A) bit-identical reconstruction: {os.path.basename(f)} ({len(ref)} tensors) ===")
all_ok = True
for k in ref:
    r, g = ref[k], rec.get(k)
    if g is None:
        print(f"  MISSING {k}"); all_ok = False; continue
    same_meta = (str(r.dtype) == str(g.dtype)) and (tuple(r.shape) == tuple(g.shape))
    # exact byte compare (handles NaN, unlike == / array_equal)
    same_bytes = np.array(r).tobytes() == np.array(g).tobytes()
    ok = same_meta and same_bytes
    all_ok &= ok
    if not ok:
        print(f"  MISMATCH {k}: dtype {r.dtype}/{g.dtype} shape {tuple(r.shape)}/{tuple(g.shape)} bytes={same_bytes}")
extra = set(rec) - set(ref)
if extra:
    print(f"  EXTRA keys not in mx.load: {extra}"); all_ok = False
print(f"  -> {'ALL BIT-IDENTICAL' if all_ok else 'MISMATCH(ES) FOUND'}  ({len(ref)} tensors checked)\n")

# ---- B) throughput: parallel F_NOCACHE read-into-buffer (+ build) vs MLX cold ----
print("=== B) throughput (cold; disjoint layers per measurement) ===")

def mlx_cold(path):
    d = mx.load(path); mx.eval(list(d.values()))
    return sum(int(v.nbytes) for v in d.values())

def zc_read_only(path):
    return len(read_bytes(path, nocache=True))

def zc_read_and_build(path):
    b = read_bytes(path, nocache=True)
    arrs = build_arrays(b)
    mx.eval(list(arrs.values()))
    return sum(int(v.nbytes) for v in arrs.values())

def bench(name, files, fn, threads):
    t0 = time.time()
    if threads == 1:
        tot = sum(fn(x) for x in files)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            tot = sum(ex.map(fn, files))
    dt = time.time() - t0
    print(f"  {name:38s} {tot/1e9:5.2f} GB  {dt:5.2f}s  {tot/1e9/dt:5.2f} GB/s")

bench("MLX cold mx.load+eval (1 thread)", layers[8:14], mlx_cold, 1)
bench("zero-copy read-only F_NOCACHE (8 thr)", layers[16:24], zc_read_only, 8)
bench("zero-copy read+build F_NOCACHE (8 thr)", layers[32:40], zc_read_and_build, 8)
bench("zero-copy read+build F_NOCACHE (4 thr)", layers[40:46], zc_read_and_build, 4)
print("\n(zero-copy read+build vs MLX cold = the Phase-2 effective speedup)")
