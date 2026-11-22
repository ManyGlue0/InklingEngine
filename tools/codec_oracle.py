"""Check InklingEngine's quant decoders against gguf-py's reference dequantizers.

usage: python tools/codec_oracle.py <model.gguf> <codectest-out dir> [path/to/gguf-py]
First run `inkling codectest <model.gguf> <dir>`; it writes the Rust-decoded rows of a few tensors per format.
gguf-py comes with llama.cpp (llama.cpp/gguf-py); the default path is reference/src/gguf-py.
"""
import os
import sys

import numpy as np

if len(sys.argv) < 3:
    sys.exit(__doc__)
model, out = sys.argv[1], sys.argv[2]
gguf_py = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "reference", "src", "gguf-py")
sys.path.insert(0, os.path.abspath(gguf_py))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

reader = GGUFReader(model)
tensors = {t.name: t for t in reader.tensors}
worst = 0.0
for line in open(os.path.join(out, "index.tsv")):
    name, file, nrows, k = line.split("\t")
    nrows, k = int(nrows), int(k)
    t = tensors[name]
    raw = np.asarray(t.data)
    rows = raw.reshape(-1, raw.shape[-1])[:nrows]
    ref = dequantize(rows, t.tensor_type).reshape(nrows, k).astype(np.float32)
    ours = np.fromfile(os.path.join(out, file), dtype=np.float32).reshape(nrows, k)
    mag = float(np.abs(ref).max())
    err = float(np.abs(ref - ours).max())
    worst = max(worst, err / max(mag, 1e-30))
    print(f"{t.tensor_type.name:8} {name:40} max |ours - gguf-py| = {err:.3e} (max |w| {mag:.3e})")
print(f"worst relative error: {worst:.2e}")
sys.exit(0 if worst < 1e-5 else 1)
