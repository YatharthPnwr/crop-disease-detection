"""Efficiency table for the app models: params, MACs, file sizes, CPU latency (batch 1).

    python scripts/latency.py --models-dir app/models --out reports/efficiency.md
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from cropdoc.export import load_checkpoint
from cropdoc.models import DISPLAY_NAMES, count_params


def _median_ms(fn, warmup=10, runs=50) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", default="app/models")
    ap.add_argument("--out", default="reports/efficiency.md")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for both runtimes")
    args = ap.parse_args()

    import onnxruntime as ort
    from fvcore.nn import FlopCountAnalysis

    torch.set_num_threads(args.threads)
    x = torch.randn(1, 3, 224, 224)
    rows = []
    for pt in sorted(Path(args.models_dir).glob("*.pt")):
        model, _, cfg = load_checkpoint(pt)
        flops = FlopCountAnalysis(model, x)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        with torch.no_grad():
            torch_ms = _median_ms(lambda: model(x))
        row = {
            "model": DISPLAY_NAMES.get(cfg["model"], cfg["model"]),
            "params_M": round(count_params(model) / 1e6, 2),
            "gmacs": round(flops.total() / 1e9, 2),  # fvcore counts one fused multiply-add as one flop
            "pt_MB": round(pt.stat().st_size / 1e6, 1),
            "torch_ms": round(torch_ms, 1),
        }
        onnx = pt.with_suffix(".onnx")
        if onnx.exists():
            so = ort.SessionOptions()
            so.intra_op_num_threads = args.threads
            sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
            feed = {sess.get_inputs()[0].name: x.numpy().astype(np.float32)}
            row["onnx_MB"] = round(onnx.stat().st_size / 1e6, 1)
            row["onnx_ms"] = round(_median_ms(lambda: sess.run(None, feed)), 1)
        rows.append(row)
        print(json.dumps(row))

    lines = [
        "# Efficiency (laptop CPU, batch 1)",
        "",
        f"Median of 50 runs after 10 warm-ups, {args.threads} CPU threads. GMACs at 224x224 (fvcore).",
        "",
        "| Model | Params (M) | GMACs | .pt (MB) | .onnx (MB) | PyTorch (ms) | ONNX Runtime (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['params_M']} | {r['gmacs']} | {r['pt_MB']} | {r.get('onnx_MB', '–')} "
                     f"| {r['torch_ms']} | {r.get('onnx_ms', '–')} |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
