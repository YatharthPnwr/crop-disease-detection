# Efficiency (laptop CPU, batch 1)

Median of 50 runs after 10 warm-ups, 4 CPU threads. GMACs at 224x224 (fvcore).

| Model | Params (M) | GMACs | .pt (MB) | .onnx (MB) | PyTorch (ms) | ONNX Runtime (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Custom CNN | 2.36 | 2.83 | 9.5 | 9.5 | 24.2 | 13.6 |
| EfficientNet-B0 | 4.06 | 0.4 | 16.5 | 16.2 | 15.1 | 4.4 |
