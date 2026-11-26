"""Turn training log lines into one compact line per epoch (for watching Modal logs).

    tail -f logs/grid.log | python -u scripts/format_log.py
"""

import json
import re
import sys

PAT = re.compile(r"\[(\S+)\] (\{[^{}]*\})")

for line in sys.stdin:
    matches = list(PAT.finditer(line))  # interleaved logs can put two records on one line
    if not matches:
        print(line.rstrip()[:220], flush=True)
        continue
    for m in matches:
        r = json.loads(m.group(2))
        print(
            f"{m.group(1):<22} {r['phase']:<4} ep {r['epoch'] + 1:>2}  "
            f"train_acc {r['train_acc'] * 100:5.1f}  val_acc {r['val_acc'] * 100:6.2f}  "
            f"val_F1 {r['val_macro_f1'] * 100:6.2f}"
            + (f"  lab_F1 {r['val_f1_plantvillage'] * 100:6.2f}  field_F1 {r['val_f1_plantdoc'] * 100:6.2f}"
               if "val_f1_plantdoc" in r else "")
            + f"  {r['seconds']:.0f}s",
            flush=True,
        )
