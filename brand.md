# Brand — Crop Disease Detection

_Status: active_

Direction: a **botanical lab notebook**. Warm paper surfaces, deep leaf-green ink, a
literary serif for headlines and monospaced numerals for every measurement, so results
read like entries in a field journal. Restrained: one accent (leaf green), colour only
where it carries meaning.

## Palette (tokens in `app/web/styles.css`)

| Role | Light | Dark |
| --- | --- | --- |
| Background | `#f6f3ec` | `#0f1412` |
| Surface (cards) | `#fffdf8` | `#151c19` |
| Surface 2 (bands, tracks) | `#efe9dc` | `#1b2420` |
| Ink / secondary / tertiary | `#1c2620` / `#4c5850` / `#5f6a62` | `#e6ebe4` / `#a8b2aa` / `#8f9a92` |
| Lines | `#e0d9c9`, strong `#c9c0ad` | `#26302b`, strong `#36423b` |
| Primary (buttons, highlights) | `#2f5d3a` on `#fbf8f1` | `#9cc98f` on `#0f1412` |
| Warning (unsure, unverified) | `#8a4b00` on `#fbeed6` | `#f0b866` on `#2a2113` |
| Focus ring | `#2563c9` | `#7fb0f5` |

Chart series (validated with the dataviz palette checker: lightness band, chroma,
CVD separation, contrast on each surface):

| Series | Light | Dark |
| --- | --- | --- |
| Custom CNN | `#2f6db3` | `#5b9be6` |
| EfficientNet-B0 | `#c4561c` | `#e0742f` |

## Typography

- Display: **Fraunces** (headlines, diagnosis names), italic for emphasis.
- UI text: **Instrument Sans**.
- Numbers and labels: **JetBrains Mono**, tabular figures.

## Voice

Plain, specific and honest about limits: "Unsure, please retake the photo", not
"Error". Numbers always carry their unit and their source (seeds, test-set size).
