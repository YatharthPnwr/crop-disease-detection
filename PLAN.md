# Explainable Crop Disease Detection — Project Plan

Nov 26, 2026 · @Yatharth Singh Panwar

We train two CNNs on PlantVillage and compare them: a custom CNN built from scratch, and an ImageNet-pretrained EfficientNet-B0. We explain their predictions with Grad-CAM and ship a local Streamlit app that names the disease from a leaf photo. The build takes one day, with training on Modal GPUs. No accuracy numbers for our models appear here; items marked (verify) must be checked before they go in the report.

## At a glance

The project trains two models on one dataset. Both models are available in the app.

**Models used**

| Role | Model | Init | Params (38 classes, approx.) | In the app |
| --- | --- | --- | --- | --- |
| Baseline | Custom CNN (5 conv blocks + GAP) | Random: trained from scratch | \~2.4M (est.) | Yes, second option |
| Main model | EfficientNet-B0 | ImageNet-pretrained (torchvision) | \~4.1M | Yes, default |

**Dataset used**

| Dataset | Use | Size | Classes | Background | Source |
| --- | --- | --- | --- | --- | --- |
| [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) (color + segmented) | Train / val / test, 70/15/15, grouped by pHash | \~54.3k images, 256x256 | 38 (14 crops) | Uniform lab background | Kaggle abdallahalidev/plantvillage-dataset (verify slug) |

The team dropped two items from scope to save time: the PlantDoc field test and the ResNet50 ± CBAM attention ablation.

## 1. Problem, objectives, scope

A farmer needs a quick answer to "what is wrong with this leaf?". We build a CNN classifier that gives that answer from a photo. We also measure how much transfer learning helps over a CNN trained from scratch, and show where the model looks.

| # | Objective | Evidence in report |
| --- | --- | --- |
| O1 | Classify 38 PlantVillage classes with a CNN | Test accuracy and macro-F1, mean ± std over 3 seeds |
| O2 | Compare a custom CNN (from scratch) with pretrained EfficientNet-B0 | One table: accuracy, macro-F1, params, FLOPs, size, CPU latency, training time |
| O3 | Explain predictions with Grad-CAM | CAM grids, plus a score for how much of each heatmap falls on the leaf |
| O4 | Local web app | Streamlit app: camera or upload, top-3, Grad-CAM, disease card, "unsure" state |

**In scope:**

- Single-leaf, image-level classification of the 38 PlantVillage classes
- Supervised training on PlantVillage
- CPU inference on a laptop

**Out of scope:**

- **Real field photos (PlantDoc).** Dropped for time. The app warns that accuracy on field photos will be lower.
- **Attention ablation (ResNet50 ± CBAM).** Dropped for time. EfficientNet-B0 still contains SE attention inside its blocks.
- **Other tasks:** detection (YOLO), severity estimation, multi-leaf images.
- **LLM-generated advice.** Treatment text comes from a static table with a cited source for each entry.
- **Deployment beyond the laptop:** cloud hosting, a native mobile app.

## 2. Repo study

None of the seven repos uses a leakage-safe split or reports variance across seeds. Several use different preprocessing at training and serving time. Those gaps shape our design. We read each repo's README and main code; the exception is the two-branch repo, whose internals live in the external K-CAI library.

| Repo | Architecture | Data / split | Borrow | Limitations found in the code | We do differently |
| --- | --- | --- | --- | --- | --- |
| [manthan89-py/Plant-Disease-Detection](https://github.com/manthan89-py/Plant-Disease-Detection) | 4 x \[Conv3x3-ReLU-BN\]x2 + MaxPool (32→256 ch), then Flatten(50,176) → FC1024 → FC39. 52.6M params, almost all in the first FC layer | Mendeley PlantVillage, 39 classes incl. background, 61,486 images. Unseeded random split, about 60/25/15 | Flask app idea; disease\_info.csv and supplement\_info.csv lookups indexed by class | No normalisation or augmentation; Adam defaults; 5 epochs. Training resizes then centre-crops, but serving squashes to 224x224. Uploads saved under their raw filenames | Custom CNN with a GAP head (about 2.4M params). Seeded grouped split. One shared preprocess() for training and serving |
| [AhmadJeddi/Plant-Disease-Classification](https://github.com/AhmadJeddi/Plant-Disease-Classification) | ResNet50 + DCBAM on the last bottleneck of layer3 and layer4 | PlantVillage official train/test lists; stratified 90/10 train/val, seed 42; test 10,709 | Seeding; best-loss and best-acc checkpoints; confusion-matrix output | 99.65% acc / 99.47% macro-F1 from one seed. model.train() each epoch un-freezes the "frozen" BN layers. Adam with L2 | Frozen BN kept in eval mode explicitly. AdamW. 3 seeds |
| [joaopauloschuler/two-branch-plant-disease](https://github.com/joaopauloschuler/two-branch-plant-disease) | Inception V3 split in Lab colour: L branch with 20% of filters, AB branch with 80% (internals in K-CAI, not read) | PlantVillage, 38 classes, 60/20/20 split (seed 7) | Evidence that colour carries the disease signal, so we keep hue augmentation small | 99.48% on PlantVillage ([MENDEL 2022](https://doi.org/10.13164/mendel.2022.1.055) abstract). TensorFlow + K-CAI; README says the old code no longer runs | Cited for the colour argument; not reimplemented |
| [kruthi-sb/leaf\_disease\_detection](https://github.com/kruthi-sb/leaf_disease_detection) | YOLOv5s detector | PlantDoc detection set, 29 classes | Streamlit file\_uploader UI pattern | No metrics. Labels paired with images by os.listdir order, not filename. JPG only. Gemini writes the remedies | Classification instead of detection. Static treatment table. Accepts jpg/png/webp |
| [BS0111/PlantAttentionCBAM](https://github.com/BS0111/PlantAttentionCBAM) ([arXiv 2512.17864](https://arxiv.org/abs/2512.17864)) | VGG16 with CBAM after each of 5 stages | Paper: 5 datasets, 80/20, no val set; the code actually loads Plant Pathology 2020 | Hook-based Grad-CAM idea | 98.72% on PlantVillage from one run. Paper and code disagree (r=8 vs 16). Deprecated hooks. CAMs judged only by eye | pytorch-grad-cam library; a quantitative leaf-vs-background score; a val set |
| [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) | Dataset | 2,598 real field images, 13 species | – | Web-scraped, so expect label noise and low resolution | Not used: field testing is out of scope. Listed as future work |
| [sumanismcse/Plant-Disease-Identification-using-CNN](https://github.com/sumanismcse/Plant-Disease-Identification-using-CNN) | Keras CNN: 32→64→128 conv, Dense1024 | 200 images per class; unstratified 80/20; the test set doubles as val | Augmentation list | binary\_crossentropy with a softmax output; divides by 225, not 255; no results reported | Cross-entropy loss, a separate val set, the full dataset |

## 3. Dataset

We use only PlantVillage. It is split by groups of near-duplicate leaf photos, so the test score measures generalisation rather than memorisation.

**PlantVillage**

- **Source:** Kaggle abdallahalidev/plantvillage-dataset, color/ and segmented/ folders (verify the slug).
  - This copy includes segmented images with a black background, which give free leaf masks for the Grad-CAM check.
  - Fallback: a sparse checkout of spMohanty raw/color, with HSV masks instead.
- **Size:** about 54.3k JPEGs, 256x256, 38 classes, 14 crops. The spMohanty README says 54,306 images; the CBAM paper says 54,303.
- **Imbalance:** about 35:1 between the largest class (Orange HLB, \~5.5k) and the smallest (Potato healthy, \~150) (verify with dataset\_stats.py). Macro-F1 is therefore the primary metric.
- **Leakage warning:**
  - PlantVillage often holds several photos of the same physical leaf. A random per-image split puts siblings in both train and test, so a "99.9%" score is partly memorisation.
  - Never use the Mendeley "with augmentation" copy: it contains augmented duplicates.

**Safe split (grouped, stratified)**

1. Compute a 64-bit pHash (imagehash) for every image.
2. Within each class, link image pairs whose Hamming distance is ≤ 6. Check 30 linked pairs by eye and adjust the threshold. Union-find turns the links into groups.
3. Run StratifiedGroupKFold for a 70/15/15 train/val/test split, keeping each group in one split only. The split uses seed 42, separate from the training seeds.
4. Commit data/splits/pv\_split.csv (path, class, group, split). Report the group count and the group-size histogram.
5. **Optional check:** train one EfficientNet-B0 on a plain random split and compare its score with the grouped split.

## 4. Architecture study

We compare a small CNN, designed by us and trained from scratch, with EfficientNet-B0 pretrained on ImageNet. Together they show what transfer learning and a modern architecture add.

- **Weights:** EfficientNet-B0 uses torchvision's weights= API (EfficientNet\_B0\_Weights.IMAGENET1K\_V1), not the deprecated pretrained=True.
- **Numbers:** params and GMACs are at 224x224. The 38-class figures are approximate; measured values from fvcore/torchinfo go in the report.

| Model | Params (38 classes) | GMACs @224 | Key ideas | Why it is here |
| --- | --- | --- | --- | --- |
| Custom CNN (ours) | \~2.4M (est.) | \~2.8 (est.) | VGG-style 3x3 conv stacks, batch norm, max pooling, dropout, GAP head | From-scratch baseline. Shows the core CNN building blocks and measures what transfer learning buys |
| EfficientNet-B0 (Tan & Le 2019) | \~4.1M (5.3M with the ImageNet head) | 0.39 | Depthwise separable conv, inverted residuals, SE attention, SiLU, compound scaling | Main model: high accuracy per FLOP, fast on a laptop CPU for the app |

### 4.1 Layer structure

Input is 224x224x3. Shapes are written H x W x C.

**Custom CNN (ours).** Each block is \[Conv3x3 → BN → ReLU\] x2 → MaxPool2x2.

| Stage | Output | Params (est.) |
| --- | --- | --- |
| Block 1 (3→32→32) | 112x112x32 | \~10k |
| Block 2 (32→64→64) | 56x56x64 | \~55k |
| Block 3 (64→128→128) | 28x28x128 | \~221k |
| Block 4 (128→256→256) | 14x14x256 | \~885k |
| Block 5 (256→256→256), Grad-CAM target | 7x7x256 | \~1.18M |
| GAP → Dropout 0.3 → FC 256→38 | 38 | \~10k |

Why this design:

- Two stacked 3x3 convs see a 5x5 area with fewer parameters than one 5x5 conv.
- BN after every conv keeps training from scratch stable.
- The GAP head avoids a huge flatten-to-FC layer (manthan's has about 51M params in that one layer).

**EfficientNet-B0.** Its building block is MBConv-t:

- **Expand:** a 1x1 conv widens the channels by a factor t.
- **Depthwise:** a kxk depthwise conv filters each channel separately.
- **SE attention** (ratio 0.25): GAP → two small FC layers → sigmoid weights that rescale each channel.
- **Project:** a linear 1x1 conv narrows the channels again.
- **Skip:** a residual connection when the input and output shapes match.

B0 is the base network found by architecture search. B1–B7 scale depth, width and resolution together by a compound coefficient φ.

| Stage | Op | k | Channels | Repeats | Output |
| --- | --- | --- | --- | --- | --- |
| Stem | Conv3x3/2 | 3 | 32 | 1 | 112x112 |
| 1 | MBConv1 | 3 | 16 | 1 | 112x112 |
| 2 | MBConv6 | 3 | 24 | 2 | 56x56 |
| 3 | MBConv6 | 5 | 40 | 2 | 28x28 |
| 4 | MBConv6 | 3 | 80 | 3 | 14x14 |
| 5 | MBConv6 | 5 | 112 | 3 | 14x14 |
| 6 | MBConv6 | 5 | 192 | 4 | 7x7 |
| 7 | MBConv6 | 3 | 320 | 1 | 7x7 |
| Head | Conv1x1 1280 (Grad-CAM target) → GAP → Dropout 0.2 → FC 1280→38 |  | 38 |  |  |

### 4.2 Why only these two

- **The pair covers every core CNN concept.** The custom CNN shows convolution, pooling, BN, dropout and a GAP head built by hand. EfficientNet-B0 adds residual connections, depthwise separable convolution, attention (SE) and compound scaling.
- **One comparison answers the main question.** Custom CNN vs EfficientNet-B0 shows how much pretraining and a modern design help: accuracy, F1, size, speed and training time side by side.
- **EfficientNet-B0 fits the app.** It has about 4M params and 0.39 GMACs, so it runs quickly on a laptop CPU. ResNet50 (about 24M params, 4.1 GMACs) was dropped to save time.

## 5. Deep learning concepts map

Every core DL topic in the course appears in one of the two models or in the training pipeline. The table says where each one is used, and why.

| Concept | Where used | Why |
| --- | --- | --- |
| Convolution (3x3, 1x1, depthwise) | Custom CNN: 3x3 stacks. EfficientNet: 1x1 expand/project and depthwise kxk | Local, translation-equivariant features. 1x1 mixes channels; depthwise cuts FLOPs about k²-fold |
| Pooling (max, global average) | Custom CNN max-pooling; GAP in both heads; GAP inside SE | Downsampling and invariance. GAP replaces huge FC layers and makes Grad-CAM-style explanation natural |
| Batch norm | Both models; frozen in eval mode during phase 1 of fine-tuning | Stable training at higher LR. Frozen statistics must not drift (the AhmadJeddi bug) |
| Dropout | Before the classifier: 0.3 custom, 0.2 EfficientNet | Regularises the freshly initialised head |
| Activations | ReLU (custom); SiLU (EfficientNet); sigmoid (SE gates); softmax (output) | Non-linearity; sigmoid gives 0–1 attention weights |
| Residual connections | Skip connections in EfficientNet's MBConv blocks | An identity path keeps gradients flowing through deep nets |
| Attention | SE blocks inside every EfficientNet MBConv | Re-weights channels so informative features count more |
| Transfer learning | EfficientNet-B0 starts from ImageNet weights | 38k training images is small; ImageNet edge and texture filters transfer. The custom CNN measures the benefit |
| Fine-tuning (two-phase, discriminative LR) | EfficientNet-B0 (sections 6 and 7) | Phase 1 trains the new head without wrecking features; phase 2 adapts everything gently |
| Cross-entropy + label smoothing 0.1 | Both models | Stops overconfident logits, so the app's "unsure" threshold is better calibrated (Müller et al. 2019) |
| Focal loss / class-weighted CE | Optional ablation | Targets minority classes under 35:1 imbalance |
| AdamW | Both models | Decoupled weight decay (Loshchilov & Hutter 2019) instead of Adam + L2 |
| LR schedule: warmup + cosine | Phase 2 and the custom CNN | Warmup avoids large early updates; cosine needs no step tuning |
| Regularisation | Weight decay, dropout, label smoothing, augmentation, early stopping | Lab data is easy to overfit |
| Augmentation | Section 7 | Simulates lighting, angle and blur changes in real photos |
| Early stopping | Patience 3 on val macro-F1; keep the best checkpoint | Saves GPU minutes. Label smoothing shifts loss values, so we track F1 instead of loss |
| Mixed precision (bf16) | All GPU training | About 2x throughput on Ampere/Ada GPUs; bf16 needs no GradScaler |

## 6. Hyperparameters

EfficientNet-B0 uses AdamW with two learning rates, 1e-3 for the head and 1e-4 for the backbone. It fine-tunes in two phases: 2 epochs training only the head, then 8 epochs training everything. The custom CNN trains for 20 epochs at a single LR of 1e-3. Both run in bf16 with seeds 0, 1 and 2.

| Hyperparameter | Value | Justification | Would tune |
| --- | --- | --- | --- |
| Image size | 224 (train: RandomResizedCrop from 256; eval: Resize 256 → CenterCrop 224) | ImageNet resolution; PlantVillage is natively 256x256 | 192 / 288 |
| Batch size | 64 | Fits a 24 GB A10G in bf16; stable BN statistics | 128, with linear LR scaling |
| Optimiser | AdamW (betas 0.9 / 0.999) | Robust defaults; decoupled weight decay | SGD + momentum 0.9 for the custom CNN |
| LR, EfficientNet head | 1e-3 | A new layer needs larger steps | 3e-4 to 3e-3 |
| LR, EfficientNet backbone (phase 2) | 1e-4 (10x lower) | Protects the pretrained features | 3e-5 to 3e-4 |
| LR, custom CNN | 1e-3, all layers | Trained from scratch | 3e-3 with OneCycle |
| Weight decay | 0.01, none on BN or bias | Standard for AdamW | 1e-4, 0.05 |
| Epochs | EfficientNet: 2 head-only + 8 full. Custom CNN: 20 | PlantVillage converges fast; one-day budget | More if val F1 is still rising |
| Scheduler | 1-epoch warmup → cosine to 0 (EfficientNet phase 2, custom CNN) | Gentle start, no step points to tune | OneCycle |
| Dropout | 0.3 custom, 0.2 EfficientNet | The custom CNN has more room to overfit | 0 to 0.5 |
| Label smoothing | 0.1 | Calibration for the "unsure" threshold | 0, 0.05, 0.2 |
| Augmentation | RRC scale (0.6, 1.0); H+V flip; rotation ±30°; ColorJitter (brightness 0.3, contrast 0.3, saturation 0.2, hue 0.02); GaussianBlur p 0.2; RandomErasing p 0.25 | See section 7 | Stronger RRC |
| Freezing (EfficientNet) | Phase 1: backbone frozen, BN in eval mode. Phase 2: everything unfrozen | Random-head gradients would wreck pretrained features | Gradual unfreezing |
| Precision | bf16 autocast, channels\_last, cudnn.benchmark | Speed | fp16 + GradScaler on a T4 |
| Seeds | 0, 1, 2 (data split fixed at seed 42) | Mean ± std measures model variance only | – |
| Early stopping | Patience 3 on val macro-F1 | Saves GPU time | – |

## 7. Training pipeline

One preprocessing function is shared by training, evaluation, export and the app. Each of the 6 training runs gets its own detached Modal GPU container, so all of them train in parallel.

**Preprocessing** (cropdoc.data.preprocess()):

- Apply ImageOps.exif\_transpose first, because phone photos carry an orientation tag.
- Convert to RGB, resize the shorter side to 256, then centre-crop to 224.
- Normalise with the ImageNet mean and std.
- The same function runs at training and serving time, avoiding manthan's train/serve mismatch.

**Augmentation, and why each suits leaves:**

| Transform | Why |
| --- | --- |
| RandomResizedCrop, scale 0.6–1.0 | Photos are framed loosely. The 0.6 lower bound keeps small lesions in view |
| H + V flip, rotation ±30° | Leaves have no fixed orientation |
| ColorJitter: brightness, contrast, saturation | Sun, shade and camera white balance vary |
| Hue jitter only 0.02 | Colour carries the disease signal (yellowing, browning). A big hue shift could make a healthy leaf look diseased |
| GaussianBlur, p 0.2 | Phone focus blur |
| RandomErasing, p 0.25 | Leaves or fingers partly covering the leaf |

**Dataloaders.**

- Each container copies one dataset tar from the Modal Volume and extracts it to local SSD. Reading many small files straight from a Volume is slow.
- The dataset reads its file list and labels from pv\_split.csv.
- Settings: num\_workers 8, pin\_memory, persistent\_workers.

**Two-phase loop (EfficientNet-B0):**

```python
# phase 1: head only, backbone BN frozen
for p in model.features.parameters(): p.requires_grad = False
model.train(); model.features.apply(lambda m: m.eval() if isinstance(m, nn.BatchNorm2d) else None)  # re-apply EVERY epoch
# phase 2: unfreeze, two param groups
opt = AdamW([{"params": model.features.parameters(), "lr": 1e-4},
             {"params": model.classifier.parameters(), "lr": 1e-3}], weight_decay=0.01)
```

The custom CNN trains in a single phase, with all layers at 1e-3.

**Checkpointing.**

- Each run saves best.pt (by val macro-F1) and last.pt to Volume cropdoc-runs/\<run\_id>/.
- A checkpoint holds the state\_dict, config, class list, epoch and metrics.
- last.pt lets a timed-out run resume.

**Logging.**

- TensorBoard logs go to the Volume and are viewed locally after download.
- Each run writes metrics.json; scripts/aggregate.py collects them.
- No W&B: it needs another account and secret, with no gain in one day.

**Reproducibility.**

- Seed random, numpy and torch. Use the fixed split CSV and pinned requirements.
- Save the config inside each checkpoint, and log the git commit hash.
- cudnn.benchmark gives up bitwise determinism for speed. The report therefore claims statistical reproducibility (mean ± std) and says so.

**Modal layout** (modal\_app.py):

- **Image:** Python 3.11 with torch, torchvision, grad-cam, imagehash, scikit-learn, tensorboard, onnx and onnxruntime.
- **Volumes:** cropdoc-data and cropdoc-runs.
- **Secret:** kaggle, holding the Kaggle API token.
- **Functions:**
  - download(): CPU.
  - make\_splits(): 8 CPU.
  - train(cfg): gpu="A10G" or "L4", 8 CPU, 32 GB RAM, 3 h timeout.
  - evaluate(run\_id).
  - export(run\_id): returns model bytes directly, because reading a Volume from the laptop is slow.
- **Launch:** `modal run --detach modal_app.py::grid` calls train.map(configs). Detached runs survive a laptop shutdown.

## 8. Experiment design

The core grid is 6 runs launched in parallel: 2 models x 3 seeds. We estimate about 1.5–2.5 A10G GPU-hours, roughly $2–3 at about $1.1/h (check modal.com/pricing). Re-check after the smoke run.

| Group | Runs | Seeds | Purpose |
| --- | --- | --- | --- |
| E1 Comparison | Custom CNN, EfficientNet-B0 | 3 each = 6 | O1, O2 |
| E2 Leakage check (optional) | EfficientNet-B0 on a plain random split | 1 | Shows why the grouped split matters |

**Metrics for every run:**

- **PlantVillage test:** accuracy, macro-F1 (primary), per-class precision and recall, and a confusion matrix.
- **Efficiency** (once per model, on the laptop CPU):
  - params, MACs (fvcore), and .pt and .onnx file sizes;
  - batch-1 latency: median of 50 runs after 10 warm-ups, for both PyTorch and ONNX Runtime;
  - training time per epoch and epochs to the best val F1.
- **Aggregation:** mean ± std over the 3 seeds.

**Expected results** (literature ranges, not our numbers):

- **What the repos report:** about 99.5% for pretrained CNNs on random or ungrouped splits. AhmadJeddi 99.65% (one seed), two-branch 99.48%, CBAM paper 98.72%.
- **What we expect:** our grouped split is harder. EfficientNet-B0 should land in the high 90s; the custom CNN somewhat lower, needing more epochs to get there.
- **Most confused classes:** look for visually similar diseases on the same crop, e.g. tomato bacterial spot vs Septoria, or early vs late blight.

**What would be suspicious, and what to check:**

| Observation | Check |
| --- | --- |
| Test = 100.00%, or test clearly above val | Leakage or a split bug: re-check the group assignment |
| Custom CNN ≥ EfficientNet-B0 | Pretrained weights not loaded, or wrong normalisation |
| EfficientNet-B0 below 90% | LR too high, the BN-freeze bug, or a preprocessing mismatch |
| Std above 1 point | Unstable training: check warmup and LR |
| One class near 0 recall | A label or folder-name bug, or too few images for that class |

## 9. Explainability

We run Grad-CAM and Grad-CAM++ on the last 7x7 conv stage of both models. A leaf-mask score then checks whether each model looks at the leaf or at the background.

**Method choice.**

- **Library:** [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam). It is tested and lets us pick the target class, unlike BS0111's hand-written hooks, which use a deprecated API.
- **Grad-CAM++:** it weights positive gradients per pixel, so it handles several separate lesion spots better than plain Grad-CAM.
- **Layer:** the last conv stage, the most semantic layer that still keeps spatial layout.

| Model | Target layer |
| --- | --- |
| Custom CNN | Last conv block, before GAP |
| EfficientNet-B0 | model.features\[-1\] (the 1x1 conv to 1280 channels) |

**Leaf-vs-background check.**

- **Leaf mask:** threshold the non-black pixels of each test image's segmented PlantVillage twin. Verify the filename pairing first. The fallback is an HSV green/brown mask, labelled approximate.
- **Leaf energy ratio:** the share of CAM mass that falls inside the leaf mask. The chance level is the leaf's share of the image area.
- **Pointing game:** whether the CAM's highest point falls inside the mask.
- **Coverage:** about 500 test images per model, compared between the custom CNN and EfficientNet-B0.
- **Background-bias test:** classify the segmented (black-background) version of each test image. A large drop in accuracy means the model relied on background cues.
  - [Noyan 2022](https://arxiv.org/abs/2206.04374) reports above-chance accuracy from a few background pixels alone (verify the exact numbers before citing).
- **Qualitative grids:** 8 correct and 8 wrong predictions per model, with both CAM types. Add a few phone photos taken through the app.
- **Sanity check:** CAMs from a randomly re-initialised head must look different (Adebayo et al. 2018, verify).

## 10. Deployment and web interface

The app is built in Streamlit and runs on the laptop. ONNX Runtime makes the predictions and eager PyTorch draws the Grad-CAM overlays. Disease advice comes from a static, sourced lookup table, not an LLM.

**Export** (export.py):

- torch.onnx.export, opset 17, dynamic batch size.
- Check that ONNX Runtime and PyTorch logits match (allclose, atol 1e-3).
- Stretch: INT8 quantisation with onnxruntime.quantization for EfficientNet-B0. Report the change in size, latency and val accuracy.
- Grad-CAM needs gradients and module hooks, which neither ONNX nor TorchScript provides. So the app also loads the eager model from the same checkpoint.

**Why Streamlit, not Flask.**

- Pure Python, with built-in file\_uploader, camera\_input and cache\_resource. It can be built in hours.
- Flask would need templates and JavaScript for the same UI.
- Streamlit has no REST endpoints. A stretch app/api.py (FastAPI) could add POST /predict (image → top-3 JSON), GET /models and GET /health.

**Camera.** st.camera\_input uses the browser's camera.

- **Laptop webcam:** works on localhost with no setup.
- **Phone on the same Wi-Fi:** browsers allow the camera only over HTTPS. Either take the photo on the phone and upload it, or run Streamlit with a self-signed certificate (server.sslCertFile and server.sslKeyFile).

**App structure.**

- app/streamlit\_app.py: the UI.
- app/inference.py: load\_model(name), predict(img) → top-3, and gradcam(img, cls) → overlay.
- app/models/: the .onnx and .pt files plus classes.json.
- data/disease\_info.json: the static disease lookup table.

**Wireframe:**

```text
┌ Sidebar ───────────────┐ ┌ Main ───────────────────────────────────────────────┐
│ Model: [EffNet-B0 v]│ │ [ Upload leaf photo ]  or  [ Take photo ]         │
│ [x] Show Grad-CAM   │ │ ┌ input ──────┐ ┌ Grad-CAM overlay ┐               │
│ CAM: (o)CAM ( )++   │ │ │            │ │                  │               │
│ Threshold: 0.60     │ │ └────────────┘ └──────────────────┘               │
│ About / limits      │ │ 1. Tomato - Early blight   ########.  82%         │
└─────────────────────┘ │ 2. Tomato - Septoria       #........   9%         │
                        │ 3. Potato - Early blight   .........   4%         │
                        │ [ Disease card: symptoms, cause, management, src ]│
                        │ ! Trained on lab images; field accuracy is lower  │
                        └───────────────────────────────────────────────────┘
```

**"Unsure, retake photo" state.**

- **Trigger:** top-1 probability below τ, or a top-1 minus top-2 margin below 0.15.
- **Choosing τ:** use PlantVillage val, never test, e.g. the confidence where val precision ≥ 0.98. The default is 0.6.
- **Shown instead:** tips (one leaf, fill the frame, daylight, plain background), with the top-3 greyed out.

**Disease info.**

- disease\_info.json is keyed by PlantVillage class. Fields: crop, disease, symptoms, cause, management, source\_url.
- The team writes the entries from extension or university sources, each with a citation.
- Healthy classes show a "no disease detected" card.

**Error handling and validation.**

- **Accepted files:** jpg, png and webp, up to 10 MB (Streamlit server.maxUploadSize).
- **Image checks:** PIL verify(), then re-open. Apply exif\_transpose and convert RGBA or greyscale to RGB. Reject images under 64 px.
- **Leaf heuristic:** under 5% green or brown pixels shows a "no leaf detected?" warning.
- **Failures:** a missing model file shows a clear message, not a traceback. Inference sits in try/except and reports through st.error.
- **Privacy:** images stay in memory and are never written to disk.

## 11. Repo layout, requirements, commands

The code is one Python package, one Modal app and one Streamlit app. Dataset images are never committed to git.

```text
deeplearning/
├─ PLAN.md  README.md  requirements.txt  requirements-app.txt
├─ configs/      # base.yaml, custom.yaml, effb0.yaml
├─ data/         # disease_info.json, splits/pv_split.csv
├─ src/cropdoc/  # data, models, train, evaluate, explain, export, utils
├─ modal_app.py  # download, make_splits, train(.map), evaluate, export
├─ scripts/      # make_splits, dataset_stats, aggregate, latency, make_figures
├─ app/          # streamlit_app.py, inference.py, models/
├─ tests/        # test_preprocess (train/serve parity), test_shapes, test_splits
└─ reports/      # results.csv, figures/, report.pdf, slides.pdf
```

**Dependencies** (pin exact versions at setup):

- **requirements.txt:** torch, torchvision, grad-cam, imagehash, scikit-learn, pandas, numpy, pillow, pyyaml, tensorboard, fvcore, onnx, onnxruntime, modal, matplotlib, seaborn, pytest.
- **requirements-app.txt:** CPU builds of torch and torchvision, onnxruntime, grad-cam, streamlit, pillow, numpy, opencv-python-headless.

**README outline:**

1. Title and one-line result
2. Demo GIF
3. Problem
4. Dataset and download steps
5. Leakage-safe split
6. Models and results table
7. Explainability
8. Training on Modal
9. Running the app
10. Limitations
11. Citations
12. Team

**Commands:**

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
modal secret create kaggle KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"   # never echo the token
modal run modal_app.py::download && modal run modal_app.py::make_splits
modal run modal_app.py::smoke                      # 1 epoch, EfficientNet-B0, 5% data: timing + sanity
modal run --detach modal_app.py::grid              # 2 models x 3 seeds in parallel
modal run modal_app.py::evaluate_all && modal run modal_app.py::export
python scripts/aggregate.py && python scripts/latency.py && pytest -q
streamlit run app/streamlit_app.py
```

## 12. Timeline (one day, about 10 working hours) and risks

The 6 training runs launch by hour 2. The GPUs train while the team builds the app, the Grad-CAM code and the report. Roles assume 3–4 people; merge them if the team is smaller.

| Hours | A: ML / Modal | B: App | C: Data / report | D: XAI / slides |
| --- | --- | --- | --- | --- |
| H0–1 | Set up venv, Modal and the Kaggle secret; download PlantVillage to the Volume | Streamlit skeleton with a dummy model | Start disease\_info.json (38 entries, each with a source) | Grad-CAM notebook on an ImageNet model |
| H1–2 | make\_splits (pHash) + dataset\_stats; smoke run, then update epochs and cost | Upload, camera and input validation | Finish disease\_info.json | Report skeleton: intro, related work |
| H2–4 | Launch the detached grid (6 runs); monitor; re-run any failures | inference.py (ONNX + Grad-CAM); "unsure" logic | Methods section: dataset, split, models | Leaf energy ratio code, tested on the first checkpoint |
| H4–5 | evaluate\_all + aggregate; ONNX export; laptop latency | Plug in the real models | Results tables from results.csv | Grad-CAM grids |
| H5–7 | Bug fixes, repo clean-up, README; optional leakage run | Test on 10 phone photos; record the demo GIF | Results, discussion, limitations | Explainability section; slides |
| H7–9 | Everyone: full report review |  |  |  |
| H9–10 | Everyone: viva rehearsal; tag v1.0 |  |  |  |

**Risks and fallbacks**

| Risk | Fallback |
| --- | --- |
| Kaggle download fails | Sparse-clone spMohanty raw/color; use HSV masks |
| Modal GPUs queue | Runs still finish, just later. Try gpu="L4" |
| Epochs slower than estimated | Usually data loading: check worker CPU use. Drop to 5 phase-2 epochs |
| pHash grouping slow or over-merging | Adjust the threshold. Worst case: random split, with the limitation stated clearly |
| Custom CNN doesn't converge | LR 3e-4 and more epochs. A weak baseline is still a valid result |
| Grad-CAM slow on CPU in the app | Compute the CAM only when the toggle is on |
| Phone camera blocked over plain HTTP | Upload the photo instead, or enable HTTPS (section 10) |
| Laptop dies mid-run | Training is detached on Modal; commit often |

## 13. Report, presentation, viva

The report and slides lead with the custom CNN vs EfficientNet-B0 comparison and the app demo. Every number in them must trace back to results.csv.

**Report (8–10 pages):**

1. Abstract
2. Introduction: why leaf-disease detection matters
3. Related work: the repos above
4. Data: PlantVillage and the leakage-safe split
5. Methods: the two architectures, losses, two-phase fine-tuning, augmentation
6. Setup: hyperparameter table, Modal hardware, seeds
7. Results:
   1. Accuracy and macro-F1 (mean ± std)
   2. Per-class results and the confusion matrix
   3. Efficiency: size, latency, training time
   4. Optional leakage check
8. Explainability: CAM grids, leaf energy ratio, background test
9. Deployment: app architecture and screenshots
10. Discussion and limitations, including lab-only training
11. Conclusion and future work: field testing on PlantDoc, attention modules
12. References

**Slides (12):**

1. Title
2. Problem
3. Dataset
4. Leakage and the split
5. The two models
6. Training recipe
7. Results
8. Confusion matrix
9. Grad-CAM
10. App demo
11. Limitations and future work
12. Q&A

**15 likely viva questions:**

| # | Question | Short answer |
| --- | --- | --- |
| 1 | Will it work on real farm photos? | Probably worse. It was trained only on lab photos with plain backgrounds. The app warns about this, and field testing on PlantDoc is future work |
| 2 | What is the leakage and how did you prevent it? | Near-duplicate photos of the same leaf can land in both train and test. We grouped them by pHash and split by group |
| 3 | Why transfer learning? | 38k training images is small. ImageNet edge and texture filters transfer; the custom CNN measures the benefit |
| 4 | Why two-phase fine-tuning? | A random head sends large gradients that would wreck the pretrained features. So: train the head first, then everything with a 10x lower backbone LR |
| 5 | Why freeze BN in phase 1? | Its running stats would drift while the weights stay frozen. AhmadJeddi's code does this by accident |
| 6 | Why did you design the custom CNN this way? | Stacked 3x3 convs, BN after each, max pooling, and a GAP head to keep params low (about 2.4M vs manthan's 52.6M) |
| 7 | Why is depthwise separable convolution cheaper? | A kxk depthwise conv plus a 1x1 conv replaces one full conv, saving about k² (8–9x) in FLOPs |
| 8 | What is compound scaling? | EfficientNet scales depth, width and resolution together with fixed ratios |
| 9 | Does your model use attention? | Yes: EfficientNet-B0 has SE blocks that reweight channels using global context |
| 10 | Why EfficientNet-B0 and not ResNet50? | About 6x fewer params and 10x fewer FLOPs, so it is fast on a laptop CPU. ResNet50 was dropped to save time |
| 11 | Why macro-F1? | With 35:1 imbalance, accuracy is dominated by the big classes. Macro-F1 weights every class equally |
| 12 | What does label smoothing do? | Targets become 0.9 and 0.1/(K−1) instead of one-hot. This stops overconfidence and calibrates the "unsure" threshold |
| 13 | How does Grad-CAM work? Why the last conv layer? | Each channel's weight is its average gradient; CAM = ReLU(Σ w·A). The last conv layer is the most semantic one that keeps spatial layout |
| 14 | How do you know the model looks at the leaf? | Leaf energy ratio and the pointing game against segmentation masks, plus the black-background accuracy test |
| 15 | What happens with a photo of something that isn't a leaf? | The model still picks a class. The confidence threshold and a green-pixel check show "unsure" or "no leaf detected?" instead |

## Checklist

Tick items as you finish them.

**Setup (H0–2)**

- [ ] venv and requirements installed; Modal token works; Kaggle secret created (token never printed)
- [ ] PlantVillage (color + segmented) on Volume cropdoc-data
- [ ] pHash groups built; 30 linked pairs checked by eye; pv\_split.csv committed
- [ ] No path or group appears in two splits (test\_splits passes)
- [ ] dataset\_stats gives class counts and the imbalance ratio
- [ ] test\_preprocess passes: training eval transform matches the app transform
- [ ] Smoke run done; epoch time measured; epochs and cost updated

**Training (H2–5)**

- [ ] Grid launched with --detach: custom CNN x3 seeds, EfficientNet-B0 x3 seeds
- [ ] BN frozen in eval mode in phase 1 (asserted in code)
- [ ] Every run has best.pt and metrics.json; failed runs re-run
- [ ] evaluate\_all: test accuracy, macro-F1, per-class report, confusion matrix
- [ ] results.csv as mean ± std; checked against the suspicious-results table

**Explainability (H4–7)**

- [ ] Grad-CAM and Grad-CAM++ grids for both models, correct and wrong predictions
- [ ] Leaf energy ratio and pointing game for both models
- [ ] Black-background accuracy test
- [ ] Randomised-head sanity check

**Deployment (H4–7)**

- [ ] ONNX exported; ONNX Runtime matches PyTorch (atol 1e-3)
- [ ] Laptop CPU latency and model size table
- [ ] App features: camera and upload, model selector, top-3, Grad-CAM toggle
- [ ] App info and safety: sourced disease card, "unsure" state with τ from val, input validation, no disk writes
- [ ] App gives the same top-1 as evaluate.py on 20 fixed test images
- [ ] Tested on 10 real phone photos; demo GIF recorded

**Deliverables (H5–10)**

- [ ] README with run commands and citations
- [ ] Report: every number traces to results.csv; every (verify) item checked or removed
- [ ] Slides done; viva Q&A rehearsed; repo tagged v1.0

## Sources

**Repos reviewed:**

- [manthan89-py/Plant-Disease-Detection](https://github.com/manthan89-py/Plant-Disease-Detection)
- [AhmadJeddi/Plant-Disease-Classification](https://github.com/AhmadJeddi/Plant-Disease-Classification)
- [joaopauloschuler/two-branch-plant-disease](https://github.com/joaopauloschuler/two-branch-plant-disease)
- [kruthi-sb/leaf\_disease\_detection](https://github.com/kruthi-sb/leaf_disease_detection)
- [BS0111/PlantAttentionCBAM](https://github.com/BS0111/PlantAttentionCBAM)
- [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) (reviewed; not used)
- [sumanismcse/Plant-Disease-Identification-using-CNN](https://github.com/sumanismcse/Plant-Disease-Identification-using-CNN)

**Dataset:**

- [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)

**Papers:**

- [Interpretable Plant Leaf Disease Detection Using Attention-Enhanced CNN](https://arxiv.org/abs/2512.17864) — Singh, Sharma, Dey, 2025
- [Color-aware two-branch DCNN for efficient plant disease classification](https://doi.org/10.13164/mendel.2022.1.055) — MENDEL 2022
- [PlantDoc: A Dataset for Visual Plant Disease Detection](https://arxiv.org/abs/1911.10317) — Singh et al., 2020 (future work)

Several papers are cited from memory and must be verified before they are quoted in the report:

- EfficientNet (Tan & Le 2019)
- SE (Hu et al. 2018)
- Grad-CAM (Selvaraju et al. 2017)
- Grad-CAM++ (Chattopadhay et al. 2018)
- Label smoothing (Müller et al. 2019)
- AdamW (Loshchilov & Hutter 2019)
- Noyan 2022
- Adebayo et al. 2018
