# IndoML Track 2

**Innovators · Raj Patil (solo) · Datathon@IndoML 2026**

Final **GAPPolar90** submission **982179**: **17.2313 Combined**, rank **7** on the Codabench leaderboard checked 30 October 2026. ASR audit passed on 100 hidden clips with no transcript mismatches. [Official competition](https://www.codabench.org/competitions/17835/).

This repository contains the final trained enhancement checkpoint (88.5 MB), training source and data source pins, a standalone inference entry point, separate training/inference requirements, and the three-page report. Competition audio and transcripts are not redistributed.

[Read the three-page report](report.pdf)

## Inference

Use **Linux x86_64, Python 3.11, NVIDIA CUDA**. The competition runs used PyTorch 2.6.0 with CUDA 12.4 on two Tesla T4 GPUs (16 GB each). A single T4 can process clips sequentially; `--gpus 0,1` splits enhancement across two GPUs. ASR runs sequentially on the first selected GPU. Keep sufficient disk space for input data, the gated ASR download, and WAV output. FFmpeg and libsndfile are useful for external-data preparation; final WAV I/O itself uses Python's PCM reader.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_torch.txt
python -m pip install -r requirements_inference.txt
python inference.py --verify-checkpoint
```

Obtain access to [SraVaani-1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0), then set `HF_TOKEN` securely in the environment (or use a notebook secret). Do not put tokens in source files or command history. The pinned model uses its reviewed remote modeling code through `trust_remote_code=True`; exact revision and artifact hashes are checked before loading.

Place the organizer's **original test WAVs directly in one directory**. Inputs must already be mono PCM16 at 16 kHz, preserving original filenames and sample counts.

```bash
python inference.py --input /path/to/test_wavs --output runs/final --gpus 0,1 --official
```

Output: `runs/final/prediction.zip`, containing 3,178 root-level WAVs and `transcripts.jsonl`; `RUN_COMPLETED.json` records counts and hashes. `--official` requires 3,178 clips. Omit it for a small smoke test. Add `--enhance-only` to produce WAVs without ASR/ZIP. Valid completed enhancement files are reused after a restart; mismatched inputs/code are rejected. Existing prediction ZIPs are never overwritten.

The final recipe is the **saved Balance token-step-600 model**, Context20 decoding, PCM16 serialization, a 128-sample shifted second pass averaged 50:50 with Context20, another PCM16 boundary, and magnitude/phase projection with both strengths 0.9. All clips use the same recipe. SraVaani transcribes the **final serialized WAV**, with unchanged official decoding and no text edits.

## Training

Install `requirements_train.txt` after the CUDA requirements. [TRAINING.md](TRAINING.md) gives complete data preparation and selected-lineage replay commands. The original loss, update, source selection, frozen teacher, and geometry implementations are retained in `experiments/`. Training requires gated Rasa access and organizer-provided input folders for overlap exclusion; these files are not public dependencies of inference.

The concise `train.py` replays the historically selected 1,000/800/600-update stages. It does not rerun every rejected candidate or retroactively call a retrained model the submitted model. GPU training may differ numerically across environments. **Use the bundled SHA-bound checkpoint for final inference reproduction.**

## Results

| Variant | Full validation Combined | Official Phase 2 Combined |
|---|---:|---:|
| GAPPredictor pretrained | 19.0517 | 15.4822 |
| GAPAdapt1 | 19.5011 | 16.2547 |
| GAPNoise1 | 19.8223 | 16.6068 |
| GAPBalance1 | 19.9536 | Not submitted directly |
| GAPContext20 | 20.0819 | 17.0198 |
| GAPShiftMean | 20.3529 | 17.1901 |
| **GAPPolar90** | **20.4934** | **17.2313** |

Final official SI-SDR is **14.417813 dB**, pooled ΔWER **2.813475 percentage points**, synthetic ΔWER **8.323597 pp**, natural ΔWER **0.216354 pp**. Combined = synthetic SI-SDR + pooled ΔWER in percentage points. Validation and hidden-test scores are different measurements. Validation was reused across experiments; these are sequential development results, not independent causal ablations. Aggregate scorer evidence is in `results/`.

## Reproducibility and limitations

- Final checkpoint SHA-256: `4629c2d75492644265c42cc93780770039d562cc9288fb1ee15520e0ceaca825`.
- Original submitted prediction ZIP SHA-256: `968bb55046e91f7e3ea1d2f285cd86a428472f8a72b5ff782d3825d763f4433d`.
- Release checks: strict checkpoint loading, numerical geometry/PCM tests, packaging tests, Python syntax and CLI checks. The consolidated release wrapper was not rerun on all 3,178 clips; the historical Kaggle stages produced the audited official result. Cross-device bitwise identity is not claimed.
- Full validation used 2,501 clips (695 synthetic / 1,806 natural); hidden test used 3,178 (1,099 synthetic / 2,079 natural). No hidden clean speech or reference transcript was used in training.
- No changes to the mandated ASR model weights or final decoder. Training-only frozen posterior matching is distinct from inference.
- The 852-second Polar90 Kaggle submission run reused already enhanced ShiftMean audio. It is **not** the end-to-end runtime of this script.

## Attribution

We build on the public [GAP-URGENet Predictor](https://huggingface.co/Xiaobin-Rong/gap-urgenet), with the TFGridNet/ESPnet architecture, [SraVaani](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0), [Rasa](https://huggingface.co/datasets/ai4bharat/Rasa), and [MS-SNSD](https://github.com/microsoft/MS-SNSD). We do not claim to invent these models or PCGrad. Our work is their task-specific adaptation and the documented reconstruction-priority gradient and inference recipe. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`.
