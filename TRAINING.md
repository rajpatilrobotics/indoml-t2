# Training reproduction

The released checkpoint follows **GAPPredictor → GAPAdapt1 (1,000 updates) → GAPNoise1 hard (800) → GAPBalance1 token (600)**. Context20, ShiftMean and Polar90 are inference transformations with no further optimizer updates.

## Data and exclusions

External clean speech: `ai4bharat/Rasa`, revision `632f55c7ac590219d41cd7adffce5b440e4604f5`. Speech training languages: Hindi, Bengali, Telugu, Kannada; development languages: Marathi and Tamil. Source IDs, encoded hashes and decoded PCM hashes separate development from training. Speaker IDs were unavailable, so speaker-disjointness is not claimed. Rasa CC BY 4.0 attribution and access terms apply.

External noise: Microsoft MS-SNSD at revision `fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742`. Pinned source manifests in each stage's `data_pins.json` identify every shard/recording and hash. Noise development categories are excluded from training. Dataset audio is downloaded from its source, never included here.

The initial adaptation used 2,366 training speech recordings / 200 development, and 35 training noise recordings / 11 development. Noise/Balance used 3,422 / 480 speech and 78 / 27 noise recordings. The later external development set contains 240 selection and 240 confirmation mixtures, each with 100 hard, 100 moderate and 40 clean examples. Later selection/confirmation sets were reused; they are not fresh unseen holdouts.

Competition Vaani validation/test audio is used for **evaluation and ID/hash overlap exclusion**, not supervised training gradients. Prepare organizer-supplied directories; the clean validation folder may contain only the synthetic clean references. The preparer copies no transcripts. Supply complete manifests rather than empty placeholders:

```bash
python prepare_manifests.py --validation /path/to/validation_noisy \
  --validation-clean /path/to/validation_clean --test /path/to/test_noisy \
  --output runs/adapt/data/manifests
```

Accept Rasa and SraVaani access on Hugging Face, and supply `HF_TOKEN` securely through the environment. Source and access licenses remain the original providers'. The external checks reject exact ID/hash overlap but cannot exclude re-encoded or previously pretrained overlap.

## Replay the selected path

Run from the repository root with Python 3.11 and CUDA. These are expensive training operations; allow multiple GPU sessions and preserve each `runs/` folder. No custom runtime or quota cutoff is added. A single GPU is used by this selected-path driver.

```bash
python -m pip install -r requirements_torch.txt
python -m pip install -r requirements_train.txt
python train.py --stage adapt --run runs/adapt --prepare-data
python train.py --stage adapt --run runs/adapt

mkdir -p runs/noise/data/manifests
cp runs/adapt/data/manifests/*.jsonl runs/noise/data/manifests/
python train.py --stage noise --run runs/noise --parent-run runs/adapt --prepare-data
python train.py --stage noise --run runs/noise --parent-run runs/adapt

python train.py --stage balance --run runs/balance --parent-run runs/noise --prepare-data
python train.py --stage balance --run runs/balance --parent-run runs/noise
```

Each stage saves optimizer/RNG state every 25 updates and resumes with the same command. Data preparation validates source hashes and reuses verified sources. Balance links to the prepared Noise data; preserve that parent directory. `selected_predictor.pt` exports the fixed last step of the historical winning arm. Replaying the selected lineage is distinct from repeating model selection: quality on a new run is not guaranteed, and its checkpoint must be evaluated separately. The exact submitted weights are `checkpoints/polar90.pt`.

## Original methods retained

- `experiments/gap_adapt_v1/train.py`: hybrid compressed-STFT/waveform reconstruction loss; original adaptation loop and external loss selection.
- `experiments/gap_noise_v1/train.py`: paired control/hard curriculum updates and external noisy/clean evaluation functions.
- `experiments/gap_balance_v1/train.py` and `objective.py`: original recognition-balanced update, raw-gradient conflict projection and norm cap.
- `experiments/gap_token_pilot_v5/teacher.py`, `native_encoder.py`, `diagnostics.py`: frozen clean-path posterior targets and numerical backward checks. Not a replacement scoring decoder.

All stages use 4-second crops, accumulation of four microbatches, FP32 AdamW (`lr=1e-5`, `betas=(0.8,0.99)`, weight decay 0.01) and gradient clipping at 3. Per-block non-reentrant activation checkpointing saves memory. Initial SNR is uniform -5 to 20 dB with 20% clean examples. Later recipes use 40% -15 to 0 dB, 40% 0 to 20 dB, 20% clean. Two external noise recordings are mixed with relative gain -10 to 0 dB; target/noisy signals share RMS scaling (-30 to -12 dB) and peak limiting (0.95).

Balance adds 0.1 times smooth capped negative centered SI-SDR (30 dB cap) and frozen SraVaani token/duration posterior matching. Teacher KL uses temperature 2, duration weight 0.25, and up to 12 emission plus 12 blank positions along the detached clean greedy path. This is a sparse surrogate, not full sequence likelihood. The recognition gradient is projected if it conflicts with reconstruction, then capped to half the reconstruction-gradient norm. The constraint concerns raw parameter gradients, not AdamW updates or generalization.

## Historical selection versus replay

The original search compared paired arms on external noisy Combined and separate clean-preservation checks, then confirmation and full competition validation. The submitted Balance checkpoint is token step 600 from the Noise hard step 800 parent. A higher original full-validation promotion threshold was not met by Balance alone; its later fixed-context/shift/projection descendants were tested incrementally. The report gives measured sequential results without claiming independent statistical validation.

The release driver is a readable consolidation of those original update implementations. It does not download private Kaggle recovery bundles, rerun rejected candidates or make new score claims. Full GPU retraining of this consolidated driver was not repeated during final packaging.
