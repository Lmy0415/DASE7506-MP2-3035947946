# DASE7506 MP2 submission

Student ID: 3035947946. Submitted method: frozen v11 multi-expert answer selector. The official scorer reports **390 / 467 = 83.5117773019272%** on the supplied test bank. Official validation is 758 / 798; on its 355 answer-disagreement questions the score is 315 / 355.

The report is [report/main.pdf](report/main.pdf), with editable source in [report/main.tex](report/main.tex). The matching inference checkpoint is [solution_v11/runs/final_bundle](solution_v11/runs/final_bundle), and the prediction entry point is [solution_v11/code/final_submission.py](solution_v11/code/final_submission.py). No retraining, model download or API call is required for inference once the supplied benchmark data is present.

## Install the supplied benchmark

Use Python 3.12. The benchmark is provided separately so this repository does not duplicate its 105 MiB data release. From the repository root:

```sh
git clone https://github.com/xudongwu-0/DASE7506-MP2.git DASE7506-MP2-main
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-inference.txt
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python DASE7506-MP2-main/code/check_data.py --data-dir DASE7506-MP2-main/code/data
python verify_freeze.py
```

The checked benchmark protocol is `dase7506-mp2-open-hard-verifier-v5`; the hashes of the expected files are in [benchmark/code/data/manifest.json](benchmark/code/data/manifest.json). Follow the benchmark license and notice included there. The course distribution supplies the evaluation code and data; it supplies no generation-model weights.

## Reproduce the frozen result

Run the official prediction and scoring tools as separate processes from the repository root. Both commands use the supplied data, and prediction does not read reference answers.

```sh
mkdir -p reproduced
python DASE7506-MP2-main/code/predict.py \
  --method solution_v11/code/final_submission.py \
  --checkpoint solution_v11/runs/final_bundle \
  --data-dir DASE7506-MP2-main/code/data \
  --split test --output reproduced/test-predictions.json
python DASE7506-MP2-main/code/score_predictions.py \
  --data-dir DASE7506-MP2-main/code/data --split test \
  --predictions reproduced/test-predictions.json \
  --output reproduced/test-score.json
```

Use `--split validation` and matching output names to reproduce validation. Evidence from the frozen run is in [evidence/frozen](evidence/frozen), including the independent scorer output, original source and asset hashes, and process timings. A separate copy of the official tools is under `benchmark/code`; the original repository is recommended for full data verification.

The checkpoint contains 14,184,418 bytes (13.53 MiB). The original test prediction plus independent scoring took 7.16 seconds on local Apple Silicon with four numerical-library threads. A clean-environment repeat took 7.84 seconds; an earlier concurrent run reported 28.96 seconds for prediction alone. The reference Xeon runtime has not been measured and must be checked for eligibility. The scorer tests final numeric answers, not the validity of each reasoning step.

## Rebuild the last training stage

The repository contains the previously generated training out-of-fold scores and intermediate model artifacts required by `prepare_final.py`. With the data installed as above, install the development dependencies and run this in a disposable copy of the repository, since the command rewrites the checkpoint:

```sh
python -m pip install -r requirements-development.txt
python solution_v11/code/prepare_final.py
```

On the original Python 3.12.2 environment, the rebuilt 24 checkpoint files all matched the frozen SHA-256 hashes. This verifies the final selector, linear expert, and bundle assembly **from the included intermediate artifacts**. It is not a verified full raw-data retraining of all earlier experts. Source code for the earlier stages is included, but some original training caches and a complete end-to-end orchestration are not included; recreating every earlier artifact from scratch remains unverified.

The training folds are grouped by question. The selector uses seeds 201, 202 and 203 with five folds each; the centred linear expert uses seed 20261101. Earlier independent ranker training source and assets are in `kimi_mp2/DASE7506-MP2-main/`. The frozen selector uses threshold 0.6. Development predictions and hyperparameter selection were not fully nested, and multiple prior frozen aggregate test scores were observed during development. No individual test error is used in training, lookup or inference.

The independent expert was originally trained with NumPy 2.5.3, in a separate environment from final inference. In a disposable copy, the following reconstructs its training-only feature cache and runs the recorded ten-member cross-validation recipe:

```sh
python3.12 -m venv .training-independent
.training-independent/bin/python -m pip install -r requirements-independent-training.txt
ln -s "$(pwd)/DASE7506-MP2-main/code/data" kimi_mp2/DASE7506-MP2-main/code/data
.training-independent/bin/python kimi_mp2/DASE7506-MP2-main/code/restore_training_feature_cache.py
.training-independent/bin/python kimi_mp2/DASE7506-MP2-main/code/train_cv.py \
  --name cv_math_cosine50_r2 --folds 5 --repeats 2 --hidden 256 \
  --epochs 50 --feature-mode base --cosine
```

The cache reconstruction stores only two original cosine-feature columns for the supplied training and validation inputs. It recomputes all remaining features and checks the SHA-256 fingerprint of the complete resulting arrays before allowing training to proceed. These files are training artifacts; they are not read during inference and contain no reference answers or test features. Their use addresses small environment-dependent rounding differences observed in those columns. The original independent-expert training record reports 63.27 seconds on its local environment; later training durations can differ by machine and workload.

For the paired validation ablation, after installing development dependencies, run:

```sh
python solution_v11/code/ablate_experts.py \
  --data-dir DASE7506-MP2-main/code/data \
  --output reproduced/leave_one_expert_out.json
```

This exploratory ablation refits the selector using the included OOF expert scores; it is not a completely nested generalization estimate. The original selector ablation and reproducibility checks are in `evidence/`.

## Data and attribution

GSM8K is distributed under its supplied MIT license. Candidate traces and fixed embeddings were prepared by the course staff using Qwen3-0.6B; those model weights are not included or needed. The independently developed candidate ranker is reused as one of the experts. OpenAI Codex and Kimi assisted with implementation, experiment orchestration, and drafting. The student should review this disclosure and the report before final submission.
