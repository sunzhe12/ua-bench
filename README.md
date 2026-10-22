<div align="center">

<p><img src="assets/readme-seu.svg" height="38" align="middle" alt="Southeast University"></p>

<h1>UA-Bench: Benchmarking Large Audio-Language Models<br>on Underwater Acoustic Recognition</h1>

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2610.16942)
[![Project page](https://img.shields.io/badge/Project-Page-0e7490.svg)](https://sunzhe12.github.io/ua-bench/)
[![Eval Data](https://img.shields.io/badge/Eval%20Data-2%2C600-0891b2.svg)](https://huggingface.co/datasets/zhesun12/water)
[![Full Data](https://img.shields.io/badge/Full%20Data-375%2C051-155e75.svg)](https://huggingface.co/datasets/zhesun12/water_all)
[![Tests](https://img.shields.io/badge/Tests-41%20passing-0f766e.svg)](#tests)
[![License](https://img.shields.io/badge/License-MIT-2da44e.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/sunzhe12/ua-bench?style=social)](https://github.com/sunzhe12/ua-bench/stargazers)

</div>

<p align="center">
  <img src="docs/static/images/teaser.jpg" width="100%" alt="UA-Bench teaser: the three signal families (sonar interpretation, underwater communications, vessel monitoring, AUV acoustic awareness) feed a hierarchical recognition benchmark of 375,051 records, 13 leaf classes and 2,600 evaluation examples, with dataset composition, label space, a concrete 2FSK example and the three diagnostic questions.">
</p>

<p align="center">
  <strong>To our knowledge, the first large-scale benchmark of large audio-language models (LALMs) on underwater acoustic signal recognition.</strong><br>
  375,051 synthetic records · a three-level hierarchy with 13 leaf classes · a gated coarse-to-fine protocol<br>
  <a href="https://sunzhe12.github.io/ua-bench/">Explore the project page →</a>
</p>

---

UA-Bench pairs parameterized signal sources with modeled multipath propagation, so that every
record can be traced to the conditions that produced it. This repository contains the
**evaluation framework**: it reads the released dataset, asks the three fixed questions embedded
in each record, then parses and scores the answers.

| | |
|:--|:--|
| Records | 375,051 synthetic audio records with full source and channel metadata |
| Evaluation subset | 2,600 balanced records (13 classes × 200) |
| Hierarchy | 3 levels: source activity → signal family → 13 leaf classes |
| Models | 9 open LALMs evaluated without task-specific adaptation |
| Headline result | 2.0–13.1% leaf accuracy, against a random reference of 7.7% |

## Taxonomy

| Level | Values |
|---|---|
| L1 | `active` / `passive` |
| L2 | `pulse` / `communication` / `ship_noise` |
| L3 | CW, LFM, HFM, 2FSK, 4FSK, BPSK, QPSK, OFDM, cargo, cruise, fishing, warship, underwater_target |

## Repository layout

```
README.md
testsite/
  core/       loading, answer parsing, scoring, model backends, statistics
  eval/       three-turn orchestration
  config/     taxonomy and default run settings
  scripts/    run, merge, silence generation, paired comparison, dataset download
  reporting/  Markdown report generation
  tests/      regression checks
```

## Dataset

The datasets are published on Hugging Face and are **not stored in this repository**
(the full archive is far above GitHub's 100 MB file limit):

- [`zhesun12/water`](https://huggingface.co/datasets/zhesun12/water) — the evaluation subset, `testset_export_v2.tar.gz`, 2,600 records, 13 classes × 200, about 2.2 GB.
- [`zhesun12/water_all`](https://huggingface.co/datasets/zhesun12/water_all) — the full dataset, 375,051 records (train / val / test splits plus metadata), for follow-up adaptation work.

```bash
bash testsite/scripts/download_water_testset.sh                # extracts into ./testset_export
bash testsite/scripts/download_water_testset.sh /data/ua        # or choose a target directory
```

The script needs the `hf` CLI (`pip install huggingface_hub`). It verifies the archive size and SHA-256 and refuses entries containing absolute paths or `..`. `TESTSET_DIR`, `DATASET_REVISION`, `EXPECTED_SIZE` and `EXPECTED_SHA256` override the defaults; if the dataset on Hugging Face is replaced, update `revision` and the two checksums in the script. The result looks like this:

```
testset_export/
  sft_test_highquality.jsonl   # manifest
  audio/...                    # processed WAV (16 kHz)
```

## Installation

Python 3.10 or newer.

```bash
python -m pip install -r testsite/requirements.txt   # PyYAML / NumPy / SoundFile
```

PyTorch, Transformers and whatever each backend needs are installed according to that model's own documentation; this repository does not pin a GPU build.

## Running an evaluation

Run from the **directory that contains `testsite/`**.

The `audio` field in the manifest is a relative path, resolved as `<audio-root> / <audio field>`. A single missing audio file aborts the run.

Verify the environment with the mock backend first:

```bash
python -m testsite.scripts.run_eval --mock 26 --backend mock --output-dir results/mock_check
```

Single GPU (or a single process):

```bash
python -m testsite.scripts.run_eval \
  --data testset_export/sft_test_highquality.jsonl \
  --audio-root testset_export \
  --backend qwen25_omni \
  --model-id Qwen/Qwen2.5-Omni-7B \
  --output-dir results/run_001
```

`--backend` also accepts `qwen2_audio`, `aero1_audio`, `voxtral_mini`, `voxtral_small`, `af_next`, `kimi_audio`, `minicpmo`, `gemma4` and `midashenglm`. Only `voxtral_*`, `qwen25_omni`, `af_next`, `gemma4` and `midashenglm` support true batching; the rest fall back to per-sample generation. Run `python -m testsite.scripts.run_eval --help` for the full option list.

### Sharding across GPUs

Every shard of one run must be given the **same `--run-id`**. The manifest is split deterministically by line number, so shards do not overlap:

```bash
RUN_ID=$(python -c 'import uuid; print(uuid.uuid4().hex)')
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python -m testsite.scripts.run_eval \
    --data testset_export/sft_test_highquality.jsonl \
    --audio-root testset_export \
    --backend qwen25_omni --model-id Qwen/Qwen2.5-Omni-7B \
    --num-shards 4 --shard-index $i --run-id "$RUN_ID" \
    --output-dir results/run_001 &
done
wait
```

### Merging

Concatenate the shard predictions into one file first, then merge; `--protocol` must list **all** protocol shards of that run:

```bash
cat results/run_001/predictions_shard_*.jsonl > results/run_001/predictions.jsonl

python -m testsite.scripts.merge_predictions \
  --input results/run_001/predictions.jsonl \
  --output results/run_001/metrics.json \
  --manifest testset_export/sft_test_highquality.jsonl \
  --protocol results/run_001/protocol_shard_*.json
```

Always point each run at a fresh output directory so that fixed-name prediction and protocol files are not overwritten.

## Output files

Each main run writes:

| File | Contents |
|---|---|
| `predictions_shard_NNN.jsonl` | Per sample: the questions actually sent, raw outputs, parse results, gating state |
| `protocol_shard_NNN.json` | Run signature, manifest hash, implementation hashes, taxonomy used |
| `eval_metrics_*.json` | L1/L2/L3 accuracy, L3 macro F1, L2\|L1, L3\|L2, per-class P/R/F1, confusion matrix |
| `eval_report_*.md` | The metrics above plus the five Turn 3 lexical diagnostics |

## Silence control (optional supplementary experiment)

Pairs the original audio with equal-duration, equal-rate silence under the same protocol and procedure.

```bash
# 1. Build the control; creates a fresh timestamped directory
python -m testsite.scripts.generate_silent_control \
  --data testset_export/sft_test_highquality.jsonl \
  --audio-root testset_export \
  --output-prefix silent_control

# 2. Run each condition once, same model and decoding settings
python -m testsite.scripts.run_eval --data CONTROL_DIR/original.jsonl --audio-root CONTROL_DIR \
  --backend qwen25_omni --model-id Qwen/Qwen2.5-Omni-7B --output-dir results/original
python -m testsite.scripts.run_eval --data CONTROL_DIR/silent.jsonl --audio-root CONTROL_DIR \
  --backend qwen25_omni --model-id Qwen/Qwen2.5-Omni-7B --output-dir results/silent

# 3. Verify the pairing and export the differences
python -m testsite.scripts.compare_silent_control \
  --control-dir CONTROL_DIR \
  --original-predictions results/original/predictions.jsonl \
  --silent-predictions results/silent/predictions.jsonl \
  --original-protocol results/original/protocol_shard_*.json \
  --silent-protocol results/silent/protocol_shard_*.json \
  --output-prefix results/comparison
```

Replace `CONTROL_DIR` with the directory printed by step 1. Differences are defined as **original minus silent** and are not assumed to be positive.

## Tests

```bash
python -m unittest discover -s testsite/tests
```

The suite runs 41 regression checks and is the executable specification of the parsing, gating, scoring and integrity rules enforced by the framework.

<details>
<summary>Citation</summary>

```bibtex
@misc{sun2026uabench,
  title         = {UA-Bench: Benchmarking Large Audio-Language Models on Underwater Acoustic Recognition},
  author        = {Sun, Zhe and An, Liang and Yao, Jiayu and Qian, Ronglai and Cai, Jianing
                   and Huang, Ziwei and Fan, Qing and Yang, Zhiyuan and Cai, Yujun and Cao, Hongli
                   and Wang, Yiwei and Wang, Xiaoyan},
  year          = {2026},
  eprint        = {2610.16942},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  note          = {Under review},
  doi           = {10.48550/arXiv.2610.16942},
  url           = {https://sunzhe12.github.io/ua-bench/}
}
```

</details>

## License

MIT — see [LICENSE](LICENSE).

## Notes

- The test set is a **diagnostic subset collected under favourable conditions** (the top 200 per class by channel gain and source SNR). It does not represent the full range of sea states or operational conditions.
- The exact answer format, cascade gating, metric definitions and integrity checks are implemented in `testsite/core/` and covered by `testsite/tests/`. Merging refuses a partial set of samples: a full-manifest evaluation cannot be substituted by a subset.
- The training partition of the full dataset (`zhesun12/water_all`) is released for follow-up adaptation work.
