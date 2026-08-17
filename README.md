# Scoped Reproducibility Claims

## Install

```sh
python -m pip install -e .
```

## Run

```sh
src-case run \
  --mode native \
  --context local \
  --output build/case-study
```

## Expected outputs

| Path | Contents |
| --- | --- |
| `build/inputs/` | Downloaded inputs verified against their registered SHA-256 digests |
| `build/case-study/predictions.csv` | Paired MNIST/QMNIST predictions for all 10,000 rows |
| `build/case-study/metrics.json` | Accuracy, agreement, and reconstruction statistics |
| `build/case-study/figure-data.json` | Structured values used by the result figure |
| `build/case-study/case-study-values.tex` | Generated LaTeX values for the manuscript |
| `build/case-study/case-study-*.png` | MNIST, difference, and QMNIST example images |
| `build/case-study/{binding,registration}.json` | Bound inputs, specification, claim, and execution context |
| `build/case-study/{artifacts,provenance}.json` | Output hashes and execution provenance |
| `build/case-study/assessment.json` | Reproduction decision and reasons |
| `build/case-study/{ro-crate-metadata,workflow}.json` | Workflow Run RO-Crate metadata |
| `build/case-study/main.pdf` | Manuscript produced by a container run with `--compile-paper` |
| `build/main.pdf` | Manuscript produced by `latexmk main.tex` |

## Test

```sh
python -m unittest discover -s tests -v
python -m compileall -q repro_case tests
```

## Build the paper

```sh
latexmk main.tex
```

## Run in the fixed container

```sh
docker build --platform linux/amd64 -t src-qmnist .
docker run --rm \
  -v "$PWD/build:/workspace/build" \
  src-qmnist run \
  --mode container \
  --context local-container \
  --output /workspace/build/case-study \
  --compile-paper
```
