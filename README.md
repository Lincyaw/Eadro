# Eadro Algorithm Reproduction

## Quick Start

### Data Preprocess
```bash
uv run eadro-train convert-rcabench \
--rcabench-root /home/nn/workspace/Eadro/data/rcabench_dataset \
--output-root /home/nn/workspace/Eadro/rcabench_chunks
```

### Train
```bash
uv run eadro-train train --dataset ./rcabench_chunks/
```

### Inference
```bash
cd eadro_infer
```