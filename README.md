# T826_KL_CNTT12

Vietnamese Financial Question Answering with RAG and SCS-LoRA.

## Overview

A thesis project building a Vietnamese financial QA system using the VietFinQA dataset (111,782 QA pairs from 107,200 Vietnamese financial news articles). The project compares baseline RAG approaches with a proposed SCS-LoRA (Self-Corrected Scale LoRA) method for improved question answering.

## Structure

```
T826_KL_CNTT12/
├── src/
│   ├── 0_dataset_construction/  # Data pipeline (14 notebooks + utils)
│   ├── 1_baseline/              # Baseline RAG (Llama-3.2-1B, Qwen3-0.6B, Qwen3-1.7B)
│   ├── 2_proposed_method/       # SCS-LoRA classification, training, inference
│   ├── 3_evaluation/            # ROUGE, BERTScore, RAGAS evaluation
│   ├── 4_dataset/               # CSV datasets (train/val/test/gold_test)
│   └── requirements.txt         # Python dependencies
├── HuongDanCaiDat.txt           # Installation guide
├── HuongDanSuDung.txt           # Usage guide
└── README.md                    # This file
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r src/requirements.txt
   ```

2. **Download datasets from Kaggle:**
   - [VietFinQA on Kaggle](https://www.kaggle.com/datasets/tiam4tt/vietfinqa)
   - Place CSV files in `src/4_dataset/data-merge/`, `src/4_dataset/data-segmentation/`, and `src/4_dataset/data-summarization/`

3. **Verify installation:**
   ```bash
   python -c "import torch, transformers, peft, sentence_transformers, faiss, rank_bm25; print('OK')"
   ```

## Running Demos

### Demo 1 - Baseline RAG
Open any notebook in `src/1_baseline/` with Jupyter and run all cells.

### Demo 2 - SCS-LoRA Inference
```bash
python src/2_proposed_method/INFERENCE/qwen3_1_7b_scs_lora_test_inference.py
```

> **Note:** Update `DATA_DIR` and `ADAPTER_DIR` paths in the script for local execution.

### Demo 3 - Evaluation
Open notebooks in `src/3_evaluation/` with Jupyter.

## Requirements

- Python 3.10 or 3.11
- PyTorch 2.x + CUDA (GPU recommended)
- Jupyter Notebook >= 7.0
- NVIDIA GPU with CUDA 11.8/12.x, VRAM >= 16 GB (recommended)

## License

See project documentation for details.
