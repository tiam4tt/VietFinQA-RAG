# VietFinQA

A pipeline for collecting and processing Vietnamese financial news into QA pairs.

## Overview

VietFinQA contains 111,782 QA pairs from 107,200 Vietnamese financial news articles (2015–2025). This repository provides the data collection and processing pipeline (scraping → segmentation → QA generation).

## Quick Start

```bash
git clone https://github.com/tiam4tt/vietfinqa.git
cd vietfinqa
pip install -r requirements.txt
```

See [NOTEBOOKS.md](NOTEBOOKS.md) for pipeline execution guide.

## Pipeline Stages

```
Raw Articles (scraped)
    ↓
Merge & Clean (all_articles.csv)
    ↓
Segmentation (segmented_data.parquet)
    ↓
Classification (economic/non-economic filter)
    ↓
[Path A: Segmentation]        [Path B: Summarization]
        ↓                                ↓
   Segment + Filter            Article Summarization
        ↓                                ↓
        └─────── QA Generation ────────┘
                     ↓
              QA Evaluation
                     ↓
              QA Pairs (Parquet)
```

### Data Schema

**Output CSV columns:**
- `question` (str): Generated question
- `context` (str): Source passage or summary
- `answer` (str): Generated answer
- `question_type` (str): FACTOID | SUMMARY | COMPARISON | VERIFICATION
- `source_url` (str): Source article URL

## API Keys

QA generation and classification notebooks require `GEMINI_API_KEY`; QA evaluation notebooks require `OPENAI_API_KEY`:

```bash
export GEMINI_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

## Dataset

The final **111,782 QA pairs** are available on **Kaggle**:

- **Segmentation subset**: 59,268 pairs from 15,630 articles
- **Summarization subset**: 52,514 pairs from 17,692 articles

**Download**: [VietFinQA on Kaggle](https://www.kaggle.com/datasets/tiam4tt/vietfinqa)

## Structure

```
notebooks/                     # 14 pipeline notebooks
  0_data_collecting/
    0_SCRAPE-baodautu.ipynb              # Web scraping from baodautu.vn + vneconomy.vn
    1_EDA-baodautu.ipynb                 # EDA on scraped baodautu data
    2_MERGE-baodautu-vneconomy.ipynb     # Merge and filter both sources
  3_SEG-segmentation.ipynb               # Greedy z-score article segmentation
  4_SEG-EDA.ipynb                        # Segmentation statistics and cleaning
  5_CLS-gemini.ipynb                     # Economic classification via Gemini
  6_CLS-classification.ipynb             # Classification EDA and filtering
  7_QA-Q_gen.ipynb                       # Question generation (shared by both paths)
  8_QA-A_gen.ipynb                       # Answer generation (shared by both paths)
  9_SUM-summarization.ipynb              # Article summarization via Gemini
  10_QA-EVAL_gpt.ipynb                   # QA evaluation via GPT-4.1 Mini (segmentation path)
  11_QA-EVAL_gpt_summarization.ipynb     # QA evaluation via GPT-4.1 Mini (summarization path)
  12_QA-EDA_segmentation.ipynb           # EDA on segmentation path QA
  13_QA-EDA_summarization.ipynb          # EDA on summarization path QA
utils/                           # Utilities (dataset loader)
requirements.txt
```
