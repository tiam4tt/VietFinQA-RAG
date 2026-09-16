# Notebook Execution Guide

This document describes all 14 notebooks in the pipeline and their execution order.

## Pipeline Overview

```
Stage 1: Scraping            0_SCRAPE-baodautu → 1_EDA-baodautu → 2_MERGE-baodautu-vneconomy
Stage 2: Segmentation+CLS    3_SEG-segmentation → 4_SEG-EDA → 5_CLS-gemini → 6_CLS-classification
Stage 3: QA Generation       [seg path] 7_QA-Q_gen + 8_QA-A_gen  /  [sum path] 9_SUM-summarization → 7_QA-Q_gen + 8_QA-A_gen
Stage 4: QA Evaluation       10_QA-EVAL_gpt (segmentation path) / 11_QA-EVAL_gpt_summarization (summarization path)
Stage 5: QA EDA              12_QA-EDA_segmentation / 13_QA-EDA_summarization
```

## Execution Order & Descriptions

### Stage 1: Web Scraping & Collection

**1. `notebooks/0_data_collecting/0_SCRAPE-baodautu.ipynb`**
- **Purpose**: Scrape Vietnamese financial news articles from baodautu.vn and vneconomy.vn using XML sitemaps
- **Input**: Sitemap URLs (automatic)
- **Output**: `data/raw/scraped/articles_batch_*.jsonl`, `data/raw/scraped/failed_urls_batch_*.jsonl`, `data/raw/scraped/urls.txt`
- **Duration**: ~2-4 hours (depends on internet speed)

**2. `notebooks/0_data_collecting/1_EDA-baodautu.ipynb`**
- **Purpose**: Exploratory data analysis and cleanup of scraped Baodautu data
- **Input**: `data/raw/scraped/articles_batch_*.jsonl`
- **Output**: Statistical analysis, visualizations

**3. `notebooks/0_data_collecting/2_MERGE-baodautu-vneconomy.ipynb`**
- **Purpose**: Compare, merge and deduplicate Baodautu + VnEconomy data
- **Input**: `data/raw/scraped/articles_batch_*.jsonl` (both sources)
- **Output**: `data/processed/all_articles.csv`
- **Duration**: ~10 minutes

### Stage 2: Segmentation & Classification

**4. `notebooks/3_SEG-segmentation.ipynb`**
- **Purpose**: Split articles into segments using greedy z-score segmentation
- **Input**: `data/processed/all_articles.csv`
- **Output**: `data/raw/segmented_data.parquet`
- **Duration**: ~30 minutes

**5. `notebooks/4_SEG-EDA.ipynb`**
- **Purpose**: Analyze segmentation structure; clean/filter segments
- **Input**: `data/raw/segmented_data.parquet`
- **Output**: `data/processed/cleaned_segmented_data.parquet`
- **Duration**: ~10 minutes

**6. `notebooks/5_CLS-gemini.ipynb`** ⚠️ Requires API key
- **Purpose**: Classify segments as economic/non-economic using Gemini 2.5 Flash batch API (57,369 segments)
- **Input**: `data/processed/cleaned_segmented_data.parquet`
- **Output**: `data/raw/classified_segmented_data.parquet`, `data/cache/batch_cls_{}.jsonl`
- **Requirements**: `GEMINI_API_KEY` environment variable **MUST** be set
- **Duration**: ~1-2 hours (batch API processing)

**7. `notebooks/6_CLS-classification.ipynb`**
- **Purpose**: Classification EDA and filtering; keep only economic segments
- **Input**: `data/raw/classified_segmented_data.parquet`
- **Output**: `data/processed/cleaned_classified_segmented_data.parquet`
- **Duration**: ~15 minutes

### Stage 3a: QA Generation — Segmentation Path

**8. `notebooks/7_QA-Q_gen.ipynb`** ⚠️ Requires API key
- **Purpose**: Generate questions from article contexts using Gemini batch API
- **Input**: `data/processed/cleaned_classified_segmented_data.parquet`
- **Output**: `data/processed/Q_data.parquet` + `data/processed/Q_data_*.csv` (incremental)
- **Duration**: ~4-8 hours (batch API processing)
- **Requirements**: `GEMINI_API_KEY` environment variable **MUST** be set
- **Cost**: ~$5-10 (Gemini API usage)
- **Checkpointing**: Automatically resumes from last checkpoint; safe to interrupt

**9. `notebooks/8_QA-A_gen.ipynb`** ⚠️ Requires API key
- **Purpose**: Generate answers for questions using Gemini API
- **Input**: Questions + contexts from QA-Q_gen
- **Output**: `data/processed/QA_data_merged.parquet`, `data/processed/QA_data_p2.parquet`
- **Duration**: ~4-8 hours
- **Requirements**: `GEMINI_API_KEY` environment variable **MUST** be set
- **Cost**: ~$5-10
- **Checkpointing**: Supports resuming from checkpoints

### Stage 3b: QA Generation — Summarization Path

**10. `notebooks/9_SUM-summarization.ipynb`** ⚠️ Requires API key
- **Purpose**: Summarize classified articles via Gemini before QA generation
- **Input**: `data/processed/cleaned_classified_segmented_data.parquet`
- **Output**: `data/processed/summarized_content_data.csv`
- **Requirements**: `GEMINI_API_KEY` environment variable **MUST** be set

Then re-run **7-8 (`7_QA-Q_gen.ipynb` + `8_QA-A_gen.ipynb`)** with the summarized content as context.

### Stage 4: QA Evaluation

**11. `notebooks/10_QA-EVAL_gpt.ipynb`** ⚠️ Requires API key (segmentation path)
- **Purpose**: Evaluate QA pairs from the segmentation path with GPT-4.1 Mini on a 20,000-article sample
- **Input**: `data/processed/QA_data.parquet` (sampled)
- **Output**: `data/processed/QA_data_evaluated.parquet`, `data/cache/batch_json/`, `data/cache/eval-cache/`
- **Result**: 62,276 QA pairs; acceptance rate **98.3%** (61,229 ACCEPT), mean score 9.97/10

**12. `notebooks/11_QA-EVAL_gpt_summarization.ipynb`** ⚠️ Requires API key (summarization path)
- **Purpose**: Evaluate QA pairs from the summarization path with GPT-4.1 Mini — full data, no sampling
- **Input**: `data/processed/Thuan_QA_data.parquet`
- **Output**: `data/processed/Thuan_QA_data_evaluated.parquet`, `data/cache/batch_json/`, `data/cache/eval-cache/`
- **Result**: 53,999 QA pairs from 18,003 articles; **98.6% ACCEPT**, ~1% REVISE (revised pairs are retained, so the effective acceptance rate is ≈99.6%), mean score 9.98/10
- **Checkpointing**: incremental batch caching; `RESUME_FROM_LAST_CHECKPOINT = False` by default

### Stage 5: QA Dataset Analysis

**13. `notebooks/12_QA-EDA_segmentation.ipynb`**
- **Purpose**: Explore QA dataset generated via segmentation path
- **Input**: Evaluated QA data (segmentation path)
- **Output**: QA statistics, question type distribution

**14. `notebooks/13_QA-EDA_summarization.ipynb`**
- **Purpose**: Explore QA dataset generated via summarization path
- **Input**: Evaluated QA data (summarization path)
- **Output**: QA statistics, question type distribution

## Key Points

### API Key Setup

Before running notebooks marked "Requires API key", set the keys:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"   # only for the QA-EVAL_gpt notebooks
```

Then verify:
```python
import os
print(os.getenv("GEMINI_API_KEY"))  # Should print your key
```

### Output Files

| Stage | Output File | Format | Size |
|-------|-------------|--------|------|
| Scraping | `data/raw/scraped/*.jsonl` | JSONL | ~2 GB total |
| Merge | `data/processed/all_articles.csv` | CSV | ~500 MB |
| Segmentation | `data/raw/segmented_data.parquet` | Parquet | ~2 GB |
| Classification | `data/raw/classified_segmented_data.parquet` | Parquet | ~2 GB |
| Q Generation | `data/processed/Q_data.parquet` | Parquet | ~3 GB |
| QA Pairs | `data/processed/QA_data_merged.parquet` | Parquet | ~5 GB |

### Storage Requirements

- **Minimum**: 2 GB (for raw data)
- **Full pipeline**: 15-20 GB (all intermediate parquet files)

### Parallelization

- Stages 1-2 must run sequentially (data dependencies)
- Stage 3a and 3b can run in parallel (independent paths)
- QA-Q_gen must run before QA-A_gen within each path
- Stage 4 evaluation notebooks can run in parallel
- Stage 5 EDA notebooks can run in parallel

### Troubleshooting

**API Key not found**: Ensure `GEMINI_API_KEY` is set:
```bash
export GEMINI_API_KEY="..."
python -c "import os; print(os.getenv('GEMINI_API_KEY'))"
```

**Out of memory**: Notebooks use batch processing; reduce batch size in notebook config.

**Network timeout**: Q/A generation is resilient to interruptions; simply re-run the notebook to resume.

---

**Total Estimated Time**: 12-24 hours (excluding manual validation)
