# BT4222-Project-Group-4--MIND Fair-Exposure News Recommendation on MIND-large

Personalised news recommender that gives newly published and rarely shown articles a fair chance of reaching relevant readers, while keeping recommendations diverse. Built on the Microsoft News Dataset (MIND-large).

## Pipeline

| Step | Notebook | Owner | What it does |
|---|---|---|---|
| 1 | `step1_data_preprocess.ipynb` | P1 | Load data, fit/tune/test splits, negative sampling, shared parquet files |
| 2 | `step2_baselines.ipynb` | P3 | Random, popularity, NCF/NMF and past-group baselines; gap diagnosis |
| 3 | `step3_features.ipynb` | P4 | User, article, context and embedding features |
| 4 | `step4_advanced_model.ipynb` | P5 | Content-based model for cold-start articles |
| 5 | `step5_rerank_diagnostics.ipynb` | P6 | Diversity re-ranking, cold/warm and segment diagnostics, ablations |

Shared code (do not copy into notebooks; import it):
- `config.py`: paths, split dates, thresholds
- `src/data_utils.py` (P1): loaders, impression expansion, negative sampling
- `src/eval.py` (P2): AUC, MRR, nDCG@5/10, cold/warm breakdown, diversity metrics

## Data splits

| Split | Source | Dates | Use |
|---|---|---|---|
| Fit | MIND-large train | 9–13 Nov 2019 | Model training (1:4 within-impression negative sampling) |
| Tune | MIND-large train | 14 Nov 2019 | Hyperparameter tuning, model selection (full slates) |
| Test | MIND-large dev | 15 Nov 2019 | Final evaluation (full slates) |

The official MIND test split has no public click labels and is not used for evaluation.

## How to run (Google Colab)

1. **Data:** MIND-large is stored in the shared Google Drive folder `BT4222_Project/data/raw/` . Add it to your Drive: *Shared with me* → right-click `BT4222_Project` → *Organize → Add shortcut to My Drive*. To use your own copy, download from https://msnews.github.io and edit the paths in `config.py`.
2. Open a notebook in Colab: *File → Open notebook → GitHub* → paste this repo's URL.
3. Run all cells top to bottom, in step order (1 → 5). Step 4 needs a GPU runtime (*Runtime → Change runtime type*).
