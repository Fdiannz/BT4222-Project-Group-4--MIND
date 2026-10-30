# BT4222 Group 4: Fair-Exposure News Recommendation on MIND-large

Personalised news recommender that gives newly published and rarely shown articles a fair chance of reaching relevant readers, while keeping recommendations diverse. Built on the Microsoft News Dataset (MIND-large).

## Pipeline

| Step | Notebook | Owner | Output it hands on |
|---|---|---|---|
| – | `utils/metrics.py` (shared) | P2 | Metric functions: AUC, MRR, nDCG@5/10, cold/warm breakdown, diversity, cold share |
| – | `utils/data_utils.py` (shared) | P1 | Loaders, impression expansion, negative sampling |
| 0 | `step0_eda.ipynb` | P2 | Interim figures, cold-threshold sensitivity |
| 1 | `step1_data_preprocess.ipynb` | P1 | Parquet files: split by date (9–13 / 14 / 15 Nov), 1:4 negative sampling, leak-free cold/warm labels |
| 2 | `step2_baselines/` | P3 | Random, popularity, NCF/NMF; cold-versus-warm gap diagnosis |
| | ├ `step2_popularity.ipynb` | P1 | Popularity and category-CTR baseline |
| | ├ `step2_content_based.ipynb` | P3 | Content-based filtering (title/abstract similarity to user history) |
| | └ `step2_ncf_nmf.ipynb` | P3 | NCF / NMF (course baselines) |
| 3 | `step3_features/` | P4 | Feature tables and text/entity embeddings, with feature ablation |
| | ├ `step3_features.ipynb` | P4 | Feature tables and embeddings |
| | └ `step3_feature_mlp.ipynb` | P4 | Feature-based MLP |
| 4 | `step4_advanced_model.ipynb` | P5 | Content-based neural model (NRMS-style), plus architecture ablation |
| 5 | `step5_rerank_mmr.ipynb` | P6 | MMR re-ranking and the λ sweep |
| 6 | `step6_diagnostics.ipynb` | P2 | Segment analysis by cold/warm, category, slate length and history length |

Shared settings live in `config.py` (paths, split dates, thresholds). Import shared code from `utils/`; don't copy it into notebooks.

## Data splits

| Split | Source | Dates | Use |
|---|---|---|---|
| Fit | MIND-large train | 9–13 Nov 2019 | Model training (1:4 within-impression negative sampling) |
| Tune | MIND-large train | 14 Nov 2019 | Hyperparameter tuning, model selection (full slates) |
| Test | MIND-large dev | 15 Nov 2019 | Final evaluation (full slates) |

The official MIND test split has no public click labels and is not used for evaluation.

## How to run (Google Colab)

1. **Data — do this first:** MIND-large is in the shared Google Drive folder `BT4222_Project/data/raw/` ([Drive link]). Add it to your Drive: open the link → click the folder name → *Organize → Add shortcut* → *My Drive*. It must sit directly under My Drive. To use your own copy instead, download MIND-large from https://msnews.github.io and edit the paths in `config.py`.
2. Open a notebook in Colab: *File → Open notebook → GitHub* → paste this repo's URL.
3. Run all cells top to bottom, in step order. Step 4 needs a GPU runtime (*Runtime → Change runtime type*).

## Team rules

- One owner per notebook; only save your own. Changes to `utils/` or `config.py` go through their owner.
- Read data only from `data/processed/`; never re-split or re-sample inside your notebook.
- Save to GitHub via *File → Save a copy in GitHub* (same file path, short commit message).
- Log useful AI interactions in the shared AI log sheet as you go.

## References

- Wu, F. et al. (2020). MIND: A Large-scale Dataset for News Recommendation. ACL.
- Wu, C. et al. (2019). Neural News Recommendation with Multi-Head Self-Attention. EMNLP.
- Past project baseline: https://github.com/bt422210/BT4222-Group-10
