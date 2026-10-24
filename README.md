# BT4222 Group 4: Fair-Exposure News Recommendation on MIND-large

Personalised news recommender that gives newly published and rarely shown articles a fair chance of reaching relevant readers, while keeping recommendations diverse. Built on the Microsoft News Dataset (MIND-large).

## Pipeline

| Step | Notebook | Owner | What it does |
|---|---|---|---|
| – | `src/eval.py` | P2 | Shared metrics: AUC, MRR, nDCG@5/10, cold/warm breakdown, diversity, cold share |
| – | `src/data_utils.py` | P1 | Shared loaders, impression expansion, negative sampling |
| 0 | `step0_eda.ipynb` | P2 | Interim figures, cold-threshold sensitivity |
| 1 | `step1_data_preprocess.ipynb` | P1 | Parquet files: date splits (9–13 / 14 / 15 Nov), 1:4 negative sampling, leak-free cold/warm labels |
| 2a | `step2a_popularity.ipynb` | P1 | Random, popularity and category-CTR baselines |
| 2b | `step2b_content_based.ipynb` | P2 | Content-based filtering (title/abstract similarity to user history) |
| 2c | `step2c_ncf_nmf.ipynb` | P3 | NCF / NMF and cold-vs-warm gap diagnosis |
| 3 | `step3_features.ipynb` | P4 | Feature tables, text and entity embeddings |
| 4a | `step4a_feature_mlp.ipynb` | P4 | Feature-based MLP, feature ablation |
| 4b | `step4b_neural_model.ipynb` | P5 | NRMS-style neural content model, architecture ablation |
| 5 | `step5_rerank_mmr.ipynb` | P6 | MMR re-ranking and λ sweep (relevance vs diversity) |
| 6 | `step6_diagnostics.ipynb` | P2 | Segment analysis: cold/warm, category, slate length, history length |

Shared settings live in `config.py` (paths, split dates, thresholds). Import shared code; don't copy it into notebooks.

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
3. Run all cells top to bottom, in step order. Step 4b needs a GPU runtime (*Runtime → Change runtime type*).

## Team rules

- One owner per notebook; only save your own. Changes to `src/` or `config.py` go through their owner.
- Read data only from `data/processed/`; never re-split or re-sample inside your notebook.
- Save to GitHub via *File → Save a copy in GitHub* (same file name, short commit message).
- Log useful AI interactions in the shared AI log sheet as you go.

## References

- Wu, F. et al. (2020). MIND: A Large-scale Dataset for News Recommendation. ACL.
- Wu, C. et al. (2019). Neural News Recommendation with Multi-Head Self-Attention. EMNLP.
- Past project baseline: https://github.com/bt422210/BT4222-Group-10
