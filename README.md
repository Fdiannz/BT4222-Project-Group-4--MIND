# BT4222 Group 4: Fair-Exposure News Recommendation on MIND-large

A personalised news recommender that gives **newly published and rarely shown articles a fair chance of reaching relevant readers**, while keeping recommendations diverse. Built on the Microsoft News Dataset (MIND-large).

## 📌 Project Overview

News platforms publish new articles all day, and most of a reader's options are stories that have barely been shown to anyone yet. Recommenders that learn from past clicks favour articles that have already been clicked many times, so new articles (**cold** articles) are pushed down even when they match a reader's interests. Over time this narrows what readers see and hides fresh content.

On our test day, **17.9% of displayed articles and 23.5% of clicks are cold** (fewer than 10 displays on earlier days), so cold articles are a large share of what readers actually want.

This project builds and evaluates recommenders with two goals:
1. **Relevance for cold articles:** rank new articles well from their content, not only from click history.
2. **Diverse recommendations:** re-rank the final list so it is not dominated by one topic.

We use **MIND-large** (about 1 million users and 160,000 articles), which reflects the scale of a large news platform. Models are compared in increasing order of complexity:

- Popularity baseline
- Content-based filtering (CBF)
- Neural Collaborative Filtering / Non-negative Matrix Factorisation (NCF / NMF)
- Feature-based Multilayer Perceptron (MLP)
- **KLAR: Knowledge-aware LSTM–Attention Recommender** (our advanced model)
- MMR re-ranking for diversity and slate calibration to match reading history, both applied to KLAR

## 🧭 Pipeline

| Step | Notebook | Output it hands on |
|---|---|---|
| – | `utils/metrics.py` (shared) | Metric functions: AUC, MRR, nDCG@5/10, cold/warm breakdown, diversity, cold share |
| – | `utils/data_utils.py` (shared) | Loaders, impression expansion, negative sampling |
| 0 | `step0_eda.ipynb` | Exploratory figures, cold-threshold sensitivity |
| 1 | `step1_data_preprocess.ipynb` | Parquet files: split by date (9–13 / 14 / 15 Nov), 1:4 negative sampling, leak-free cold/warm labels |
| 2 | `step2_baselines/` | Popularity, content-based, NCF/NMF baselines and their cold-versus-warm gap |
| ├ | `step2_popularity.ipynb` | Popularity baseline |
| ├ | `step2_content_based.ipynb` | Content-based filtering (title/abstract similarity to user history) |
| └ | `step2_ncf_nmf.ipynb` | NCF / NMF (course baselines) |
| 3 | `step3_features/` | Feature tables, text embeddings and the feature-based MLP, with feature ablation |
| ├ | `step3_features.ipynb` | User, news, context and user × news feature tables; text embeddings |
| └ | `step3_feature_mlp.ipynb` | Feature-based MLP |
| 4 | `step4_advanced_model.ipynb` | KLAR advanced model, its score files and article vectors, plus architecture ablation |
| 5 | `step5_rerank_mmr.ipynb` | Tune sweeps and Test rankings for MMR |
| └ | `step5_rerank_calibration.ipynb` | Tune sweeps and Test rankings for slate calibration |
| 6 | `step6_diagnostics.ipynb` | Model comparison by cold/warm, category, slate length and history length |

Shared settings live in `config.py` (paths, split dates, thresholds). Import shared code from `utils/`; don't copy it into notebooks.

## 🔎 Workflow (Steps 0–6)

### Step 0: Exploratory Data Analysis
`step0_eda.ipynb` analyses user behaviour, article exposure, categories and click patterns, and tests how sensitive the cold share is to the cold threshold. These findings set the threshold (fewer than 10 earlier displays) and guide later design choices.

### Step 1: Data Preprocessing
`step1_data_preprocess.ipynb` loads MIND-large, cleans the article metadata, and splits impressions **by date** into Fit, Tune and Test (see below). Each candidate is labelled **cold or warm** using only displays on **earlier days**, so the label never uses information from the day being predicted. For training, each click is paired with 4 unclicked articles from the same impression (1:4 negative sampling). All outputs are parquet files in `data/processed/`.

### Step 2: Baselines
- `step2_popularity.ipynb`: ranks articles by how often they were clicked before. A strong baseline for warm articles but, by design, weak for cold ones.
- `step2_content_based.ipynb`: scores each article by the similarity of its title and abstract to the user's click history.
- `step2_ncf_nmf.ipynb`: collaborative filtering with NCF and NMF. These learn user and article ID embeddings, so they show the cold-versus-warm gap that ID-based models suffer from.

### Step 3: Feature Engineering and Feature-based MLP
- `step3_features.ipynb`: builds leak-free user, news, context and user × news features (e.g. history length, article age, content and category match with the user's history) and sentence embeddings of each article.
- `step3_feature_mlp.ipynb`: an MLP trained on these features, with feature ablation to show which groups matter.

### Step 4: Advanced Model (KLAR)
`step4_advanced_model.ipynb` trains **KLAR (Knowledge-aware LSTM–Attention Recommender)**:
- **News representation:** word2vec title embedding, category and subcategory embeddings, and TransE knowledge-graph entity embeddings, fused into one vector per article. No article IDs or click counts are used, so new articles are represented from the moment they are published.
- **User encoder:** the user's last 50 clicks, read in order by an LSTM, then multi-head self-attention and additive attention to form one user vector.
- **Score:** dot product of the user vector and each candidate's news representation.

It outputs score files for Tune and Test (`data/scores/step4_klar_*.parquet`) and article vectors (`data/embeddings/step4_klar_news_vectors.npy`), and runs an architecture ablation (without LSTM, without entities, fewer heads, lower dropout).

### Step 5: Re-ranking
`step5_rerank_mmr.ipynb` re-ranks KLAR's scores with **Maximal Marginal Relevance (MMR)**, trading a little relevance for diversity using the similarity between KLAR's article vectors. The trade-off λ is chosen on Tune and applied once to Test.

`step5_rerank_calibration.ipynb` re-ranks KLAR's top 10 to better match the categories and subcategories in the user's click history. The calibration weight is chosen on Tune and fixed for Test. 

Both notebooks compare their results against KLAR and can run in either order after Step 4.

### Step 6: Diagnostics
`step6_diagnostics.ipynb` compares all models on the Test day with the shared metrics, broken down by cold/warm impressions, category, slate length and the user's history length, and reports the diversity and cold share of the top-10 lists.

## 📅 Data splits

| Split | Source | Dates | Use |
|---|---|---|---|
| Fit | MIND-large train | 9–13 Nov 2019 | Model training (1:4 within-impression negative sampling) |
| Tune | MIND-large train | 14 Nov 2019 | Hyperparameter tuning, model selection (full slates) |
| Test | MIND-large dev | 15 Nov 2019 | Final evaluation (full slates) |

The official MIND test split has no public click labels and is not used for evaluation.

## ▶️ How to run

**Data:** MIND-large and all processed files are in the shared Google Drive folder [BT4222_Project](https://drive.google.com/drive/folders/1D4Db1s7DJ5PVK9CFiV-pklpCW5dr1uOM?usp=sharing). The raw data can also be downloaded from https://msnews.github.io (folders `MINDlarge_train`, `MINDlarge_dev`, `MINDlarge_test` under `data/raw/`).

**Google Colab (recommended):**
1. In Google Drive, add a shortcut to the shared `BT4222_Project` folder in **My Drive** (right-click → Organise → Add shortcut). Colab can only see shared folders through a shortcut.
2. Open a notebook in Colab and run its setup cell. It mounts Drive and clones this repository, and `config.py` finds the data at `/content/drive/MyDrive/BT4222_Project`.
3. Run the notebooks from step 0 to step 6 in order. Step 1 creates the parquet files in `data/processed/` that later steps need (they are already on the shared Drive).

Step 4 needs a GPU (*Runtime → Change runtime type → T4 GPU*); it takes about 2.5–3 hours including ablations.

**Locally:** install the dependencies with `pip install -r requirements.txt`, put the data under `data/` in this repository (or set `MIND_PROJECT_DIR` to the folder that contains `data/`), and run the notebooks in the same order. Step 4 still needs a CUDA GPU.
