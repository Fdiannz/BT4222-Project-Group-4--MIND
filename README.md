# BT4222 Group 4: Fair-Exposure News Recommendation on MIND-large

A personalised news recommender that gives **newly published and rarely shown articles a fair chance of reaching relevant readers**, while keeping recommendations diverse. Built on the Microsoft News Dataset (MIND-large).

## 📌 Project Overview

News platforms publish new articles all day, and most of a reader's options are stories that have barely been shown to anyone yet. Recommenders that learn from past clicks favour articles that have already been clicked many times, so new articles (**cold** articles) are pushed down even when they match a reader's interests. Over time this narrows what readers see and hides fresh content.

On our test day, **17.9% of displayed articles and 23.5% of clicks are cold** (fewer than 10 displays on earlier days), so cold articles are a large share of what readers actually want.

This project builds and evaluates recommenders with two goals:
1. **Relevance for cold articles:** rank new articles well from their content, not only from click history.
2. **Diverse recommendations:** re-rank the final list so it is not dominated by one topic.

We use **MIND-large** (about 1 million users and 160,000 articles), which reflects the scale of a large news platform. Models are compared in increasing order of complexity:

- Popularity, Content-based filtering (CBF), NCF (GMF and NeuMF) baselines
- Feature-based Multilayer Perceptron (MLP)
- **KLAR: Knowledge-aware LSTM–Attention Recommender** (our advanced model)
- MMR re-ranking for diversity and slate calibration to match reading history, both applied to KLAR

## 📅 Data splits

| Split | Source | Dates | Use |
|---|---|---|---|
| Fit | MIND-large train | 9–13 Nov 2019 | Training (1 click : 4 non-clicks from the same impression) |
| Tune | MIND-large train | 14 Nov 2019 | Hyperparameters, model selection and ablations (full slates) |
| Test | MIND-large dev | 15 Nov 2019 | Final evaluation (full slates) |

MIND's own test split has no click labels, so it is not used.

## 🧭 Pipeline

| Step | Notebook | Output it hands on |
|---|---|---|
| – | `config.py` | Shared paths, split dates, seed, cold threshold, top-K |
| – | `utils/metrics.py` (shared) | Metric functions: AUC, MRR, nDCG@5/10, cold/warm breakdown, diversity, cold share |
| – | `utils/data_utils.py` (shared) | Loaders, impression expansion, negative sampling |
| 0 | `step0_eda.ipynb` | Exploratory figures, cold-threshold sensitivity |
| 1 | `step1_data_preprocess.ipynb` | Parquet files: split by date (9–13 / 14 / 15 Nov), 1:4 negative sampling, leak-free cold/warm labels |
| 2 | `step2_baselines/` | Popularity, content-based, NCF/NMF baselines and their cold-versus-warm gap |
| ├ | `step2_popularity.ipynb` | Popularity baseline |
| ├ | `step2_content_based.ipynb` | Content-based filtering (TF-IDF + SVD similarity to the user's history) |
| └ | `step2_ncf_nmf.ipynb` | NCF baselines, GMF and NeuMF, on user and article IDs, with the cold-versus-warm gap|
| 3 | `step3_features.ipynb` | Feature Engineering : User, news, context and user × news features, and MiniLM article embeddings |
| 3 | `step3_feature_mlp.ipynb` | Feature-based MLP with feature ablation |
| 4 | `step4_advanced_model.ipynb` | KLAR advanced model, its score files and article vectors, plus architecture ablation |
| 5 | `step5_rerank_mmr.ipynb` | MMR re-ranking of KLAR for topic variety |
| 5 | `step5_rerank_calibration.ipynb` | Re-ranks KLAR so its category mix matches the user's history |
| 6 | `step6_diagnostics.ipynb` | Compares all models on Test by cold/warm clicks, cold exposure, category, slate length and history length |

Every model setting, including λ for MMR and α for calibration, is chosen on Tune and applied once to Test.

## 🔎 Workflow (Steps 0–6)

### Step 0: Exploratory Data Analysis
`step0_eda.ipynb` analyses user behaviour, article exposure, categories and click patterns, and tests how sensitive the cold share is to the cold threshold. These findings set the threshold (fewer than 10 earlier displays) and guide later design choices.

### Step 1: Data Preprocessing
`step1_data_preprocess.ipynb` loads MIND-large, cleans the article metadata, and splits impressions **by date** into Fit, Tune and Test (see below). Each candidate is labelled **cold or warm** using only displays on **earlier days**, so the label never uses information from the day being predicted. For training, each click is paired with 4 unclicked articles from the same impression (1:4 negative sampling). All outputs are parquet files in `data/processed/`.

### Step 2: Baselines
- `step2_popularity.ipynb`: ranks articles by past click rate, smoothed towards the global rate (α chosen on Tune). Articles never shown before get the global rate.
- `step2_content_based.ipynb`:turns each article's title and abstract into a TF-IDF vector reduced to 128 dimensions with SVD, then scores candidates by cosine similarity to the average of the user's clicked articles.
- `step2_ncf_nmf.ipynb`: collaborative filtering with NCF (GMF and NeuMF). These learn user and article ID embeddings, so they show the cold-versus-warm gap that ID-based models suffer from.

### Step 3: Feature Engineering and Feature-based MLP
- `step3_features.ipynb`: builds leak-free features for each candidate (history length, category and subcategory, exposure percentile within the slate, article age, MiniLM content similarity and category match with the user's history) and MiniLM embeddings for every article.
- `step3_feature_mlp.ipynb`: an MLP trained on these features, with feature ablation to show which groups matter.

### Step 4: Advanced Model (KLAR)
`step4_advanced_model.ipynb` trains **KLAR (Knowledge-aware LSTM–Attention Recommender)**:
- **News representation:** word2vec title embedding, category and subcategory embeddings, and TransE knowledge-graph entity embeddings, fused into one vector per article. No article IDs or click counts are used, so new articles are represented from the moment they are published.
- **User encoder:** the user's last 50 clicks, read in order by an LSTM, then multi-head self-attention and additive attention to form one user vector.
- **Score:** dot product of the user vector and each candidate's news representation.

### Step 5: Re-ranking
`step5_rerank_mmr.ipynb` re-ranks KLAR's scores with **Maximal Marginal Relevance (MMR)**, trading a little relevance for diversity using the similarity between KLAR's article vectors. The trade-off λ is chosen on Tune and applied once to Test.

`step5_rerank_calibration.ipynb` re-ranks KLAR's top 10 to better match the categories and subcategories in the user's click history. The calibration weight is chosen on Tune and fixed for Test. 

Both notebooks compare their results against KLAR and can run in either order after Step 4.

### Step 6: Diagnostics
`step6_diagnostics.ipynb` compares all models on the Test day with the shared metrics, broken down by cold/warm impressions, category, slate length and the user's history length, and reports the diversity and cold share of the top-10 lists.


## ▶️ How to run

**Data:** MIND-large and all processed files are in the shared Google Drive folder [BT4222_Project](https://drive.google.com/drive/folders/1D4Db1s7DJ5PVK9CFiV-pklpCW5dr1uOM?usp=sharing). The raw data can also be downloaded from https://msnews.github.io (folders `MINDlarge_train`, `MINDlarge_dev`, `MINDlarge_test` under `data/raw/`).

1. In Google Drive, add a shortcut to the shared `BT4222_Project` folder in **My Drive** (right-click → Organise → Add shortcut).
2. Open a notebook in Colab and run its setup cell. It mounts Drive, and `config.py` finds the data at `/content/drive/MyDrive/BT4222_Project`.
3. Run the notebooks in step order. Every processed file is already on the Drive, so any step can also be run on its own.
4. Use a GPU runtime (T4) for steps 3 and 4.

All notebooks are saved with their outputs, so results can be checked without re-running.
