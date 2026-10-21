"""Shared data loaders. Owner: P1.

Everyone loads MIND through these functions so parsing is identical.
"""
import csv

import numpy as np
import pandas as pd

NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract",
             "url", "title_entities", "abstract_entities"]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
TIME_FMT = "%m/%d/%Y %I:%M:%S %p"   # e.g. 11/15/2019 10:22:32 AM


def load_news(path):
    """Read news.tsv.

    quoting=QUOTE_NONE is required: some titles/abstracts contain unmatched
    quote marks, and pandas' default parsing merges fields in those rows.
    """
    return pd.read_csv(path, sep="\t", header=None, names=NEWS_COLS,
                       quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def load_behaviors(path):
    """Read behaviors.tsv with parsed timestamps and empty-string histories."""
    df = pd.read_csv(path, sep="\t", header=None, names=BEHAVIOR_COLS,
                     dtype=str, keep_default_na=False)
    df["time"] = pd.to_datetime(df["time"], format=TIME_FMT)
    return df


def explode_impressions(beh, keep_cols=("impression_id", "user_id", "time")):
    """One row per displayed candidate: keep_cols + news_id + label (1 = clicked).

    For test-split rows (no "-0/-1" suffix) the label is set to -1.
    """
    df = beh[list(keep_cols) + ["impressions"]].copy()
    df["cand"] = df["impressions"].str.split()
    df = df.explode("cand", ignore_index=True)
    has_label = df["cand"].str.contains("-", regex=False)
    df["news_id"] = np.where(has_label, df["cand"].str.rsplit("-", n=1).str[0], df["cand"])
    df["label"] = np.where(has_label, df["cand"].str[-1], "-1").astype(np.int8)
    return df.drop(columns=["impressions", "cand"])


def split_by_date(beh, fit_dates, tune_date):
    """Split training behaviors into fit (date range) and tune (single day)."""
    day = beh["time"].dt.strftime("%Y-%m-%d")
    fit = beh[(day >= fit_dates[0]) & (day <= fit_dates[1])]
    tune = beh[day == tune_date]
    return fit.reset_index(drop=True), tune.reset_index(drop=True)


def negative_sample(cands, neg_per_pos=4, seed=42):
    """Within-impression negative sampling for TRAINING ONLY.

    Keeps every click and, per impression, up to neg_per_pos * n_clicks
    unclicked candidates from that same impression. Never apply to tune/test.
    """
    rng = np.random.default_rng(seed)
    cands = cands.assign(_r=rng.random(len(cands)))
    pos = cands[cands["label"] == 1]
    n_pos = pos.groupby("impression_id").size().rename("_n_pos")
    neg = cands[cands["label"] == 0].join(n_pos, on="impression_id")
    neg = neg.sort_values(["impression_id", "_r"])
    neg["_rank"] = neg.groupby("impression_id").cumcount()
    neg = neg[neg["_rank"] < neg["_n_pos"] * neg_per_pos]
    out = pd.concat([pos, neg.drop(columns=["_n_pos", "_rank"])])
    return out.drop(columns="_r").sort_values(["impression_id", "label"],
                                              ascending=[True, False]).reset_index(drop=True)
