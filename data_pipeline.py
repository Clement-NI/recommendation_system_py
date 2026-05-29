"""
Data pipeline for the NCF recommendation system.

Separates data processing into clear stages:
  1. Extract   — pull raw data from database
  2. Transform — clean, merge, feature engineer
  3. Validate  — quality checks before training
  4. Train     — train model and save weights
  5. Evaluate  — test on held-out data, gate deployment

Usage:
    python data_pipeline.py                    # full pipeline
    python data_pipeline.py --step extract     # run one step
    python data_pipeline.py --step evaluate    # evaluate existing model
"""

import sqlite3
import os
import sys
import json
import math
import argparse
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get("SQLITE_DB_PATH", "./fake_database.db")
OUTPUT_DIR = "./pipeline_output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pt")
DATA_PATH = os.path.join(OUTPUT_DIR, "training_data.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "quality_report.json")

N_FACTORS = 24
MLP_LAYERS = (48, 24, 12)
DROPOUT = 0.3
NUM_EPOCHS = 80
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 5e-4
TEST_RATIO = 0.2
LIKE_THRESHOLD = 4.0
TOP_K = 5

MIN_VIEW_DURATION = 5
IMPLICIT_SCORE_RANGE = (1.0, 4.0)
HISTORY_LOOKBACK_DAYS = 90

# Quality gates
MIN_TOTAL_RATINGS = 1000
MIN_USERS = 50
MIN_ITEMS = 20
MAX_RMSE = 1.2
MIN_HIT_RATE = 0.3


# ---------------------------------------------------------------------------
# Model (same as recommendation_api.py)
# ---------------------------------------------------------------------------

class NeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=32, mlp_layers=(64, 32, 16, 8), dropout=0.3):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.user_mlp_emb = torch.nn.Embedding(n_users, n_factors)
        self.item_mlp_emb = torch.nn.Embedding(n_items, n_factors)
        self.user_biases = torch.nn.Embedding(n_users, 1)
        self.item_biases = torch.nn.Embedding(n_items, 1)
        self.global_bias = torch.nn.Parameter(torch.zeros(1))

        layers = []
        input_dim = 2 * n_factors
        for hidden_dim in mlp_layers:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            input_dim = hidden_dim
        self.mlp = torch.nn.Sequential(*layers)
        self.fusion = torch.nn.Linear(n_factors + mlp_layers[-1], 1)

        torch.nn.init.normal_(self.user_factors.weight, std=0.1)
        torch.nn.init.normal_(self.item_factors.weight, std=0.1)
        torch.nn.init.normal_(self.user_mlp_emb.weight, std=0.1)
        torch.nn.init.normal_(self.item_mlp_emb.weight, std=0.1)
        torch.nn.init.zeros_(self.user_biases.weight)
        torch.nn.init.zeros_(self.item_biases.weight)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.xavier_uniform_(self.fusion.weight)
        torch.nn.init.zeros_(self.fusion.bias)

    def forward(self, data):
        users, items = data[:, 0], data[:, 1]
        gmf_vec = self.user_factors(users) * self.item_factors(items)
        mlp_input = torch.cat([self.user_mlp_emb(users), self.item_mlp_emb(items)], dim=-1)
        mlp_vec = self.mlp(mlp_input)
        fused = torch.cat([gmf_vec, mlp_vec], dim=-1)
        score = self.fusion(fused).squeeze(-1)
        score = score + self.user_biases(users).squeeze(-1) \
                      + self.item_biases(items).squeeze(-1) \
                      + self.global_bias
        return score


class Loader:
    def __init__(self, ratings_df):
        self.ratings = ratings_df.copy()
        self.ratings["userID"] = self.ratings["userID"].astype(str)
        self.ratings["providerID"] = self.ratings["providerID"].astype(str)
        self.userid2idx = {uid: idx for idx, uid in enumerate(self.ratings["userID"].unique())}
        self.providerid2idx = {pid: idx for idx, pid in enumerate(self.ratings["providerID"].unique())}
        self.idx2userid = {v: k for k, v in self.userid2idx.items()}
        self.idx2providerid = {v: k for k, v in self.providerid2idx.items()}
        self.ratings["userID"] = self.ratings["userID"].map(self.userid2idx)
        self.ratings["providerID"] = self.ratings["providerID"].map(self.providerid2idx)
        self.x = torch.tensor(self.ratings[["userID", "providerID"]].values, dtype=torch.long)
        self.y = torch.tensor(self.ratings["score"].fillna(0).values, dtype=torch.float32)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.ratings)


# =========================================================================
# STEP 1: EXTRACT
# =========================================================================

def step_extract():
    """Pull raw data from database."""
    logger.info("=" * 60)
    logger.info("STEP 1: EXTRACT")
    logger.info("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    # Explicit ratings
    explicit_df = pd.read_sql_query(
        "SELECT userID, providerID, score FROM providerReviews", conn
    )
    logger.info(f"  Explicit ratings: {len(explicit_df)} rows")

    # Implicit history
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    all_tables = [t[0] for t in cursor.fetchall()]

    cutoff = datetime.now() - timedelta(days=HISTORY_LOOKBACK_DAYS)
    history_tables = []
    for t in all_tables:
        try:
            table_date = datetime.strptime(t, "%m%d%Y")
            if table_date >= cutoff:
                history_tables.append(t)
        except ValueError:
            continue

    implicit_df = pd.DataFrame()
    if history_tables:
        queries = [f"SELECT user_id, provider_id, duration FROM `{t}` WHERE duration >= {MIN_VIEW_DURATION}"
                   for t in history_tables]
        full_query = " UNION ALL ".join(queries)
        implicit_df = pd.read_sql_query(full_query, conn)
        logger.info(f"  Implicit history: {len(implicit_df)} rows from {len(history_tables)} tables")
    else:
        logger.info("  Implicit history: 0 rows (no tables found)")

    # Providers
    providers_df = pd.read_sql_query("SELECT providerID, providerName, category, genre FROM provider", conn)
    logger.info(f"  Providers: {len(providers_df)} rows")

    conn.close()
    return explicit_df, implicit_df, providers_df


# =========================================================================
# STEP 2: TRANSFORM
# =========================================================================

def step_transform(explicit_df, implicit_df, providers_df):
    """Clean, convert implicit signals, merge."""
    logger.info("=" * 60)
    logger.info("STEP 2: TRANSFORM")
    logger.info("=" * 60)

    # --- Clean explicit ---
    explicit_df["userID"] = explicit_df["userID"].astype(str)
    explicit_df["providerID"] = explicit_df["providerID"].astype(str)
    explicit_df["score"] = pd.to_numeric(explicit_df["score"], errors="coerce")
    before = len(explicit_df)
    explicit_df.dropna(subset=["score", "userID", "providerID"], inplace=True)
    dropped = before - len(explicit_df)
    if dropped > 0:
        logger.info(f"  Dropped {dropped} invalid explicit rows")

    # --- Convert implicit to scores ---
    if not implicit_df.empty:
        implicit_df["user_id"] = implicit_df["user_id"].astype(str)
        implicit_df["provider_id"] = implicit_df["provider_id"].astype(str)

        agg = implicit_df.groupby(["user_id", "provider_id"])["duration"].agg(["sum", "count"]).reset_index()
        agg.rename(columns={"sum": "total_duration", "count": "view_count"}, inplace=True)
        agg["log_duration"] = np.log1p(agg["total_duration"])

        min_log = agg["log_duration"].min()
        max_log = agg["log_duration"].max()
        score_min, score_max = IMPLICIT_SCORE_RANGE

        if max_log > min_log:
            agg["score"] = score_min + (score_max - score_min) * (agg["log_duration"] - min_log) / (max_log - min_log)
        else:
            agg["score"] = (score_min + score_max) / 2

        agg["score"] = agg["score"].clip(upper=score_max)
        implicit_scores = agg[["user_id", "provider_id", "score"]].rename(
            columns={"user_id": "userID", "provider_id": "providerID"}
        )
        logger.info(f"  Implicit scores computed: {len(implicit_scores)} user-item pairs")
    else:
        implicit_scores = pd.DataFrame(columns=["userID", "providerID", "score"])

    # --- Merge: explicit priority ---
    if not implicit_scores.empty:
        explicit_df["key"] = explicit_df["userID"] + "_" + explicit_df["providerID"]
        implicit_scores["key"] = implicit_scores["userID"] + "_" + implicit_scores["providerID"]
        implicit_only = implicit_scores[~implicit_scores["key"].isin(explicit_df["key"])]
        combined = pd.concat([explicit_df.drop(columns=["key"]),
                              implicit_only.drop(columns=["key"])], ignore_index=True)
        logger.info(f"  Merged: {len(explicit_df)} explicit + {len(implicit_only)} implicit = {len(combined)} total")
    else:
        combined = explicit_df.copy()
        logger.info(f"  No implicit data, using {len(combined)} explicit only")

    # --- Save processed data ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined.to_csv(DATA_PATH, index=False)
    logger.info(f"  Saved to {DATA_PATH}")

    return combined


# =========================================================================
# STEP 3: VALIDATE
# =========================================================================

def step_validate(combined_df):
    """Quality checks before training. Returns (passed, report)."""
    logger.info("=" * 60)
    logger.info("STEP 3: VALIDATE")
    logger.info("=" * 60)

    n_ratings = len(combined_df)
    n_users = combined_df["userID"].nunique()
    n_items = combined_df["providerID"].nunique()
    avg_per_user = n_ratings / n_users if n_users > 0 else 0
    score_mean = combined_df["score"].mean()
    score_std = combined_df["score"].std()
    score_min = combined_df["score"].min()
    score_max = combined_df["score"].max()
    null_count = combined_df[["userID", "providerID", "score"]].isnull().sum().sum()

    checks = {
        "total_ratings >= minimum": (n_ratings >= MIN_TOTAL_RATINGS, f"{n_ratings} vs {MIN_TOTAL_RATINGS}"),
        "users >= minimum": (n_users >= MIN_USERS, f"{n_users} vs {MIN_USERS}"),
        "items >= minimum": (n_items >= MIN_ITEMS, f"{n_items} vs {MIN_ITEMS}"),
        "no null values": (null_count == 0, f"{null_count} nulls"),
        "scores in valid range": (score_min >= 0 and score_max <= 5.5, f"[{score_min:.1f}, {score_max:.1f}]"),
    }

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "total_ratings": n_ratings,
            "users": n_users,
            "items": n_items,
            "avg_ratings_per_user": round(avg_per_user, 1),
            "sparsity": round(1 - n_ratings / (n_users * n_items), 4),
            "score_mean": round(score_mean, 3),
            "score_std": round(score_std, 3),
            "score_range": [round(score_min, 1), round(score_max, 1)],
        },
        "checks": {k: {"passed": bool(v[0]), "detail": v[1]} for k, v in checks.items()},
    }

    all_passed = all(v[0] for v in checks.values())

    for name, (passed, detail) in checks.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {name}: {detail}")

    logger.info(f"  Dataset stats: {n_users} users, {n_items} items, {n_ratings} ratings, "
                f"sparsity={report['dataset']['sparsity']:.2%}")

    return all_passed, report


# =========================================================================
# STEP 4: TRAIN
# =========================================================================

def step_train(combined_df):
    """Train NCF model, save weights."""
    logger.info("=" * 60)
    logger.info("STEP 4: TRAIN")
    logger.info("=" * 60)

    train_df, test_df = train_test_split(combined_df, test_size=TEST_RATIO, random_state=42)
    logger.info(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    train_set = Loader(train_df)
    n_users = len(train_set.userid2idx)
    n_items = len(train_set.providerid2idx)
    logger.info(f"  Model: {n_users} users, {n_items} items, {N_FACTORS} factors")

    model = NeuralCollaborativeFiltering(n_users, n_items, N_FACTORS, MLP_LAYERS, DROPOUT)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x).squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Avg Loss: {epoch_loss/len(train_loader):.4f}")

    model.eval()

    # Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_users": n_users,
        "n_items": n_items,
        "n_factors": N_FACTORS,
        "mlp_layers": MLP_LAYERS,
        "userid2idx": train_set.userid2idx,
        "providerid2idx": train_set.providerid2idx,
    }, MODEL_PATH)
    logger.info(f"  Model saved to {MODEL_PATH}")

    return model, train_set, train_df, test_df


# =========================================================================
# STEP 5: EVALUATE
# =========================================================================

def step_evaluate(model, train_set, train_df, test_df):
    """Evaluate on held-out data. Returns (passed, metrics)."""
    logger.info("=" * 60)
    logger.info("STEP 5: EVALUATE")
    logger.info("=" * 60)

    test_df = test_df.copy()
    test_df["userID"] = test_df["userID"].astype(str)
    test_df["providerID"] = test_df["providerID"].astype(str)
    train_df = train_df.copy()
    train_df["userID"] = train_df["userID"].astype(str)
    train_df["providerID"] = train_df["providerID"].astype(str)

    # --- Regression ---
    preds, actuals = [], []
    for _, row in test_df.iterrows():
        uid, pid = str(row["userID"]), str(row["providerID"])
        if uid not in train_set.userid2idx or pid not in train_set.providerid2idx:
            continue
        x = torch.tensor([[train_set.userid2idx[uid], train_set.providerid2idx[pid]]], dtype=torch.long)
        with torch.no_grad():
            preds.append(model(x).item())
        actuals.append(row["score"])

    preds, actuals = np.array(preds), np.array(actuals)
    rmse = math.sqrt(np.mean((preds - actuals) ** 2))
    mae = np.mean(np.abs(preds - actuals))
    logger.info(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # --- Ranking ---
    n_items = len(train_set.providerid2idx)
    all_item_indices = np.array(list(train_set.providerid2idx.values()))
    train_user_items = train_df.groupby("userID")["providerID"].apply(set).to_dict()

    precisions, recalls, ndcgs, hits = [], [], [], []
    for uid in test_df["userID"].unique():
        if uid not in train_set.userid2idx:
            continue
        liked = set(test_df[(test_df["userID"] == uid) & (test_df["score"] >= LIKE_THRESHOLD)]["providerID"])
        if not liked:
            continue

        seen = train_user_items.get(uid, set())
        u_idx = train_set.userid2idx[uid]
        batch = torch.stack([
            torch.tensor([u_idx] * n_items, dtype=torch.long),
            torch.tensor(all_item_indices, dtype=torch.long)
        ], dim=1)

        with torch.no_grad():
            scores = model(batch).numpy()

        top_internal = []
        for idx in np.argsort(scores)[::-1]:
            if train_set.idx2providerid[idx] not in seen:
                top_internal.append(idx)
            if len(top_internal) == TOP_K:
                break

        top_pids = {train_set.idx2providerid[i] for i in top_internal}
        n_hit = len(top_pids & liked)

        precisions.append(n_hit / TOP_K)
        recalls.append(n_hit / len(liked) if liked else 0)
        dcg = sum(1.0 / math.log2(r + 2) for r, idx in enumerate(top_internal) if train_set.idx2providerid[idx] in liked)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(liked), TOP_K)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
        hits.append(1.0 if n_hit > 0 else 0.0)

    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "nrmse": round(rmse / 4.0, 4),
        f"precision@{TOP_K}": round(np.mean(precisions), 4) if precisions else 0,
        f"recall@{TOP_K}": round(np.mean(recalls), 4) if recalls else 0,
        f"ndcg@{TOP_K}": round(np.mean(ndcgs), 4) if ndcgs else 0,
        f"hitrate@{TOP_K}": round(np.mean(hits), 4) if hits else 0,
        "users_evaluated": len(precisions),
        "test_samples": len(preds),
    }

    logger.info(f"  P@{TOP_K}: {metrics[f'precision@{TOP_K}']:.4f}, "
                f"Recall@{TOP_K}: {metrics[f'recall@{TOP_K}']:.4f}, "
                f"NDCG@{TOP_K}: {metrics[f'ndcg@{TOP_K}']:.4f}, "
                f"HitRate@{TOP_K}: {metrics[f'hitrate@{TOP_K}']:.4f}")

    # Quality gates
    gates = {
        f"RMSE <= {MAX_RMSE}": rmse <= MAX_RMSE,
        f"HitRate@{TOP_K} >= {MIN_HIT_RATE}": metrics[f"hitrate@{TOP_K}"] >= MIN_HIT_RATE,
    }

    all_passed = all(gates.values())
    for name, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {name}")

    if all_passed:
        logger.info("  Model APPROVED for deployment")
    else:
        logger.warning("  Model REJECTED — does not meet quality gates")

    return all_passed, metrics


# =========================================================================
# MAIN: run full pipeline
# =========================================================================

def run_pipeline(step=None):
    logger.info("=" * 60)
    logger.info("  NCF DATA PIPELINE")
    logger.info(f"  {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Step 1
    if step and step != "extract":
        combined_df = pd.read_csv(DATA_PATH)
        logger.info(f"Loaded cached data from {DATA_PATH}")
    else:
        explicit_df, implicit_df, providers_df = step_extract()
        combined_df = step_transform(explicit_df, implicit_df, providers_df)
        if step == "extract":
            return

    # Step 3
    passed, report = step_validate(combined_df)
    if not passed:
        logger.error("PIPELINE ABORTED: data quality checks failed")
        report["pipeline_status"] = "ABORTED"
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        sys.exit(1)

    # Step 4
    model, train_set, train_df, test_df = step_train(combined_df)

    # Step 5
    eval_passed, metrics = step_evaluate(model, train_set, train_df, test_df)

    # Final report
    report["metrics"] = metrics
    report["model"] = {
        "n_factors": N_FACTORS,
        "mlp_layers": list(MLP_LAYERS),
        "dropout": DROPOUT,
        "epochs": NUM_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
    }
    report["pipeline_status"] = "APPROVED" if eval_passed else "REJECTED"
    report["model_path"] = MODEL_PATH if eval_passed else None

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved to {REPORT_PATH}")

    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE RESULT")
    print("=" * 60)
    status = report["pipeline_status"]
    print(f"  Status:     {status}")
    print(f"  RMSE:       {metrics['rmse']}")
    print(f"  NRMSE:      {metrics['nrmse']:.1%}")
    print(f"  HitRate@{TOP_K}: {metrics[f'hitrate@{TOP_K}']:.2%}")
    if eval_passed:
        print(f"  Model:      {MODEL_PATH}")
    print("=" * 60)

    if not eval_passed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCF Data Pipeline")
    parser.add_argument("--step", choices=["extract", "validate", "train", "evaluate"],
                        help="Run a specific step only")
    args = parser.parse_args()
    run_pipeline(step=args.step)
