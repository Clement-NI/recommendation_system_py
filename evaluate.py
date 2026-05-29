"""
Automated evaluation for the NCF recommendation model.

Performs three types of tests:
  1. Regression metrics  — RMSE / MAE on held-out ratings
  2. Ranking metrics     — Precision@K, Recall@K, NDCG@K, Hit Rate@K
  3. Differentiation test — verify that distinct users get distinct lists

Usage:
    python evaluate.py
"""

import sqlite3
import math
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Model & Loader — copied from recommendation_api.py to keep evaluate.py
# fully standalone (no Flask import, no server startup needed).
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get("SQLITE_DB_PATH", "./fake_database.db")
TEST_RATIO = 0.2
LIKE_THRESHOLD = 4.0
TOP_K = 5
N_FACTORS = 16
NUM_EPOCHS = 64
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-3
MLP_LAYERS = (32, 16, 8)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ratings():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT userID, providerID, score FROM providerReviews", conn)
    conn.close()
    df["userID"] = df["userID"].astype(str)
    df["providerID"] = df["providerID"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df.dropna(subset=["score", "userID", "providerID"], inplace=True)
    return df


def load_provider_names():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT providerID, providerName FROM provider", conn)
    conn.close()
    df["providerID"] = df["providerID"].astype(str)
    return df.set_index("providerID")["providerName"].to_dict()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(train_df):
    train_set = Loader(train_df)
    n_users = len(train_set.userid2idx)
    n_items = len(train_set.providerid2idx)

    model = NeuralCollaborativeFiltering(n_users, n_items, n_factors=N_FACTORS, mlp_layers=MLP_LAYERS)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(NUM_EPOCHS):
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x).squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 32 == 0:
            avg_loss = sum(loss_fn(model(x).squeeze(), y.squeeze()).item()
                          for x, y in train_loader) / len(train_loader)
            print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Avg Loss: {avg_loss:.4f}")

    model.eval()
    return model, train_set


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_regression(model, train_set, test_df):
    """RMSE and MAE on held-out ratings."""
    preds, actuals = [], []
    for _, row in test_df.iterrows():
        uid, pid = str(row["userID"]), str(row["providerID"])
        if uid not in train_set.userid2idx or pid not in train_set.providerid2idx:
            continue
        u_idx = train_set.userid2idx[uid]
        p_idx = train_set.providerid2idx[pid]
        x = torch.tensor([[u_idx, p_idx]], dtype=torch.long)
        with torch.no_grad():
            pred = model(x).item()
        preds.append(pred)
        actuals.append(row["score"])

    if not preds:
        return None, None, 0

    preds, actuals = np.array(preds), np.array(actuals)
    rmse = math.sqrt(np.mean((preds - actuals) ** 2))
    mae = np.mean(np.abs(preds - actuals))
    return rmse, mae, len(preds)


def evaluate_ranking(model, train_set, test_df, train_df, top_k=TOP_K):
    """Precision@K, Recall@K, NDCG@K, Hit Rate@K."""
    test_df = test_df.copy()
    test_df["userID"] = test_df["userID"].astype(str)
    test_df["providerID"] = test_df["providerID"].astype(str)
    train_df = train_df.copy()
    train_df["userID"] = train_df["userID"].astype(str)
    train_df["providerID"] = train_df["providerID"].astype(str)

    train_user_items = train_df.groupby("userID")["providerID"].apply(set).to_dict()

    n_items = len(train_set.providerid2idx)
    all_item_indices = np.array(list(train_set.providerid2idx.values()))

    precisions, recalls, ndcgs, hits = [], [], [], []

    users_in_test = test_df["userID"].unique()
    for uid in users_in_test:
        if uid not in train_set.userid2idx:
            continue

        liked = set(
            test_df[(test_df["userID"] == uid) & (test_df["score"] >= LIKE_THRESHOLD)]["providerID"]
        )
        if not liked:
            continue

        seen = train_user_items.get(uid, set())

        u_idx = train_set.userid2idx[uid]
        user_tensor = torch.tensor([u_idx] * n_items, dtype=torch.long)
        item_tensor = torch.tensor(all_item_indices, dtype=torch.long)
        batch = torch.stack([user_tensor, item_tensor], dim=1)

        with torch.no_grad():
            scores = model(batch).numpy()

        ranked = np.argsort(scores)[::-1]
        top_internal = []
        for idx in ranked:
            pid = train_set.idx2providerid[idx]
            if pid not in seen:
                top_internal.append(idx)
            if len(top_internal) == top_k:
                break
        top_pids = {train_set.idx2providerid[i] for i in top_internal}

        n_hit = len(top_pids & liked)

        precision = n_hit / top_k
        recall = n_hit / len(liked) if liked else 0.0

        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, idx in enumerate(top_internal)
            if train_set.idx2providerid[idx] in liked
        )
        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(liked), top_k)))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        hits.append(1.0 if n_hit > 0 else 0.0)

    return {
        f"Precision@{top_k}": np.mean(precisions) if precisions else 0,
        f"Recall@{top_k}": np.mean(recalls) if recalls else 0,
        f"NDCG@{top_k}": np.mean(ndcgs) if ndcgs else 0,
        f"HitRate@{top_k}": np.mean(hits) if hits else 0,
        "users_evaluated": len(precisions),
    }


def test_differentiation(model, train_set, n_users=5, top_k=TOP_K):
    """Check that different users receive different recommendation lists."""
    n_items = len(train_set.providerid2idx)
    all_item_indices = np.array(list(train_set.providerid2idx.values()))

    sample_uids = list(train_set.userid2idx.keys())[:n_users]
    lists = {}

    for uid in sample_uids:
        u_idx = train_set.userid2idx[uid]
        user_tensor = torch.tensor([u_idx] * n_items, dtype=torch.long)
        item_tensor = torch.tensor(all_item_indices, dtype=torch.long)
        batch = torch.stack([user_tensor, item_tensor], dim=1)

        with torch.no_grad():
            scores = model(batch).numpy()

        top_internal = np.argsort(scores)[::-1][:top_k]
        top_pids = tuple(train_set.idx2providerid[i] for i in top_internal)
        lists[uid] = top_pids

    unique_lists = len(set(lists.values()))
    return lists, unique_lists, n_users


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  NCF Model Evaluation")
    print("=" * 70)

    print(f"\n[1/4] Loading data from {DB_PATH} ...")
    df = load_ratings()
    print(f"  Total ratings: {len(df)}")
    print(f"  Users: {df['userID'].nunique()}, Providers: {df['providerID'].nunique()}")

    print(f"\n[2/4] Splitting data (train {1-TEST_RATIO:.0%} / test {TEST_RATIO:.0%}) ...")
    train_df, test_df = train_test_split(df, test_size=TEST_RATIO, random_state=42)
    print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    print(f"\n[3/4] Training NCF model ({NUM_EPOCHS} epochs) ...")
    model, train_set = train_model(train_df)
    print("  Training complete.")

    print(f"\n[4/4] Evaluating ...")

    # --- Regression ---
    rmse, mae, n_samples = evaluate_regression(model, train_set, test_df)
    print(f"\n  Regression Metrics ({n_samples} test samples):")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")

    # --- Ranking ---
    ranking = evaluate_ranking(model, train_set, test_df, train_df, top_k=TOP_K)
    print(f"\n  Ranking Metrics ({ranking['users_evaluated']} users with liked items):")
    print(f"    Precision@{TOP_K} : {ranking[f'Precision@{TOP_K}']:.4f}")
    print(f"    Recall@{TOP_K}    : {ranking[f'Recall@{TOP_K}']:.4f}")
    print(f"    NDCG@{TOP_K}      : {ranking[f'NDCG@{TOP_K}']:.4f}")
    print(f"    HitRate@{TOP_K}   : {ranking[f'HitRate@{TOP_K}']:.4f}")

    # --- Differentiation ---
    pnames = load_provider_names()
    lists, unique, total = test_differentiation(model, train_set)
    print(f"\n  Differentiation Test:")
    for uid, top in lists.items():
        names = [pnames.get(pid, pid) for pid in top]
        print(f"    {uid:<10} -> {names}")
    status = "PASS" if unique == total else "FAIL"
    print(f"    Unique lists: {unique}/{total}  {status}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  P@{TOP_K}={ranking[f'Precision@{TOP_K}']:.2%}"
          f"  NDCG@{TOP_K}={ranking[f'NDCG@{TOP_K}']:.2%}  Diff={unique}/{total} {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
