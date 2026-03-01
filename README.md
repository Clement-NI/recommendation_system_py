# Recommendation System

A hybrid recommendation engine built with **PyTorch** and **Flask**. It combines collaborative filtering (matrix factorisation), content-based filtering (cosine similarity), and popularity ranking to serve personalised provider recommendations via a REST API.

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Environments](#environments)
- [Testing](#testing)
- [Configuration](#configuration)

## Architecture

```
┌──────────────┐   POST /api/recommendations   ┌──────────────────┐
│  Client App  │ ─────────────────────────────► │  Flask API       │
│  (frontend)  │ ◄───────────────────────────── │  (port 5100)     │
└──────────────┘         JSON response          └────────┬─────────┘
                                                         │
                         ┌───────────────────────────────┼───────────────────────┐
                         │                               │                       │
                ┌────────▼────────┐          ┌───────────▼──────┐    ┌───────────▼──────┐
                │  Collaborative  │          │  Content-Based   │    │   Popularity     │
                │  Filtering (CF) │          │  Filtering (CB)  │    │   Ranking        │
                │  PyTorch MF     │          │  Cosine Sim +    │    │   Avg scores     │
                │  with biases    │          │  Fuzzy Matching  │    │   (anonymous)    │
                └────────┬────────┘          └───────────┬──────┘    └───────────┬──────┘
                         │                               │                       │
                         └───────────────┬───────────────┘                       │
                                         ▼                                       │
                                  ┌──────────────┐                               │
                                  │ Hybrid Merge  │◄──────────────────────────────┘
                                  │ (weighted α)  │
                                  └──────┬───────┘
                                         ▼
                                   JSON Response
```

## Project Structure

```
recommendation_system_py/
├── recommendation_api.py    # Main Flask application & ML models
├── db_adapter.py            # Database abstraction (MySQL / SQLite)
├── create_fake_db.py        # Generates SQLite DB with realistic fake data
├── test_differentiation.py  # Verifies per-user recommendation diversity
├── requirements.txt         # Python dependencies
├── fake_kaopuvip.db         # Generated SQLite database (standalone mode)
├── .gitignore
└── README.md
```

### File Details

| File | Purpose |
|---|---|
| `recommendation_api.py` | Flask server with all API endpoints, the `MatrixFactorization` PyTorch model, data loading, training loop, content-based filtering, popularity calculation, and a background re-training thread. |
| `db_adapter.py` | Abstraction layer that lets the same SQL queries work against both MySQL (production) and SQLite (standalone). Translates query syntax and wraps cursors to return consistent dict-style rows. |
| `create_fake_db.py` | Creates `fake_kaopuvip.db` with 50 providers across 10 categories, 200 users with preference profiles (2-4 liked categories each), ~3,500 reviews, and 7 days of browsing history. |
| `test_differentiation.py` | End-to-end test that starts the server, queries 5 different users, and asserts they each receive different top-5 recommendations. |

## How It Works

### Data Sources

1. **Explicit ratings** - `providerReviews` table: user scores (1-5) with comments.
2. **Implicit signals** - Date-named history tables (`MMDDYYYY`): browsing durations per user-provider pair, converted to implicit scores (1-4) via log-scaled aggregation.
3. **Provider metadata** - `provider` table: name, category, genre (pipe-separated), used for content-based features.

### Recommendation Strategies

| User State | Has Search Query | Strategy |
|---|---|---|
| Logged-in | No | **Collaborative Filtering** - personalised scores from the trained MF model |
| Logged-in | Yes | **Hybrid** - weighted merge of CF + content-based scores (`alpha` controls the blend) |
| Anonymous | Yes | **Content-Based** - fuzzy-match the query to a provider name, then rank by cosine similarity on genre/category features |
| Anonymous | No | **Popularity-Based** - top providers by average rating (cached at startup) |

### Model Architecture

The core model is **Matrix Factorisation with biases**:

```
predicted_score = dot(user_embedding, item_embedding)
                + user_bias + item_bias + global_bias
```

- **32 latent factors** per user/item
- User & item **bias terms** separate popularity effects from personal preference patterns
- Trained with **Adam** (lr=1e-3, weight_decay=1e-5) and **MSE loss**
- Weights initialised with `normal(std=0.1)`; biases initialised to zero

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd recommendation_system_py

# Install dependencies
pip install -r requirements.txt
```

### Quick Start (Standalone / SQLite)

```bash
# 1. Generate the fake database
python create_fake_db.py

# 2. Start the server
RECOMMENDATION_ENV=standalone python recommendation_api.py
```

The API will be available at `http://localhost:5100`.

### Quick Start (MySQL / Production)

Set the environment variables or let it auto-detect:

```bash
RECOMMENDATION_ENV=production USE_SQLITE=false python recommendation_api.py
```

The server connects to the MySQL databases defined in `DB_CONFIGS` inside `recommendation_api.py`.

## API Reference

### `POST /api/recommendations`

Returns personalised recommendations.

**Request body (JSON):**

| Field | Type | Default | Description |
|---|---|---|---|
| `user_id` | string \| null | `null` | User ID. Omit or `null` for anonymous. |
| `search_query` | string | `""` | Optional text query for content-based filtering. |
| `top_n` | int | `10` | Number of recommendations to return. |
| `alpha` | float | `0.5` | CF weight in hybrid mode (0 = pure CB, 1 = pure CF). |
| `environment` | string | `"production"` | Target environment (`standalone`, `dev`, `production`). |

**Response:**

```json
{
  "user_id": "user_1",
  "search_query": null,
  "recommendation_type": "collaborative",
  "recommendations": [
    {"providerID": "12", "providerName": "Star Wellness", "hybrid_score": 4.23},
    {"providerID": "7",  "providerName": "Pure Place",    "hybrid_score": 4.01}
  ],
  "environment": "standalone",
  "inference_device": "cpu"
}
```

`recommendation_type` will be one of: `collaborative`, `hybrid`, `content_based_anonymous`, `popularity_based`.

### `GET /api/health`

Returns system status, model state, environment info, and initialisation history.

### `POST /api/environment`

Manually switch environment. Body: `{"environment": "dev"}`.

### `POST /api/auto-detect-environment`

Re-detect and auto-switch environment based on env vars and config files.

### `POST /api/force-dev`

Force switch to the dev environment and retrain the model.

## Environments

| Environment | Database | Training Epochs | Re-init Interval |
|---|---|---|---|
| `standalone` | SQLite (`fake_kaopuvip.db`) | 128 | 10 min |
| `dev` | MySQL (thearchyhelios.com) | 128 | 10 min |
| `production` | MySQL (AWS RDS) | 64 | 30 min |

Environment is detected automatically in this priority order:
1. `RECOMMENDATION_ENV` environment variable
2. Presence of `.env.dev` or `dev_mode` file
3. `USE_SQLITE=true` defaults to `standalone`
4. Falls back to `production`

## Testing

Run the end-to-end differentiation test:

```bash
RECOMMENDATION_ENV=standalone python test_differentiation.py
```

This starts the server, waits for the model to train, queries 5 users, and checks that each receives a unique set of recommendations.

Example output:

```
user_1   -> ['Star Wellness', 'Star Lab', 'Pure Place', ...]
user_10  -> ['Pacific Wellness', 'Mountain Lounge', 'Summit Center', ...]
user_50  -> ['Crystal Retreat', 'Diamond Center', 'Star Zone', ...]

Unique recommendation lists: 5 / 5
PASS: Users received DIFFERENT recommendations.
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RECOMMENDATION_ENV` | *(auto-detect)* | Set to `standalone`, `dev`, or `production` |
| `USE_SQLITE` | `"true"` | `"true"` for SQLite, `"false"` for MySQL |
| `SQLITE_DB_PATH` | `./fake_kaopuvip.db` | Path to the SQLite database file |

### Key Constants (in `recommendation_api.py`)

| Constant | Value | Description |
|---|---|---|
| `HISTORY_LOOKBACK_DAYS` | 90 | Days of browsing history to include |
| `IMPLICIT_SCORE_RANGE` | (1.0, 4.0) | Min/max for implicit scores |
| `MIN_VIEW_DURATION` | 5 | Ignore history entries shorter than 5 seconds |

### Fake Data Configuration (in `create_fake_db.py`)

| Constant | Value |
|---|---|
| `NUM_PROVIDERS` | 50 |
| `NUM_USERS` | 200 |
| `MIN_REVIEWS_PER_USER` | 12 |
| `MAX_REVIEWS_PER_USER` | 25 |
| `NUM_HISTORY_DAYS` | 7 |
| `CATEGORIES` | Massage, Yoga, Fitness, Meditation, Nutrition, Physiotherapy, Acupuncture, Chiropractic, Pilates, Spa |
