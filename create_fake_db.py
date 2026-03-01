# create_fake_db.py
# Generates a SQLite database with fake data matching the KaopuVIP schema.
# Users are assigned preference profiles (favorite categories) so the
# collaborative-filtering model can learn distinct per-user tastes and
# produce differentiated recommendations.
# Run this once to create 'fake_kaopuvip.db' that the recommendation API can use.

import sqlite3
import random
import os
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "fake_database.db")

# --- Configuration ---
NUM_PROVIDERS = 50
NUM_USERS = 200
MIN_REVIEWS_PER_USER = 12
MAX_REVIEWS_PER_USER = 25
NUM_HISTORY_DAYS = 7
NUM_HISTORY_RECORDS_PER_DAY = 300

CATEGORIES = [
    "Massage", "Yoga", "Fitness", "Meditation", "Nutrition",
    "Physiotherapy", "Acupuncture", "Chiropractic", "Pilates", "Spa"
]

PROVIDER_PREFIXES = [
    "Zen", "Harmony", "Pure", "Golden", "Crystal",
    "Royal", "Elite", "Pacific", "Summit", "Lotus",
    "Sunrise", "Moonlight", "Ocean", "Mountain", "Forest",
    "River", "Sky", "Star", "Diamond", "Silver"
]

PROVIDER_SUFFIXES = [
    "Wellness", "Studio", "Center", "Clinic", "Hub",
    "Place", "Retreat", "Lounge", "Lab", "Zone"
]

# Positive / negative comment pools keyed by score range
POSITIVE_COMMENTS = ["Great!", "Excellent", "Will come again", "Highly recommend", "Loved it"]
NEUTRAL_COMMENTS = ["Good service", "Not bad", "Decent", "OK experience", ""]
NEGATIVE_COMMENTS = ["Average", "Could be better", "Disappointing", "Would not return", ""]


def _score_for_preference(liked: bool) -> float:
    """Return a biased score depending on whether the user likes the category."""
    if liked:
        return round(random.gauss(4.2, 0.5), 1)
    else:
        return round(random.gauss(2.3, 0.7), 1)


def _clamp_score(score: float) -> float:
    return max(1.0, min(5.0, score))


def _comment_for_score(score: float) -> str:
    if score >= 4.0:
        return random.choice(POSITIVE_COMMENTS)
    if score >= 2.5:
        return random.choice(NEUTRAL_COMMENTS)
    return random.choice(NEGATIVE_COMMENTS)


def create_database():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # --- Main database tables ---

    # provider table
    cursor.execute("""
        CREATE TABLE provider (
            providerID TEXT PRIMARY KEY,
            providerName TEXT NOT NULL,
            category TEXT,
            description TEXT,
            genre TEXT,
            rating REAL DEFAULT 0,
            isActive INTEGER DEFAULT 1
        )
    """)

    # providerReviews table
    cursor.execute("""
        CREATE TABLE providerReviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userID TEXT NOT NULL,
            providerID TEXT NOT NULL,
            score REAL NOT NULL,
            comment TEXT,
            createdAt TEXT
        )
    """)

    # --- Generate providers ---
    providers = []
    provider_category_map = {}  # provider_id -> category
    used_names = set()
    for i in range(1, NUM_PROVIDERS + 1):
        provider_id = str(i)
        # Generate unique name
        while True:
            name = f"{random.choice(PROVIDER_PREFIXES)} {random.choice(PROVIDER_SUFFIXES)}"
            if name not in used_names:
                used_names.add(name)
                break

        category = random.choice(CATEGORIES)
        provider_category_map[provider_id] = category

        # Some providers have multi-genre (pipe-separated)
        if random.random() < 0.3:
            genres = random.sample(CATEGORIES, k=random.randint(2, 3))
            genre = "|".join(genres)
        else:
            genre = category

        description = f"A top-quality {category.lower()} service provider."
        rating = round(random.uniform(2.5, 5.0), 1)
        providers.append((provider_id, name, category, description, genre, rating, 1))

    cursor.executemany(
        "INSERT INTO provider (providerID, providerName, category, description, genre, rating, isActive) VALUES (?, ?, ?, ?, ?, ?, ?)",
        providers
    )

    # --- Build user preference profiles ---
    # Each user likes 2-4 categories.  Reviews for liked categories get high
    # scores; reviews for other categories get low scores.  This gives the CF
    # model clear per-user signals to learn from.
    user_ids = [f"user_{i}" for i in range(1, NUM_USERS + 1)]
    provider_ids = [str(i) for i in range(1, NUM_PROVIDERS + 1)]

    user_preferences = {}  # user_id -> set of liked categories
    for uid in user_ids:
        num_liked = random.randint(2, 4)
        user_preferences[uid] = set(random.sample(CATEGORIES, k=num_liked))

    # Pre-group providers by category for efficient sampling
    providers_by_category = {}
    for pid, cat in provider_category_map.items():
        providers_by_category.setdefault(cat, []).append(pid)

    # --- Generate reviews (explicit ratings) ---
    reviews = []
    seen_pairs = set()

    for uid in user_ids:
        liked_cats = user_preferences[uid]
        num_reviews = random.randint(MIN_REVIEWS_PER_USER, MAX_REVIEWS_PER_USER)

        # ~70 % of a user's reviews go to liked categories
        num_liked_reviews = int(num_reviews * 0.7)
        num_other_reviews = num_reviews - num_liked_reviews

        liked_providers = [p for p in provider_ids if provider_category_map[p] in liked_cats]
        other_providers = [p for p in provider_ids if provider_category_map[p] not in liked_cats]

        for pool, count, liked in [
            (liked_providers, num_liked_reviews, True),
            (other_providers, num_other_reviews, False),
        ]:
            if not pool:
                continue
            sampled = random.sample(pool, k=min(count, len(pool)))
            for pid in sampled:
                pair = (uid, pid)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                score = _clamp_score(_score_for_preference(liked))
                comment = _comment_for_score(score)
                created = (datetime.now() - timedelta(days=random.randint(0, 90))).isoformat()
                reviews.append((uid, pid, score, comment, created))

    cursor.executemany(
        "INSERT INTO providerReviews (userID, providerID, score, comment, createdAt) VALUES (?, ?, ?, ?, ?)",
        reviews
    )

    # --- Generate history tables (date-named like MMDDYYYY) ---
    # Users interact more often (and longer) with providers in liked categories.
    today = datetime.now()
    for day_offset in range(NUM_HISTORY_DAYS):
        date = today - timedelta(days=day_offset)
        table_name = date.strftime("%m%d%Y")

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                duration INTEGER NOT NULL
            )
        """)

        history_rows = []
        for _ in range(NUM_HISTORY_RECORDS_PER_DAY):
            uid = random.choice(user_ids)
            liked_cats = user_preferences[uid]

            # 75 % chance the user visits a liked-category provider
            if random.random() < 0.75:
                cat = random.choice(list(liked_cats))
                pid = random.choice(providers_by_category[cat])
                duration = random.randint(120, 600)  # longer sessions
            else:
                pid = random.choice(provider_ids)
                duration = random.randint(1, 180)  # shorter sessions

            history_rows.append((uid, pid, duration))

        cursor.executemany(
            f"INSERT INTO `{table_name}` (user_id, provider_id, duration) VALUES (?, ?, ?)",
            history_rows
        )

    conn.commit()
    conn.close()

    print(f"Fake database created at: {DB_PATH}")
    print(f"  - {NUM_PROVIDERS} providers across {len(CATEGORIES)} categories")
    print(f"  - {len(reviews)} reviews from {NUM_USERS} users "
          f"(avg {len(reviews) / NUM_USERS:.1f} per user)")
    print(f"  - {NUM_HISTORY_DAYS} history tables with ~{NUM_HISTORY_RECORDS_PER_DAY} records each")
    print(f"  - Each user has 2-4 preferred categories for differentiated recommendations")


if __name__ == "__main__":
    create_database()
