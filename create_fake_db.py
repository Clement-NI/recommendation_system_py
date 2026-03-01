# create_fake_db.py
# Generates a SQLite database with fake data matching the KaopuVIP schema.
# Run this once to create 'fake_kaopuvip.db' that the recommendation API can use.

import sqlite3
import random
import os
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "fake_kaopuvip.db")

# --- Configuration ---
NUM_PROVIDERS = 50
NUM_USERS = 200
NUM_REVIEWS = 800
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

    # --- Generate user IDs ---
    user_ids = [f"user_{i}" for i in range(1, NUM_USERS + 1)]
    provider_ids = [str(i) for i in range(1, NUM_PROVIDERS + 1)]

    # --- Generate reviews (explicit ratings) ---
    reviews = []
    seen_pairs = set()
    for _ in range(NUM_REVIEWS):
        uid = random.choice(user_ids)
        pid = random.choice(provider_ids)
        pair = (uid, pid)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        score = round(random.uniform(1.0, 5.0), 1)
        comment = random.choice(["Great!", "Good service", "Average", "Excellent", "Not bad", "Will come again", ""])
        created = (datetime.now() - timedelta(days=random.randint(0, 90))).isoformat()
        reviews.append((uid, pid, score, comment, created))

    cursor.executemany(
        "INSERT INTO providerReviews (userID, providerID, score, comment, createdAt) VALUES (?, ?, ?, ?, ?)",
        reviews
    )

    # --- Generate history tables (date-named like MMDDYYYY) ---
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
            pid = random.choice(provider_ids)
            duration = random.randint(1, 600)  # 1 to 600 seconds
            history_rows.append((uid, pid, duration))

        cursor.executemany(
            f"INSERT INTO `{table_name}` (user_id, provider_id, duration) VALUES (?, ?, ?)",
            history_rows
        )

    conn.commit()
    conn.close()

    print(f"Fake database created at: {DB_PATH}")
    print(f"  - {NUM_PROVIDERS} providers")
    print(f"  - {len(reviews)} reviews from {NUM_USERS} users")
    print(f"  - {NUM_HISTORY_DAYS} history tables with ~{NUM_HISTORY_RECORDS_PER_DAY} records each")


if __name__ == "__main__":
    create_database()
