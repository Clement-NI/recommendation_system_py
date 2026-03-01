"""Quick test: verify that different users get different recommendations."""

import subprocess
import sys
import time
import requests
import json

SERVER_URL = "http://127.0.0.1:5100"
TEST_USERS = ["user_1", "user_10", "user_50", "user_100", "user_150"]
TOP_N = 5


def wait_for_server(timeout=120):
    """Wait for the Flask server to respond."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{SERVER_URL}/api/health", timeout=3)
            data = r.json()
            if data.get("model_loaded"):
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2)
    return False


def get_recommendations(user_id=None, top_n=TOP_N):
    """Fetch recommendations for a given user (POST with JSON body)."""
    payload = {"top_n": top_n, "environment": "standalone"}
    if user_id is not None:
        payload["user_id"] = user_id
    r = requests.post(
        f"{SERVER_URL}/api/recommendations",
        json=payload,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def main():
    # Start the server in the background
    print("Starting recommendation server...")
    proc = subprocess.Popen(
        [sys.executable, "recommendation_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**__import__("os").environ, "RECOMMENDATION_ENV": "standalone"},
    )

    try:
        print("Waiting for model to train and server to be ready...")
        if not wait_for_server():
            print("ERROR: Server did not become ready in time.")
            proc.terminate()
            sys.exit(1)

        print(f"\nServer ready. Querying {len(TEST_USERS)} users (top_n={TOP_N}):\n")

        results = {}
        for uid in TEST_USERS:
            data = get_recommendations(uid)
            names = [r["providerName"] for r in data["recommendations"]]
            results[uid] = names
            print(f"  {uid:12s} -> {names}")

        # --- Check differentiation ---
        name_lists = list(results.values())
        unique_lists = set(tuple(n) for n in name_lists)

        print(f"\n--- Differentiation check ---")
        print(f"  Unique recommendation lists: {len(unique_lists)} / {len(TEST_USERS)}")

        if len(unique_lists) == 1:
            print("  FAIL: All users got the SAME recommendations.")
            sys.exit(1)
        else:
            print("  PASS: Users received DIFFERENT recommendations.")

        # Also test anonymous user (should get popularity-based)
        anon = get_recommendations(user_id=None)
        anon_names = [r["providerName"] for r in anon["recommendations"]]
        print(f"\n  Anonymous     -> {anon_names}  (type: {anon['recommendation_type']})")

    finally:
        proc.terminate()
        proc.wait()
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
