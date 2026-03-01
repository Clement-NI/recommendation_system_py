# recommendation_api.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process  # Using fuzzy matching library
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import logging
import threading
import time
from datetime import datetime, timedelta  # Added timedelta

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# --- Configure logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Configuration ---
# Database connection details (recommend using environment variables or config files)
DB_CONFIGS = {
    "production": {
        "main": {
            "host": "cobook-2.cbvyujhj4dwc.eu-west-3.rds.amazonaws.com",
            "user": "root",
            "passwd": "Kaopu38000VIP",  # WARNING: hardcoded password
            "database": "kaopuvipv2"
        },
        "history": {
            "host": "cobook-2.cbvyujhj4dwc.eu-west-3.rds.amazonaws.com",
            "user": "root",
            "passwd": "Kaopu38000VIP",  # WARNING: hardcoded password
            "database": "kaopuvipv2_history_provider"
        }
    },
    "dev": {
        "main": {
            "host": "thearchyhelios.com",
            "user": "root",
            "passwd": "Kaopu38000VIP",
            "database": "kaopuvipv2"
        },
        "history": {
            "host": "thearchyhelios.com",
            "user": "root",
            "passwd": "Kaopu38000VIP",
            "database": "kaopuvipv2_history_provider"
        }
    }
}

HISTORY_LOOKBACK_DAYS = 90  # How many days of history to consider
IMPLICIT_SCORE_RANGE = (1.0, 4.0)  # Implicit score range (viewing behavior, etc.) slightly below explicit ratings (1-5)
MIN_VIEW_DURATION = 5  # Ignore view records shorter than 5 seconds (optional)


# --- PyTorch Model Definition ---
class MatrixFactorization(torch.nn.Module):
    """Matrix factorization model class"""

    def __init__(self, n_users, n_items, n_factors=30):
        super().__init__()
        # Embedding layers for users and items
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        # Initialize weights
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)

    def forward(self, data):
        # Forward pass to compute predicted scores
        users, items = data[:, 0], data[:, 1]
        return (self.user_factors(users) * self.item_factors(items)).sum(1)


# --- PyTorch Data Loader ---
class Loader:
    """Data loader class for handling ratings data"""

    def __init__(self, ratings_df):
        self.ratings = ratings_df.copy()
        # Ensure IDs are string type to handle UUIDs, etc.
        self.ratings["userID"] = self.ratings["userID"].astype(str)
        self.ratings["providerID"] = self.ratings["providerID"].astype(str)

        # Create mappings from user/provider IDs to internal indices
        self.userid2idx = {uuid: idx for idx,
                           uuid in enumerate(self.ratings["userID"].unique())}
        self.providerid2idx = {uuid: idx for idx, uuid in enumerate(
            self.ratings["providerID"].unique())}
        # Create reverse mappings from internal indices to user/provider IDs
        self.idx2userid = {idx: uuid for uuid, idx in self.userid2idx.items()}
        self.idx2providerid = {idx: uuid for uuid,
                               idx in self.providerid2idx.items()}

        # Map original IDs to internal indices
        self.ratings["userID"] = self.ratings["userID"].map(self.userid2idx)
        self.ratings["providerID"] = self.ratings["providerID"].map(
            self.providerid2idx)

        # Convert to PyTorch tensors
        self.x = torch.tensor(
            self.ratings[["userID", "providerID"]].values, dtype=torch.long)
        self.y = torch.tensor(self.ratings["score"].fillna(
            0).values, dtype=torch.float32)  # Fill missing scores with 0

    def __getitem__(self, index):
        # Get data at specified index
        return self.x[index], self.y[index]

    def __len__(self):
        # Return dataset size
        return len(self.ratings)


# --- Global Variables ---
model = None                # Stores the trained model
provider_names = None       # Stores providerID -> providerName mapping
providers_df = None         # Stores full provider data
ratings_df_global = None    # Stores full ratings data
train_set = None            # Stores the Loader instance
cosine_sim = None           # Stores item-to-item cosine similarity matrix (for CBF)
provider_idx = None         # Stores providerName -> index in providers_df (for CBF)
popular_items_cache = []    # Stores list of popular items (for anonymous users)
reinit_thread = None        # Background re-initialization thread
is_running = True           # Controls background thread running state
last_init_time = None       # Time of last successful initialization
init_history = []           # Initialization history records

# --- Auto-detect and set computation device (CPU or GPU) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(
        f"PyTorch version: {torch.__version__}. CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info(
        f"PyTorch version: {torch.__version__}. MPS available. Using MPS.")
else:
    device = torch.device("cpu")
    logger.info(f"PyTorch version: {torch.__version__}. CUDA not available. Using CPU.")


def calculate_popularity(ratings_df, providers_df, min_votes=5, top_n=20):
    """Calculate popular items based on average rating"""
    if ratings_df is None or providers_df is None:
        logger.warning("Cannot calculate popular items: missing ratings or provider data.")
        return []

    try:
        # Calculate average score and rating count per provider
        provider_stats = ratings_df.groupby('providerID')['score'].agg([
            'mean', 'count']).reset_index()

        # Filter out providers with fewer ratings than the threshold
        qualified_providers = provider_stats[provider_stats['count'] >= min_votes]

        # Sort by average score descending
        popular_providers = qualified_providers.sort_values(
            by='mean', ascending=False)

        # Get top-ranked provider IDs
        top_provider_ids = popular_providers.head(top_n)['providerID'].tolist()

        # Create ID-to-name and ID-to-score mappings
        id_to_name_map = providers_df.set_index(
            'providerID')['providerName'].to_dict()
        id_to_mean_score_map = popular_providers.set_index('providerID')[
            'mean'].to_dict()

        # Build popular items list
        popular_list = []
        for pid in top_provider_ids:
            if pid in id_to_name_map:
                popular_list.append({
                    'providerID': pid,
                    'providerName': id_to_name_map[pid],
                    # Use 'popularity_score' to distinguish from other recommendation scores
                    'popularity_score': round(id_to_mean_score_map.get(pid, 0), 4)
                })
        logger.info(f"Calculated Top {len(popular_list)} popular items (min_votes={min_votes}).")
        return popular_list

    except Exception as e:
        logger.error(f"Error calculating popular items: {e}", exc_info=True)
        return []


def initialize_model(environment="production"):
    """Initialize model using explicit ratings and implicit history"""
    # Declare global variables to be modified
    global model, provider_names, train_set, cosine_sim, provider_idx, providers_df
    global ratings_df_global, popular_items_cache  # Keep explicit ratings for popularity calculation

    logger.info(f"--- Starting model initialization (environment: {environment}) ---")
    
    # Select database config based on environment
    if environment not in DB_CONFIGS:
        logger.error(f"Invalid environment parameter: {environment}, using default production config")
        environment = "production"
    
    db_config_main = DB_CONFIGS[environment]["main"]
    db_config_history = DB_CONFIGS[environment]["history"]
    
    explicit_ratings_df = pd.DataFrame()
    implicit_ratings_df = pd.DataFrame()
    providers_df_local = pd.DataFrame()  # Use local variable to store provider data

    try:
        # --- Fetch provider data (from main database) ---
        logger.info(f"Fetching provider data from {environment} environment main database...")
        conn_main = mysql.connector.connect(**db_config_main)
        cursor_main = conn_main.cursor(dictionary=True)
        cursor_main.execute("SELECT * FROM provider")
        providers_data = cursor_main.fetchall()
        providers_df_local = pd.DataFrame(providers_data)
        providers_df_local["providerID"] = providers_df_local["providerID"].astype(str)
        cursor_main.close()
        conn_main.close()
        logger.info(f"Fetched {len(providers_df_local)} providers.")
        # Assign to global variable after fetching
        providers_df = providers_df_local

        # --- Fetch explicit ratings (from main database) ---
        logger.info(f"Fetching explicit ratings (providerReviews) from {environment} environment main database...")
        conn_main = mysql.connector.connect(**db_config_main)
        cursor_main = conn_main.cursor(dictionary=True)
        cursor_main.execute("SELECT userID, providerID, score FROM providerReviews")  # Select only needed columns
        rating_data = cursor_main.fetchall()
        explicit_ratings_df = pd.DataFrame(rating_data)
        # Ensure correct types and naming
        explicit_ratings_df["userID"] = explicit_ratings_df["userID"].astype(str)
        explicit_ratings_df["providerID"] = explicit_ratings_df["providerID"].astype(str)
        explicit_ratings_df["score"] = pd.to_numeric(explicit_ratings_df["score"], errors='coerce')
        explicit_ratings_df.dropna(subset=['score'], inplace=True)  # Remove rows where score cannot be converted to numeric
        cursor_main.close()
        conn_main.close()
        logger.info(f"Fetched {len(explicit_ratings_df)} explicit ratings.")
        # Assign to global variable after fetching - used for popularity calculation
        ratings_df_global = explicit_ratings_df.copy()

        # --- Fetch and process implicit history data ---
        implicit_ratings_df = fetch_and_process_history(db_config_history, HISTORY_LOOKBACK_DAYS, MIN_VIEW_DURATION)

        # --- Merge explicit and implicit data ---
        logger.info("Merging explicit and implicit ratings...")
        if not implicit_ratings_df.empty:
            # Create unique key for merging/filtering
            explicit_ratings_df['user_provider_key'] = explicit_ratings_df['userID'] + "_" + explicit_ratings_df['providerID']
            implicit_ratings_df['user_provider_key'] = implicit_ratings_df['userID'] + "_" + implicit_ratings_df['providerID']

            # Keep only implicit interactions that don't exist in explicit ratings
            implicit_only_df = implicit_ratings_df[~implicit_ratings_df['user_provider_key'].isin(explicit_ratings_df['user_provider_key'])]

            # Combine explicit ratings and filtered implicit ratings
            combined_ratings_df = pd.concat([explicit_ratings_df, implicit_only_df], ignore_index=True)

            # Remove temporary key
            combined_ratings_df.drop(columns=['user_provider_key'], inplace=True)
            logger.info(f"Combined dataset size: {len(combined_ratings_df)} (explicit + unique implicit)")
        else:
            # If no implicit data, use only explicit data
            combined_ratings_df = explicit_ratings_df.copy()
            logger.info("No implicit history data found or processed, using only explicit ratings for training.")

        # --- Final check on combined data ---
        if combined_ratings_df.empty:
            logger.error("No training data available after combining explicit and implicit sources. Aborting initialization.")
            return False
        # Ensure score column is numeric
        combined_ratings_df['score'] = pd.to_numeric(combined_ratings_df['score'], errors='coerce')
        combined_ratings_df.dropna(subset=['score', 'userID', 'providerID'], inplace=True)  # Remove rows with invalid core data
        logger.info(f"Final training dataset size after cleaning: {len(combined_ratings_df)}")
        if combined_ratings_df.empty:
            logger.error("Training data is empty after cleaning. Aborting initialization.")
            return False

        # --- Prepare model data ---
        provider_names = providers_df.set_index('providerID')['providerName'].to_dict()
        # Use unique IDs from *combined* dataset to determine model dimensions
        n_users = combined_ratings_df.userID.nunique()
        n_items = combined_ratings_df.providerID.nunique()
        logger.info(f"Model dimensions based on combined data: users={n_users}, providers={n_items}")

        # --- Initialize model ---
        model = MatrixFactorization(n_users, n_items, n_factors=8)
        model.to(device)
        logger.info(f"Model initialized and moved to {device}.")

        # --- Train model (using combined data) ---
        # Reduced training time: from 256 epochs to 64 epochs
        num_epochs = 64 if environment == "production" else 32  # Further reduced for dev environment
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Pass combined dataframe to Loader
        train_set = Loader(combined_ratings_df)
        # Check that the loader correctly created the mappings
        if not train_set.userid2idx or not train_set.providerid2idx:
             logger.error("Failed to create user/provider mappings in Loader. Check combined data.")
             return False

        train_loader = DataLoader(train_set, 128, shuffle=True)

        logger.info(f"Starting model training with combined data ({num_epochs} epochs)...")
        model.train()
        # Training loop (same as before, now uses combined data via train_loader)
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = loss_fn(outputs.squeeze(), y.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                 logger.info(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {epoch_loss/len(train_loader):.4f}")

        logger.info("Model training complete.")
        model.eval()

        # --- Prepare content-based filtering (with full error handling) ---
        logger.info("Preparing data for content-based filtering...")
        
        try:
            # Check providers_df column structure
            logger.info(f"Providers DataFrame shape: {providers_df.shape}")
            logger.info(f"Providers DataFrame columns: {list(providers_df.columns)}")
            
            # Initialize content filtering variables
            cosine_sim = None
            provider_idx = None
            content_columns = []
            
            # Check if genre column or other columns usable for content filtering exist
            if 'genre' in providers_df.columns:
                try:
                    # If genre column exists, use it for content filtering
                    genre_series = providers_df['genre'].dropna()
                    if not genre_series.empty:
                        genres = set()
                        for genre_str in genre_series:
                            if isinstance(genre_str, str) and '|' in genre_str:
                                genres.update(g.strip() for g in genre_str.split('|') if g.strip())
                            elif isinstance(genre_str, str) and genre_str.strip():
                                genres.add(genre_str.strip())
                        
                        if genres:
                            for g in genres: 
                                providers_df[f"genre_{g}"] = providers_df['genre'].transform(
                                    lambda x: int(g in str(x)) if pd.notna(x) else 0
                                )
                                content_columns.append(f"genre_{g}")
                            logger.info(f"Using genre column for content filtering, found {len(genres)} genres.")
                except Exception as e:
                    logger.warning(f"Error processing genre column: {e}")
            
            # Check other possible content feature columns
            potential_columns = ['category', 'type', 'tag', 'subject', 'difficulty', 'level', 'topic', 'field']
            for col in potential_columns:
                if col in providers_df.columns:
                    try:
                        unique_values = providers_df[col].dropna().unique()
                        if len(unique_values) > 0 and len(unique_values) <= 20:  # Limit number of unique values
                            for val in unique_values:
                                if val and str(val).strip():  # Ensure value is not empty
                                    col_name = f"{col}_{str(val).replace(' ', '_')}"
                                    providers_df[col_name] = (providers_df[col] == val).astype(int)
                                    content_columns.append(col_name)
                            logger.info(f"Added features for {col} column with {len(unique_values)} unique values.")
                    except Exception as e:
                        logger.warning(f"Error processing {col} column: {e}")
            
            # If no content features found, use basic provider info as features
            if not content_columns:
                logger.warning("No genre or other content feature columns found, using basic features.")
                try:
                    # Use providerName length as a feature
                    if 'providerName' in providers_df.columns:
                        providers_df['name_length'] = providers_df['providerName'].str.len().fillna(0)
                        providers_df['name_words'] = providers_df['providerName'].str.split().str.len().fillna(0)
                        content_columns.extend(['name_length', 'name_words'])
                    
                    # Use some features from providerID (if numeric)
                    if 'providerID' in providers_df.columns:
                        try:
                            providers_df['id_numeric'] = pd.to_numeric(providers_df['providerID'], errors='coerce').fillna(0)
                            content_columns.append('id_numeric')
                        except:
                            pass
                    
                    logger.info(f"Using basic features: {content_columns}")
                except Exception as e:
                    logger.warning(f"Error creating basic features: {e}")
            
            # Attempt to build content filtering matrix
            if content_columns:
                try:
                    df_providers_features = providers_df[content_columns].fillna(0)
                    # Ensure all features are numeric
                    df_providers_features = df_providers_features.astype(float)
                    
                    # Check that feature matrix is valid
                    if df_providers_features.shape[1] > 0 and df_providers_features.shape[0] > 0:
                        cosine_sim = cosine_similarity(df_providers_features, df_providers_features)
                        provider_idx = dict(zip(providers_df['providerName'], providers_df.index))
                        logger.info(f"✅ Content-based filtering data prepared successfully using {len(content_columns)} features.")
                    else:
                        logger.warning("Feature matrix is empty, cannot perform content-based filtering.")
                        cosine_sim = None
                        provider_idx = None
                except Exception as e:
                    logger.error(f"Content-based filtering preparation failed: {e}")
                    cosine_sim = None
                    provider_idx = None
            
            # If content filtering completely failed, at least ensure provider_idx is available for basic functionality
            if provider_idx is None and 'providerName' in providers_df.columns:
                try:
                    provider_idx = dict(zip(providers_df['providerName'], providers_df.index))
                    logger.info("Created basic provider index; content filtering will be disabled.")
                except Exception as e:
                    logger.warning(f"Failed to create basic provider index: {e}")
            
        except Exception as e:
            logger.error(f"Content filtering initialization completely failed: {e}", exc_info=True)
            cosine_sim = None
            provider_idx = None
            # Ensure collaborative filtering still works even if content filtering fails
            if 'providerName' in providers_df.columns:
                try:
                    provider_idx = dict(zip(providers_df['providerName'], providers_df.index))
                    logger.info("At least created provider index; system can continue running.")
                except:
                    logger.error("Failed to create even the basic index; content filtering will be completely unavailable.")

        # --- Calculate popular items (using explicit ratings only) ---
        # Pass original explicit ratings dataframe 'ratings_df_global'
        popular_items_cache = calculate_popularity(ratings_df_global, providers_df, min_votes=5, top_n=20)

        logger.info("--- Model initialization completed successfully ---")
        return True

    # Keep specific error handling
    except mysql.connector.Error as err:
        logger.error(f"Database connection or query error during initialization: {err}")
        return False
    except Exception as e:
        logger.error(f"General error during model initialization: {e}", exc_info=True)
        # Clear potentially partially-initialized global variables on failure
        model = None
        providers_df = None
        ratings_df_global = None
        popular_items_cache = []
        cosine_sim = None
        provider_idx = None
        train_set = None
        return False


# --- Global environment variable ---
# Check environment variable or file to determine initial environment
import os
def detect_initial_environment():
    """Detect initial environment setting"""
    # Priority: environment variable > file flag > default
    env_from_var = os.getenv('RECOMMENDATION_ENV', '').lower()
    if env_from_var in ['dev', 'development']:
        return 'dev'
    elif env_from_var in ['prod', 'production']:
        return 'production'
    
    # Check if dev flag file exists
    if os.path.exists('.env.dev') or os.path.exists('dev_mode'):
        return 'dev'
    
    return "production"  # Default environment

current_environment = detect_initial_environment()
logger.info(f"🔧 Initial environment set to: {current_environment}")

# --- Scheduled re-initialization ---
def scheduled_reinitialize():
    """Fully re-initialize the model"""
    global last_init_time, init_history, model, provider_names, train_set, cosine_sim, provider_idx, providers_df, ratings_df_global, popular_items_cache, current_environment

    logger.info(f"Starting scheduled model re-initialization (environment: {current_environment})...")
    start_time = datetime.now()

    try:
        # Clean up existing model and data
        model = None
        provider_names = None
        train_set = None
        cosine_sim = None
        provider_idx = None
        providers_df = None
        ratings_df_global = None
        popular_items_cache = []
        logger.info("Old model and data cleared.")

        # Re-initialize using current environment
        success = initialize_model(current_environment)  # Call initialization function

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        status_message = {
            'status': 'success' if success else 'failed', 
            'duration_seconds': duration,
            'environment': current_environment
        }

        if success:
            last_init_time = end_time
            logger.info(f"Model re-initialization completed successfully in {duration:.2f} seconds.")
        else:
            logger.error(f"Model re-initialization failed after {duration:.2f} seconds.")

        # Record initialization history
        init_history.append(
            {'timestamp': end_time.isoformat(), **status_message})
        # Keep approximately the last 24 records
        init_history = init_history[-24:]

    except Exception as e:
        logger.error(f"Error during scheduled re-initialization: {str(e)}", exc_info=True)
        init_history.append({
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e),
            'environment': current_environment
        })
        init_history = init_history[-24:]


# --- Re-initialization loop thread ---
def reinitialization_loop():
    """Run the re-initialization loop, adjusting interval based on environment"""
    global is_running, current_environment
    while is_running:
        # Adjust re-initialization interval based on environment
        # production: 30 minutes (1800 seconds)
        # dev: 10 minutes (600 seconds)
        sleep_duration = 1800 if current_environment == "production" else 600
        logger.info(f"Re-initialization thread sleeping for {sleep_duration} seconds (environment: {current_environment})...")
        time.sleep(sleep_duration)

        if is_running:
            scheduled_reinitialize()


# --- Functions to start/stop the re-initialization thread ---
def start_reinit_thread():
    """Start the re-initialization thread"""
    global reinit_thread, is_running, current_environment
    is_running = True
    if reinit_thread is None or not reinit_thread.is_alive():
        # Use daemon=True so thread exits automatically when main program exits
        reinit_thread = threading.Thread(
            target=reinitialization_loop, daemon=True)
        reinit_thread.start()
        interval = "30 minutes" if current_environment == "production" else "10 minutes"
        logger.info(f"Model re-initialization thread started (interval: {interval}).")


def stop_reinit_thread():
    """Stop the re-initialization thread"""
    global is_running
    is_running = False
    if reinit_thread and reinit_thread.is_alive():
        logger.info("Stopping re-initialization thread... (will exit after current sleep ends)")
        # Daemon threads don't need explicit join
    else:
        logger.info("Re-initialization thread is not running.")


# --- Helper function for content-based recommendations ---
def _get_content_based_recs(search_query, top_n_cb):
    """Generate content-based recommendations based on search query"""
    global provider_idx, cosine_sim, providers_df

    cb_recommendations_data = []
    
    try:
        # Check if required data exists
        if not search_query or search_query.strip() == "":
            logger.info("Search query is empty, skipping content-based filtering.")
            return cb_recommendations_data
            
        if provider_idx is None or providers_df is None:
            logger.info("Provider data unavailable, skipping content-based filtering.")
            return cb_recommendations_data
            
        if cosine_sim is None:
            logger.info("Content similarity matrix unavailable, skipping content-based filtering.")
            return cb_recommendations_data

        logger.info(f"Performing content-based filtering for query: '{search_query}'")
        all_provider_titles = list(provider_idx.keys())  # Get names from mapping used for CBF indexing

        if not all_provider_titles:
            logger.warning("No provider titles available for matching.")
            return cb_recommendations_data

        # Use fuzzywuzzy to find the closest provider name
        try:
            match_result = process.extractOne(search_query, all_provider_titles)
        except Exception as e:
            logger.warning(f"Error during fuzzy matching: {e}")
            return cb_recommendations_data

        # Check match result and match quality (threshold is adjustable)
        if match_result and len(match_result) >= 2 and match_result[1] > 60:  # Lowered threshold to 60
            title = match_result[0]
            logger.info(f"CBF closest match: '{title}' (score: {match_result[1]})")

            # Get the matched name's index in the original providers_df
            matched_provider_df_index = provider_idx.get(title)

            if matched_provider_df_index is not None and matched_provider_df_index < len(cosine_sim):
                try:
                    # Get similarity scores between this provider and all others
                    sim_scores = list(enumerate(cosine_sim[matched_provider_df_index]))

                    # Sort by similarity descending
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

                    # Get Top N most similar providers (starting from index 1, excluding self)
                    sim_scores = sim_scores[1:(top_n_cb) + 1]

                    # Map indices back to providerID/Name and store scores
                    for i, score in sim_scores:
                        try:
                            # Ensure index is within valid range
                            if i < len(providers_df):
                                # 'i' is the index in the original providers_df
                                provider_row = providers_df.iloc[i]
                                provider_id = provider_row.get('providerID')
                                provider_name = provider_row.get('providerName')
                                
                                if provider_id and provider_name:
                                    cb_recommendations_data.append({
                                        'providerID': str(provider_id),
                                        'providerName': str(provider_name),
                                        'cb_score': float(score)  # Use specific score name
                                    })
                        except (IndexError, KeyError, ValueError) as e:
                            logger.warning(f"Error retrieving CBF details for index {i}: {e}")
                            
                    logger.info(f"Content-based filtering returned {len(cb_recommendations_data)} recommendations.")
                            
                except Exception as e:
                    logger.error(f"Error computing content similarity: {e}")
            else:
                logger.warning(f"Matched title '{title}' not found in provider_idx mapping or index is invalid.")
        else:
            logger.info(f"No good content match found for query '{search_query}' (best match score: {match_result[1] if match_result and len(match_result) >= 2 else 'N/A'}).")

    except Exception as e:
        logger.error(f"Error during content-based recommendation: {e}", exc_info=True)

    return cb_recommendations_data


# --- API Route: Get Recommendations ---
@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Recommendations API endpoint, handles logged-in and anonymous users"""
    # Access required global variables
    global model, train_set, provider_names, providers_df, popular_items_cache, cosine_sim, provider_idx, current_environment, init_history

    # Check if core data is loaded
    if providers_df is None or provider_names is None or cosine_sim is None or provider_idx is None:
        logger.error("Recommendation request failed: core data not initialized.")
        return jsonify({'error': 'Service not ready yet, please try again later.'}), 503  # Service unavailable status code

    try:
        data = request.json
        # Get user ID, allow null or empty string (treated as anonymous)
        user_id = data.get('user_id')
        if user_id is not None:
            user_id = str(user_id).strip()
            if not user_id:  # Treat empty string as None (anonymous)
                user_id = None

        search_query = data.get('search_query', '')
        top_n = int(data.get('top_n', 10))
        alpha = float(data.get('alpha', 0.5))  # Weight of CF score in hybrid recommendation
        
        # Get environment parameter
        environment = data.get('environment', 'production')
        if environment not in DB_CONFIGS:
            environment = 'production'  # Default to production
        
        # Auto-detect environment change and switch
        if environment != current_environment:
            logger.info(f"🔄 Environment change detected: auto-switching from {current_environment} to {environment}")
            old_environment = current_environment
            current_environment = environment
            
            # Immediately trigger re-initialization to use new environment database
            logger.info("🚀 Starting automatic model re-training for new environment...")
            start_time = datetime.now()
            
            if initialize_model(current_environment):
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"✅ Environment auto-switch successful! Switched from {old_environment} to {environment}, training took {duration:.2f} seconds")
                
                # Record switch history
                init_history.append({
                    'timestamp': end_time.isoformat(),
                    'status': 'auto_switch_success',
                    'from_environment': old_environment,
                    'to_environment': environment,
                    'duration_seconds': duration
                })
                init_history = init_history[-24:]  # Keep last 24 records
                
            else:
                # Roll back environment setting if initialization fails
                current_environment = old_environment
                logger.error(f"❌ Auto environment switch failed! Rolled back to {old_environment}")
                
                # Record failure history
                init_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'status': 'auto_switch_failed',
                    'attempted_environment': environment,
                    'rolled_back_to': old_environment
                })
                init_history = init_history[-24:]
                
                return jsonify({
                    'error': f'Auto-switch to {environment} environment failed, rolled back to {old_environment}',
                    'current_environment': current_environment
                }), 500

        recommendations = []
        recommendation_type = "unknown"  # Initialize recommendation type

        # --- Determine if user is a valid logged-in user ---
        # Requires model and train_set to be loaded, and user_id to exist in the mapping
        is_logged_in_user = (
            user_id is not None and
            model is not None and
            train_set is not None and
            user_id in train_set.userid2idx
        )

        if is_logged_in_user:
            # --- Logged-in user logic ---
            recommendation_type = "collaborative"  # Default to collaborative filtering
            logger.info(f"Processing recommendation request for logged-in user: {user_id}")

            # --- Collaborative Filtering (CF) section ---
            user_idx = train_set.userid2idx[user_id]
            provider_internal_indices = np.array(
                list(train_set.providerid2idx.values()))  # Model internal indices

            # Prepare tensors for batch prediction
            user_tensor = torch.tensor(
                [user_idx] * len(provider_internal_indices), dtype=torch.long).to(device)
            provider_tensor = torch.tensor(
                provider_internal_indices, dtype=torch.long).to(device)
            logger.info(f"Running CF inference on {device}.")

            # Get model predicted scores
            with torch.no_grad():  # Disable gradient computation during inference
                predicted_ratings = model.forward(torch.stack(
                    (user_tensor, provider_tensor), dim=1)).cpu().numpy()

            # Get internal indices of Top N providers with highest scores
            top_provider_internal_indices = np.argsort(predicted_ratings)[::-1]

            # Map internal indices back to providerID, get names and scores
            cf_recommendations_data = []
            # Get more CF results if hybrid recommendation is possible (e.g. 2*top_n)
            num_cf_to_get = (top_n * 2) if search_query else top_n
            for internal_idx in top_provider_internal_indices:
                original_provider_id = train_set.idx2providerid.get(
                    internal_idx)
                if original_provider_id:  # Ensure mapping exists
                    provider_name = provider_names.get(
                        original_provider_id, 'Unknown Provider')  # Safely get name
                    score = float(predicted_ratings[internal_idx])
                    cf_recommendations_data.append({
                        'providerID': original_provider_id,
                        'providerName': provider_name,
                        'cf_score': score  # Use specific score name
                    })
                    # Stop after getting enough, to optimize performance
                    if len(cf_recommendations_data) >= num_cf_to_get:
                        break

            # --- Content-Based (CB) section (if search query provided) ---
            cb_recommendations_data = []
            if search_query and search_query.strip() != "":
                # Call helper function to get CB recommendations, also get more for merging
                cb_recommendations_data = _get_content_based_recs(
                    search_query, top_n * 2)

            # --- Merge recommendations (hybrid) or finalize CF recommendations ---
            if cb_recommendations_data:  # If CB has results, do hybrid
                recommendation_type = "hybrid"
                logger.info("Merging CF and CB recommendations (hybrid approach).")
                cf_df = pd.DataFrame(cf_recommendations_data)
                cb_df = pd.DataFrame(cb_recommendations_data)

                # Normalize scores (0 to 1) to ensure fair weighting
                scaler = MinMaxScaler()
                if not cf_df.empty and 'cf_score' in cf_df.columns:
                    cf_df['cf_score_norm'] = scaler.fit_transform(
                        cf_df[['cf_score']])
                else:
                    cf_df['cf_score_norm'] = 0  # Handle empty DataFrame case

                if not cb_df.empty and 'cb_score' in cb_df.columns:
                    cb_df['cb_score_norm'] = scaler.fit_transform(
                        cb_df[['cb_score']])
                else:
                    cb_df['cb_score_norm'] = 0  # Handle empty DataFrame case

                # Merge based on providerID (unique identifier)
                hybrid_df = pd.merge(
                    cf_df[['providerID', 'providerName', 'cf_score_norm']],
                    # CB only needs ID and score for merging
                    cb_df[['providerID', 'cb_score_norm']],
                    on='providerID',
                    how='outer'  # Keep all providers from both lists
                )

                # Fill missing values that may appear after merge
                hybrid_df = hybrid_df.fillna({
                    'cf_score_norm': 0,
                    'cb_score_norm': 0,
                })
                # Ensure providerName exists (prefer name from CF)
                # Need original provider_names mapping
                hybrid_df['providerName'] = hybrid_df['providerID'].map(
                    provider_names)
                # Remove row if name cannot be filled for some reason
                hybrid_df = hybrid_df.dropna(subset=['providerName'])

                # Calculate final hybrid score
                hybrid_df['hybrid_score'] = alpha * hybrid_df['cf_score_norm'] + \
                    (1 - alpha) * hybrid_df['cb_score_norm']

                # Sort by hybrid score and take Top N
                hybrid_df = hybrid_df.sort_values(
                    by='hybrid_score', ascending=False).head(top_n)

                # Select final columns and convert to list of dicts
                recommendations = hybrid_df[[
                    'providerID', 'providerName', 'hybrid_score']].to_dict('records')
                logger.info(f"Generated {len(recommendations)} hybrid recommendations.")

            else:  # No CB results, use CF results only
                logger.info("Using collaborative filtering recommendations only (no query or no CB matches).")
                # Take Top N directly from CF results
                cf_only_df = pd.DataFrame(cf_recommendations_data).head(top_n)
                # Rename cf_score to hybrid_score to keep output format consistent
                cf_only_df = cf_only_df.rename(
                    columns={'cf_score': 'hybrid_score'})
                recommendations = cf_only_df[[
                    'providerID', 'providerName', 'hybrid_score']].to_dict('records')
                logger.info(f"Generated {len(recommendations)} CF recommendations.")

        else:
            # --- Anonymous user logic ---
            # Check whether an invalid ID was provided or this is truly an anonymous user
            if user_id is not None:  # User ID was provided but not found in training set
                logger.warning(f"User ID '{user_id}' provided but not found in training data. Treating as anonymous user.")

            logger.info(f"Processing recommendation request for anonymous user.")
            if search_query and search_query.strip() != "":
                # --- Anonymous + search query => CBF only ---
                recommendation_type = "content_based_anonymous"
                logger.info("Anonymous user provided a search query, performing content-based recommendations.")
                cb_recommendations_data = _get_content_based_recs(
                    search_query, top_n)
                # Format CBF results directly
                cb_df = pd.DataFrame(cb_recommendations_data)
                # Rename score for output consistency
                cb_df = cb_df.rename(columns={'cb_score': 'hybrid_score'})
                recommendations = cb_df[[
                    'providerID', 'providerName', 'hybrid_score']].to_dict('records')
                logger.info(f"Generated {len(recommendations)} CBF recommendations for anonymous user.")
            else:
                # --- Anonymous + no search query => Popularity-based ---
                recommendation_type = "popularity_based"
                logger.info("Anonymous user provided no search query, returning popular items.")
                recommendations = []  # Initialize as empty list
                if not popular_items_cache:
                     logger.warning("Popular items cache is empty. Cannot provide popularity-based recommendations.")
                else:
                    # Get cached Top N slice
                    popular_slice = popular_items_cache[:top_n]
                    # Create new list, safely renaming 'popularity_score' to 'hybrid_score'
                    recommendations = []
                    for item in popular_slice:
                        # Safely get values using .get()
                        p_id = item.get('providerID')
                        p_name = item.get('providerName')
                        pop_score = item.get('popularity_score')

                        # Only add to final list if core info is present
                        if p_id and p_name and pop_score is not None:
                            recommendations.append({
                                'providerID': p_id,
                                'providerName': p_name,
                                'hybrid_score': pop_score  # Assign retrieved score to hybrid_score
                            })
                        else:
                            logger.warning(f"Skipping malformed item in popular_items_cache: {item}")

                logger.info(f"Returned {len(recommendations)} popular items for anonymous user.")

        # --- Return results ---
        return jsonify({
            'user_id': user_id,  # Will be null for anonymous users
            'search_query': search_query if search_query else None,  # Return null if empty
            'recommendation_type': recommendation_type,  # Return type of recommendation
            'recommendations': recommendations,
            'environment': current_environment,  # Return currently used environment
            'inference_device': str(device)  # Add inference device info
        })

    except KeyError as e:
        logger.error(f"KeyError during recommendation generation: {e}", exc_info=True)
        return jsonify({'error': f'Missing key or mapping: {e}'}), 500
    except Exception as e:
        logger.error(f"Error fetching recommendations: {e}", exc_info=True)
        return jsonify({'error': 'Internal error generating recommendations.'}), 500


# --- API Route: Health Check ---
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model, last_init_time, init_history, popular_items_cache, providers_df, current_environment
    status = 'healthy' if model is not None and providers_df is not None else 'model_or_data_not_initialized'
    # Re-detect environment to provide more accurate info
    detected_env = detect_initial_environment()
    env_mismatch = detected_env != current_environment
    
    return jsonify({
        'status': status,
        'model_loaded': model is not None,
        'provider_data_loaded': providers_df is not None,
        'current_environment': current_environment,  # Show current running environment
        'detected_environment': detected_env,  # Show detected environment
        'environment_mismatch': env_mismatch,  # Whether there is an environment mismatch
        'available_environments': list(DB_CONFIGS.keys()),  # Show available environments
        'environment_detection': {
            'env_var': os.getenv('RECOMMENDATION_ENV', 'not_set'),
            'dev_file_exists': os.path.exists('.env.dev'),
            'dev_mode_file_exists': os.path.exists('dev_mode')
        },
        'popularity_items_cached': len(popular_items_cache),  # Show number of cached popular items
        'pytorch_device': str(device),  # Show device in use
        'last_init_time': last_init_time.isoformat() if last_init_time else None,  # Last initialization time
        'initialization_history': init_history,  # Show recent initialization attempt records
        'auto_switch_recommendation': 'Recommend calling /api/auto-detect-environment for environment sync' if env_mismatch else None
    })


# --- API Route: Environment Switch ---
@app.route('/api/environment', methods=['POST'])
def switch_environment():
    """Environment switch endpoint"""
    global current_environment
    
    try:
        data = request.json
        target_environment = data.get('environment')
        
        if target_environment not in DB_CONFIGS:
            return jsonify({
                'error': f'Invalid environment parameter: {target_environment}',
                'available_environments': list(DB_CONFIGS.keys())
            }), 400
        
        if target_environment == current_environment:
            return jsonify({
                'message': f'Already in {current_environment} environment',
                'current_environment': current_environment
            })
        
        old_environment = current_environment
        current_environment = target_environment
        
        logger.info(f"Manual environment switch: from {old_environment} to {current_environment}")
        
        # Trigger re-initialization
        if initialize_model(current_environment):
            return jsonify({
                'message': f'Successfully switched to {current_environment} environment',
                'previous_environment': old_environment,
                'current_environment': current_environment
            })
        else:
            # Roll back environment setting if initialization fails
            current_environment = old_environment
            return jsonify({
                'error': f'Failed to switch to {target_environment} environment, rolled back to {old_environment}',
                'current_environment': current_environment
            }), 500
            
    except Exception as e:
        logger.error(f"Error during environment switch: {e}", exc_info=True)
        return jsonify({'error': 'Internal error during environment switch.'}), 500


# --- API Route: Auto environment detection and forced switch ---
@app.route('/api/auto-detect-environment', methods=['POST'])
def auto_detect_environment():
    """Auto-detect environment and switch endpoint"""
    global current_environment
    
    try:
        # Re-detect environment
        detected_environment = detect_initial_environment()
        
        if detected_environment != current_environment:
            logger.info(f"🔍 Environment change auto-detected: switching from {current_environment} to {detected_environment}")
            old_environment = current_environment
            current_environment = detected_environment
            
            # Re-initialize immediately
            if initialize_model(current_environment):
                return jsonify({
                    'message': f'Auto-detected and switched to {current_environment} environment successfully',
                    'previous_environment': old_environment,
                    'current_environment': current_environment,
                    'detection_method': 'auto'
                })
            else:
                # Roll back
                current_environment = old_environment
                return jsonify({
                    'error': f'Auto-switch to {detected_environment} environment failed, rolled back to {old_environment}',
                    'current_environment': current_environment
                }), 500
        else:
            return jsonify({
                'message': f'Environment detection complete, already in {current_environment} environment',
                'current_environment': current_environment
            })
            
    except Exception as e:
        logger.error(f"Error during auto environment detection: {e}", exc_info=True)
        return jsonify({'error': 'Internal error during auto environment detection.'}), 500


# --- API Route: Force dev environment ---
@app.route('/api/force-dev', methods=['POST'])
def force_dev_environment():
    """Force switch to dev environment and retrain"""
    global current_environment
    
    try:
        if current_environment == 'dev':
            return jsonify({
                'message': 'Already in dev environment',
                'current_environment': current_environment
            })
        
        logger.info(f"🔧 Forcing switch to dev environment...")
        old_environment = current_environment
        current_environment = 'dev'
        
        # Re-initialize immediately
        start_time = datetime.now()
        if initialize_model(current_environment):
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ Forced switch to dev environment successful! Training took {duration:.2f} seconds")
            
            return jsonify({
                'message': f'Forced switch to dev environment successful',
                'previous_environment': old_environment,
                'current_environment': current_environment,
                'training_duration_seconds': duration
            })
        else:
            # Roll back
            current_environment = old_environment
            return jsonify({
                'error': f'Forced switch to dev environment failed, rolled back to {old_environment}',
                'current_environment': current_environment
            }), 500
            
    except Exception as e:
        logger.error(f"Error during forced switch to dev environment: {e}", exc_info=True)
        return jsonify({'error': 'Internal error during forced switch to dev environment.'}), 500


# --- Get names of existing history tables ---
def get_existing_history_tables(db_config, lookback_days):
    """Get history tables that actually exist within the query period"""
    existing_tables = []
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        today = datetime.now()
        possible_tables = []
        for i in range(lookback_days):
            date = today - timedelta(days=i)
            table_name = date.strftime("%m%d%Y")
            possible_tables.append(table_name)

        if not possible_tables:
            return []

        # Query information_schema to find which tables exist
        # Use placeholders to prevent SQL injection (if table names are not predictable enough)
        format_strings = ','.join(['%s'] * len(possible_tables))
        query = f"""
            SELECT TABLE_NAME
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME IN ({format_strings})
        """
        params = [db_config['database']] + possible_tables
        cursor.execute(query, tuple(params))
        results = cursor.fetchall()
        existing_tables = [row[0] for row in results]
        logger.info(f"Found history tables within past {lookback_days} days: {existing_tables}")

    except mysql.connector.Error as err:
        logger.error(f"Error checking history tables: {err}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
    return existing_tables

# --- Fetch and process history data ---
def fetch_and_process_history(db_config, lookback_days, min_duration=5):
    """Fetch history data, aggregate and compute implicit scores"""
    history_tables = get_existing_history_tables(db_config, lookback_days)
    if not history_tables:
        logger.warning("No history tables found for the specified time period.")
        return pd.DataFrame(columns=['userID', 'providerID', 'score'])

    all_history_df = pd.DataFrame()
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)  # Get dictionary format

        # Dynamically build UNION ALL query
        select_queries = []
        for table_name in history_tables:
            # Important: use backticks for numeric table names like MMDDYYYY
            select_queries.append(f"SELECT user_id, provider_id, duration FROM `{table_name}` WHERE duration >= {min_duration}")

        full_query = " UNION ALL ".join(select_queries)

        logger.info(f"Fetching history data from {len(history_tables)} tables...")
        cursor.execute(full_query)
        history_data = cursor.fetchall()
        all_history_df = pd.DataFrame(history_data)
        logger.info(f"Fetched {len(all_history_df)} history records (duration >= {min_duration}).")

        if all_history_df.empty:
            return pd.DataFrame(columns=['userID', 'providerID', 'score'])

        # Convert IDs to strings
        all_history_df['user_id'] = all_history_df['user_id'].astype(str)
        all_history_df['provider_id'] = all_history_df['provider_id'].astype(str)

        # --- Calculate implicit scores ---
        # Aggregate data for each user-provider pair (sum duration)
        logger.info("Aggregating history data and calculating implicit scores...")
        user_provider_history = all_history_df.groupby(['user_id', 'provider_id'])['duration'].agg(['sum', 'count']).reset_index()
        user_provider_history = user_provider_history.rename(columns={'sum': 'total_duration', 'count': 'view_count'})

        # Apply log transform to duration (handle duration=0 if min_duration allows it)
        user_provider_history['log_duration'] = np.log1p(user_provider_history['total_duration'])

        # Scale log_duration to desired implicit score range (e.g. 1-4)
        min_log_dur = user_provider_history['log_duration'].min()
        max_log_dur = user_provider_history['log_duration'].max()
        score_min, score_max = IMPLICIT_SCORE_RANGE

        if max_log_dur > min_log_dur:
            user_provider_history['implicit_score'] = score_min + (score_max - score_min) * \
                (user_provider_history['log_duration'] - min_log_dur) / (max_log_dur - min_log_dur)
        else:  # Avoid division by zero if all log durations are the same
            user_provider_history['implicit_score'] = (score_min + score_max) / 2

        # Optional: add small bonus for multiple views (e.g. +0.1 for each view beyond the first)
        # user_provider_history['implicit_score'] += (user_provider_history['view_count'] - 1) * 0.1
        # Clip score to maximum allowed implicit score
        user_provider_history['implicit_score'] = user_provider_history['implicit_score'].clip(upper=score_max)

        # Prepare final implicit scores dataframe
        implicit_ratings_df = user_provider_history[['user_id', 'provider_id', 'implicit_score']].copy()
        implicit_ratings_df.rename(columns={'user_id': 'userID', 'provider_id': 'providerID', 'implicit_score': 'score'}, inplace=True)

        logger.info(f"Calculated implicit scores for {len(implicit_ratings_df)} user-provider interactions.")
        return implicit_ratings_df

    except mysql.connector.Error as err:
        logger.error(f"Error fetching or processing history data: {err}")
        return pd.DataFrame(columns=['userID', 'providerID', 'score'])
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()


# --- Main entry point ---
if __name__ == '__main__':
    logger.info(f"Attempting initial model initialization (default environment: {current_environment})...")
    if initialize_model(current_environment):  # Initialize model with default environment
        start_reinit_thread()  # Start background re-initialization thread if successful
        logger.info(f"Starting Flask server on host 0.0.0.0, port 5100 (debug={True})")
        # debug=True for development, should be set to False in production
        # use_reloader=False prevents initialization from running twice in debug mode
        app.run(host='0.0.0.0', port=5100, debug=True, use_reloader=False)
        # When server stops (e.g. Ctrl+C), attempt to stop background thread
        stop_reinit_thread()
    else:
        logger.error("Initial model initialization failed. Server not started.")
