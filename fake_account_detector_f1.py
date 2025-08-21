
import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
import pandas as pd
import streamlit as st
import re
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import instaloader
import requests
from PIL import Image
import io
import time
import joblib
import os
from groq import Groq # Import Groq client

st.set_page_config(page_title="Instagram Fake Account Detector", layout="wide")
st.title("📸 Instagram Fake Account Detector v2.2 (w/ Groq Agent)")

# --- Configuration & Model Loading (Requires Pre-trained Model) ---
EXPECTED_COLUMNS = [ # MUST match features model was trained with
    "followers_ratio", "followers", "following", "mutual_connections",
    "profile_completion", "has_profile_pic", "is_default_pic",
    # If you retrain adding Groq score, add 'bio_suspicion_score' here
]
MODEL_FILENAME = "xgb_model.json" # Assumes pre-trained model exists

# --- Groq API Client Setup ---
try:
    GROQ_API_KEY = st.secrets["groq"]["api_key"]
    groq_client = Groq(api_key=GROQ_API_KEY)
    GROQ_ENABLED = True
    st.sidebar.success("Groq client initialized.")
except KeyError:
    st.sidebar.error("Groq API key not found in st.secrets.")
    st.error("Add Groq API key to .streamlit/secrets.toml to enable Bio Analysis Agent.")
    st.success("API loaded successfully")
    groq_client = None
    GROQ_ENABLED = False
except Exception as e:
    st.sidebar.error(f"Failed to initialize Groq client: {e}")
    groq_client = None
    GROQ_ENABLED = False


# --- Model Loading Function (Loads pre-trained model) ---
@st.cache_resource
def load_model(filename=MODEL_FILENAME):
    # ... (Keep the load_model function from the previous 'pre-trained' version) ...
    if not os.path.exists(filename):
        st.error(f"Fatal Error: Model file '{filename}' not found.")
        return None
    try:
        model = xgb.XGBClassifier()
        model.load_model(filename)  # Load from .json
        print("XGBoost model loaded from JSON.")
        return model
    except Exception as e:
        st.error(f"Error loading model from '{filename}': {e}")
        return None

xgb_model = load_model() # Load the pre-trained model

# --- Agent Functions ---

# 🔗 Efficient Profile Data Fetching Agent (Cached)
@st.cache_data(ttl=3600) # Cache results for 1 hour
# 📂 Load Dataset
def load_dataset():
    # 📂 Load Dataset
    dataset_path = "C:\\Users\\ANITHA\\Downloads\\fake_id\\fake_id\\datas.csv"
    dataset = pd.read_csv(dataset_path)
    return dataset

dataset = load_dataset()

# 🔍 Ensure expected columns exist
expected_columns = ["followers_ratio", "followers", "following", "mutual_connections", "profile_completion"]
for col in expected_columns:
    if col not in dataset.columns:
        default_value = 1.0 if col == "profile_completion" else 0  # Default for missing columns
        dataset[col] = default_value  # Fill missing column

# 🎯 Define Feature Set
X = dataset[expected_columns]
y = dataset["label"]

# 🏆 Train XGBoost Model
xgb_model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=150, objective='binary:logistic')
xgb_model.fit(X, y)

# 🔥 Define GNN Model
class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = nn.Linear(len(expected_columns), 16)
        self.conv2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x)
        return x

gnn_model = GNN()

# 🎨 Streamlit UI Setup


# =======================================================================
# === NEW AGENT: Groq Bio Analysis ===
# =======================================================================
@st.cache_data(ttl=3600) # Cache LLM analysis results too
def agent_analyze_bio_with_groq(bio_text):
    """
    Analyzes the provided bio text using an LLM via Groq API
    for signs of fakeness, spam, or impersonation.
    Returns a dictionary with 'score' (0-10) and 'justification'.
    """
    default_response = {"score": -1, "justification": "Bio analysis skipped."}
    if not GROQ_ENABLED or not groq_client:
        default_response["justification"] = "Groq client not available."
        return default_response
    if not bio_text or not isinstance(bio_text, str) or len(bio_text.strip()) < 10:
        default_response["justification"] = "Bio is empty or too short for analysis."
        return default_response

    st.info("Analyzing bio with Groq LLM...")
    prompt = f"""
    Analyze the following Instagram bio for signs that the account might be fake, a bot, spam, or an impersonation account.
    Consider factors like: excessive emojis, promotional language, vague descriptions, requests for follows/clicks, claims that seem unrealistic, generic phrasing, or conflicting information.
    Do not evaluate based on personal opinions or beliefs expressed in the bio. Focus only on indicators of a potentially non-genuine account.

    Instagram Bio:
    ---
    {bio_text}
    ---

    Based on your analysis, provide:
    1. A suspicion score from 0 to 10, where 0 means 'very likely genuine' and 10 means 'highly suspicious'.
    2. A brief justification (1-2 sentences) explaining the main reasons for your score.

    Format your response exactly like this:
    Score: [score number]
    Justification: [your justification text]
    """

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI assistant analyzing Instagram bios for potential signs of fake accounts. Provide a score and justification as requested.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192", # Use a capable model available on Groq
            temperature=0.2, # Lower temperature for more deterministic analysis
            max_tokens=1500,
            stop=None, # Let the model finish
        )

        response_content = chat_completion.choices[0].message.content
        st.write("Groq Response:", response_content) # For debugging

        # Parse the response
        score = -1
        justification = "Could not parse LLM response."
        score_match = re.search(r"Score:\s*(\d+)", response_content)
        just_match = re.search(r"Justification:\s*(.*)", response_content, re.DOTALL)

        if score_match:
            score = int(score_match.group(1))
            # Clamp score between 0 and 10
            score = max(0, min(10, score))
        if just_match:
            justification = just_match.group(1).strip()

        return {"score": score, "justification": justification}

    except Exception as e:
        st.error(f"Error contacting Groq API or processing response: {e}")
        default_response["justification"] = f"API/Processing Error: {e}"
        return default_response
# =======================================================================
# === End of Groq Agent ===
# =======================================================================

# Agent 2: Analyze Profile Completion & Basic Features (Using fetched data)
def agent_analyze_profile_completion(profile_data):
    # ... (Keep this function, it calculates features like ratio, completion score) ...
    # Ensure it uses profile_data dictionary correctly
    if not profile_data or not profile_data.get("success"): return None
    features = {}
    followers = profile_data.get('followers', 0)
    following = profile_data.get('following', 0)
    post_count = profile_data.get('post_count', 0)
    pic_url = profile_data.get('profile_pic_url', '')

    features["followers_ratio"] = followers / (following + 1e-6)
    completion_score = 0.0
    if profile_data.get('bio', ''): completion_score += 0.4
    # Basic pic check
    if pic_url and ('cdninstagram.com' in pic_url or 'scontent' in pic_url) and 'profile_pic_anonymous' not in pic_url:
         completion_score += 0.4
    if post_count > 0: completion_score += 0.2
    features["profile_completion"] = completion_score

    features["followers"] = followers
    features["following"] = following
    # Note: mutual_connections is still a placeholder unless fetched
    features["mutual_connections"] = 0.0

    return features


# Agent 3: Image Analysis (Basic Example - Fetches image features)
def agent_analyze_profile_picture(pic_url):
    # ... (Keep this function as defined previously) ...
    features = {"has_profile_pic": 0, "is_default_pic": 1}
    if not pic_url: return features
    try:
        is_likely_default_by_url = "profile_pic_anonymous" in pic_url or "/static/" in pic_url
        try:
            response = requests.get(pic_url, timeout=5, stream=True)
            response.raise_for_status()
            features["has_profile_pic"] = 1
            content_length = response.headers.get('content-length')
            is_small_file = content_length is not None and int(content_length) < 2000
            features["is_default_pic"] = 1 if is_likely_default_by_url or is_small_file else 0
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            features["has_profile_pic"] = 0
            features["is_default_pic"] = 1 if is_likely_default_by_url else 1
    except Exception:
         features = {"has_profile_pic": 0, "is_default_pic": 1}
    return features


# 🔬 Fake Account Prediction (using combined features & loaded model)
def predict_fakeness(features_dict, model):
    # ... (Keep this function, ensure it uses EXPECTED_COLUMNS) ...
    if model is None: return 0.0
    if not features_dict: return 0.0
    try:
        feature_values = [features_dict.get(col, 0) for col in EXPECTED_COLUMNS]
        features_df = pd.DataFrame([feature_values], columns=EXPECTED_COLUMNS)
        xgb_pred_proba = model.predict_proba(features_df)[:, 1]
        final_score = xgb_pred_proba[0] * 100
        return final_score
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return 0.0

# 📡 Network Graph Visualization
def visualize_network_graph(username, followers_count, following_count):
    G = nx.Graph()
    G.add_node(username, size=1000, color='red', label='Target (Red)')

    for i in range(min(followers_count, 50)):
        follower_node = f'Follower_{i}'
        G.add_node(follower_node, size=300, color='blue', label='Follower (Blue)')
        G.add_edge(username, follower_node)

    for i in range(min(following_count, 50)):
        following_node = f'Following_{i}'
        G.add_node(following_node, size=300, color='green', label='Following (Green)')
        G.add_edge(username, following_node)


    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    colors = [G.nodes[n]['color'] for n in G.nodes]
    sizes = [G.nodes[n]['size'] for n in G.nodes]
    nx.draw(G, pos, node_color=colors, node_size=sizes, with_labels=False, edge_color='gray')
    legend_labels = {'Target (Red)': 'red', 'Follower (Blue)': 'blue', 'Following (Green)': 'green'}
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for label, color in legend_labels.items()]
    plt.legend(handles=legend_patches, title="Node Types", loc='upper right',fontsize = 8)

    st.pyplot(plt)


# --- Streamlit UI ---
#st.set_page_config(page_title="Instagram Fake Account Detector", layout="wide")
# ... (sidebar setup remains similar) ...
#st.title("📸 Instagram Fake Account Detector v2.2 (w/ Groq Agent)")
# ... (rest of UI setup) ...
st.sidebar.image("C:\\Users\\ANITHA\\Downloads\\fake_id\\fake_id\\logo.png.webp", width=150) 

# 🔍 Extract Instagram Username from URL
def extract_instagram_username(url):
    instagram_pattern = r"https?://(?:www\.)?instagram\.com/([^/?]+)"
    match = re.search(instagram_pattern, url)
    return match.group(1) if match else None
def fetch_instagram_data(username):
    # ... (Keep the fetch_instagram_data function from the previous step) ...
    loader = instaloader.Instaloader()
    try:
        st.info(f"Fetching live data for @{username} via Instaloader...")
        profile = instaloader.Profile.from_username(loader.context, username)
        return {
                "followers": profile.followers,
                "following": profile.followees,
                "bio": profile.biography,
                "profile_pic_url": profile.profile_pic_url, # Fetch pic url too
                "post_count": profile.mediacount,
                "is_private": profile.is_private,
                "is_verified": profile.is_verified,
                "success": True
               }
    except instaloader.exceptions.QueryReturnedNotFoundException:
         st.error(f"⚠ Profile @{username} not found.")
         return {"success": False, "error": "Profile not found"}
    except instaloader.exceptions.LoginRequiredException:
         st.error(f"⚠ Profile @{username} requires login to view.")
         return {"success": False, "error": "Login required"}
    # ... (other instaloader exceptions) ...
    except Exception as e:
        st.error(f"⚠ Error fetching Instagram data for @{username}: {e}")
        return {"success": False, "error": str(e)}


url = st.text_input("🔗 Enter Instagram Profile URL", key="url_input", placeholder="e.g., https://www.instagram.com/instagram/")

if st.button("🔍 Analyze Profile", key="analyze_button", disabled=(xgb_model is None)):
    if not url:
        st.warning("Please enter an Instagram URL.")
    elif xgb_model is None:
        st.error("Analysis cannot proceed: Pre-trained model not loaded.")
    else:
        username = extract_instagram_username(url)

        if not username:
            st.error("⚠ Invalid Instagram URL format.")
        else:
            st.info(f"Analyzing profile: @{username}")
            progress_bar = st.progress(0, text="Starting analysis...")

            # --- Execute Agents ---
            all_features = {}
            groq_analysis_result = {}

            # Agent 1: Fetch base data (Cached)
            progress_bar.progress(10, text=f"Fetching profile data...")
            profile_data = fetch_instagram_data(username)

            if profile_data and profile_data.get("success"):
                progress_bar.progress(30, text="Analyzing profile structure...")
                basic_features = agent_analyze_profile_completion(profile_data)
                if basic_features: all_features.update(basic_features)

                progress_bar.progress(50, text="Analyzing profile picture...")
                image_features = agent_analyze_profile_picture(profile_data.get('profile_pic_url'))
                if image_features: all_features.update(image_features)

                # Agent X: Analyze Bio with Groq (Cached)
                progress_bar.progress(70, text="Analyzing bio with AI...")
                if GROQ_ENABLED:
                    groq_analysis_result = agent_analyze_bio_with_groq(profile_data.get("bio", ""))
                    # Optional: Add Groq score as feature IF model was trained with it
                    # all_features["bio_suspicion_score"] = groq_analysis_result.get("score", -1) / 10.0 # Normalize if needed
                else:
                    groq_analysis_result = {"score": -1, "justification": "Groq analysis disabled."}


                # --- Predict ---
                progress_bar.progress(80, text="Calculating fakeness score...")
                # IMPORTANT: Current prediction only uses features in EXPECTED_COLUMNS
                # Groq score is displayed separately unless model is retrained
                fake_score = predict_fakeness(all_features, xgb_model)
                progress_bar.progress(90, text="Generating report...")

                # --- Display Results ---
                st.subheader(f"📌 Analysis for: @{username}")
                # (Display profile pic, followers, following, bio as before)
                col1, col2 = st.columns([1, 2])
                with col1:
                    if profile_data.get('profile_pic_url'):
                        st.image(profile_data['profile_pic_url'], width=150, caption="Profile Picture")
                    st.metric("Followers", f"{profile_data.get('followers', 0):,}")
                    st.metric("Following", f"{profile_data.get('following', 0):,}")
                    st.metric("Posts", f"{profile_data.get('post_count', 0):,}")
                with col2:
                    st.write(f"*Bio:*")
                    bio_text = profile_data.get('bio', 'N/A')
                    st.markdown(f"> {bio_text if bio_text else 'No bio provided'}")
                    st.write(f"*Private:* {'Yes' if profile_data.get('is_private', False) else 'No'}")
                    st.write(f"*Verified:* {'Yes' if profile_data.get('is_verified', False) else 'No'}")

                # --- Display Groq Bio Analysis ---
                st.subheader("🤖 AI Bio Analysis (via Groq)")
                if groq_analysis_result and groq_analysis_result.get("score", -1) != -1:
                    g_score = groq_analysis_result["score"]
                    g_just = groq_analysis_result["justification"]
                    st.metric("Bio Suspicion Score (0-10)", f"{g_score}/10")
                    st.caption(f"AI Justification: {g_just}")
                else:
                    st.info(f"Bio analysis skipped: {groq_analysis_result.get('justification', 'Reason unknown.')}")


                st.subheader("📊 Fakeness Assessment")
                # (Display Gauge chart and prediction text as before)
                if fake_score > 75: gauge_color = "red"; result_text = "🚨 HIGH LIKELIHOOD of being FAKE Account"
                elif fake_score > 50: gauge_color = "orange"; result_text = "⚠ Potential FAKE / Suspicious Account (Model)"
                else: gauge_color = "green"; result_text = "✅ Likely a REAL Account"

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=fake_score, title={"text": "Model Fakeness Score (%)"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": gauge_color}, "threshold": {'value': 50}}
                ))
                fig_gauge.update_layout(height=300, margin={'t':60, 'b':10, 'l':30, 'r':30})
                st.plotly_chart(fig_gauge, use_container_width=True)
                # Display result text with appropriate styling
                if fake_score > 75: st.error(f"#### {result_text}")
                elif fake_score > 50: st.warning(f"#### {result_text}")
                else: st.success(f"#### {result_text}")


                # (Display Network Graph as before)
                st.subheader("📡 Simplified Network Representation")
                visualize_network_graph(username, profile_data.get("followers"), profile_data.get("following"))

                progress_bar.progress(100, text="Analysis complete!")
                time.sleep(1)
                progress_bar.empty()

            else: # Handle case where fetch_instagram_data failed
                st.error("Analysis could not be completed because profile data fetching failed.")
                if profile_data and profile_data.get("error"): st.caption(f"Details: {profile_data.get('error')}")
                progress_bar.empty()


# Optional: Cache clearing buttons
if st.sidebar.button("Clear Instagram Data Cache"):
    fetch_instagram_data.clear() # Clear specific function cache
    st.sidebar.success("Instaloader cache cleared!")
if st.sidebar.button("Clear AI Bio Analysis Cache"):
    agent_analyze_bio_with_groq.clear()
    st.sidebar.success("Groq Bio cache cleared!")
