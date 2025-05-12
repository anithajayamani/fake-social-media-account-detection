import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
import pandas as pd
import streamlit as st
import re
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
import instaloader
import requests
from PIL import Image
import io
import time
import joblib
import os
from groq import Groq  # Import Groq client

# --- Page Configuration with Custom Theme ---
st.set_page_config(
    page_title="InstaGuard | Fake Account Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Better UI ---
st.markdown("""
<style>
  /* Main Theme Colors */
  :root {
      --primary: #4F46E5;
      --secondary: #FF5A5F;
      --text-color: #111827;
      --bg-color: #F3F4F6;
      --card-bg: white;
      --success: #10B981;
      --warning: #F59E0B;
      --danger: #EF4444;
  }

  /* Global Colors */
  :root {
    --accent: #F59E0B;
    --bg-dark: #111827;
    --panel-dark: #1F2937;
    --text-light: #FFFFFF;
    --placeholder: #9CA3AF;
    --border-dark: #374151;
    --highlight: #FBBF24;
  }

  /* Animations */
  @keyframes fadeIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
  }
  @keyframes glow {
    0% { text-shadow: 0 0 5px #fbbf24; }
    50% { text-shadow: 0 0 15px #fbbf24; }
    100% { text-shadow: 0 0 5px #fbbf24; }
  }

  /* General Background */
  .stApp,
  .sidebar .sidebar-content {
    background-color: var(--bg-dark) !important;
    color: var(--text-light) !important;
  }

  /* Compact Layout */
  .stContainer {
    display: flex;
    flex-direction: column;
    gap: 10px !important;
  }

  .stCardContainer {
    margin: 0 !important;
    padding: 1rem !important;
    border-radius: 8px !important;
    background-color: var(--panel-dark) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
  }

  /* Fix for Input and TextArea Boxes */
  .stTextInput > div,
  .stTextArea > div {
    background-color: var(--panel-dark) !important;
    color: white !important;
    border: 1px solid var(--border-dark) !important;
    border-radius: 8px;
    padding: 0.5rem !important;
  }

  .stTextInput input {
    color: white !important;
    background-color: var(--panel-dark) !important;
  }

  .stTextInput input::placeholder {
    color: var(--placeholder) !important;
  }
  .stTextInput label {
    color: white !important;
    border: 1px solid var(--bg-dark) !important; /* Debug: Highlight label area */
}
  /* Button Styling */
  .stButton > button {
    background-color: var(--accent) !important;
    color: var(--text-light) !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    box-shadow: 0 4px 8px rgba(245, 158, 11, 0.3);
    transition: all 0.3s ease;
  }

  .stButton > button:hover {
    background-color: #D97706 !important;
    transform: translateY(-2px);
  }

  /* Sidebar Styling */
  .sidebar .sidebar-content a,
  .sidebar .sidebar-content h3 {
    color: var(--highlight) !important;
    font-weight: bold;
    text-decoration: none;
  }

  .sidebar .sidebar-content a:hover {
    background-color: rgba(251, 191, 36, 0.2) !important;
    border-radius: 6px;
  }

  /* Groq Section */
  .groq-enabled {
    background-color: var(--panel-dark) !important;
    color: var(--text-light) !important;
    padding: 10px !important;
    border-radius: 6px !important;
    font-size: 14px;
    text-align: center;
  }

  /* Headings with Animation */
  h1, h2, h3 {
    text-align: left;
    color: var(--accent) !important;
    animation: glow 2s infinite alternate;
  }

  h1 {
    font-size: 2.75rem !important;
    letter-spacing: 0.5px;
    font-weight: 700 !important;
    display: inline-block;
    margin-left: 20px;
  }

  h2 {
    font-size: 2rem !important;
    margin-left: 0;
  }

  h3 {
    font-size: 1.75rem !important;
  }

/* Force tab text color and size */
    div[data-testid="stTabs"] button {
        color: white !important;         /* Make text white */
        font-size: 1.4rem !important;    /* Make it larger */
        font-weight: 700 !important;     /* Bold */
    }

    /* Highlight selected tab with accent color */
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #F59E0B !important;       /* Accent color for active tab */
        text-shadow: 0 0 5px #FBBF24;
    }

  img {
    display: inline-block;
    vertical-align: middle;
  }

  .stMarkdown p {
    text-align: left;
    color: var(--text-light);
    font-size: 1.2rem;
  }

  footer {
    background-color: var(--panel-dark) !important;
    color: var(--placeholder) !important;
    text-align: center;
    padding: 1rem;
    font-size: 12px;
  }

  .stMarkdown {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
  }

  div[data-testid="stVerticalBlock"] div:empty {
    display: none !important;
  }

  .main {
    background-color: var(--bg-color);
    color: var(--text-color);
  }
  [data-testid="stMetricValue"], 
[data-testid="stMetricLabel"] {
  color: white !important;
}
/* Fix for AI Analysis Justification Box */
.ai-analysis-box {
  background-color: #F9FAFB !important; /* dark background */
  color: black !important;           /* black text */
  border-left: 4px solid var(--accent);
  padding: 15px;
  border-radius: 4px;
  margin-top: 10px;
}
.bio-box {
  background-color: var(--panel-dark);
  padding: 10px 15px;
  border-radius: 6px;
  border-left: 3px solid var(--accent);
  margin-bottom: 15px;
}


</style>
""", unsafe_allow_html=True)


# --- App Header ---
col1, col2 = st.columns([1, 9])
with col1:
    st.image("C:\\Users\\ANITHA\\Downloads\\fake_id\\fake_id\\logo.png.webp", width=100)
with col2:
    st.title("InstaGuard: Advanced Fake Account Detector")
    st.markdown("""
    <p style="color: #6B7280; margin-top: -10px;">
        Powered by AI and network analysis to identify suspicious Instagram profiles
    </p>
    """, unsafe_allow_html=True)

# --- Configuration & Model Loading (Requires Pre-trained Model) ---
EXPECTED_COLUMNS = [
    "followers_ratio", "followers", "following", "mutual_connections",
    "profile_completion", "has_profile_pic", "is_default_pic",
]
MODEL_FILENAME = "xgb_model.json"  # Assumes pre-trained model exists

# Fetch Instagram Profile Data
@st.cache_data(ttl=3600)
def fetch_instagram_data(username):
    loader = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        return {
            "followers": profile.followers,
            "following": profile.followees,
            "bio": profile.biography,
            "profile_pic_url": profile.profile_pic_url,
            "post_count": profile.mediacount,
            "is_private": profile.is_private,
            "is_verified": profile.is_verified,
            "success": True
        }
    except instaloader.exceptions.QueryReturnedNotFoundException:
        return {"success": False, "error": "Profile not found"}
    except instaloader.exceptions.LoginRequiredException:
        return {"success": False, "error": "Login required to view this profile"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    
# Analyze Bio with Groq LLM
@st.cache_data(ttl=3600)
def agent_analyze_bio_with_groq(bio_text):
    default_response = {"score": -1, "justification": "Bio analysis skipped."}
    
    if not GROQ_ENABLED or not groq_client:
        default_response["justification"] = "Groq client not available."
        return default_response
        
    if not bio_text or not isinstance(bio_text, str) or len(bio_text.strip()) < 10:
        default_response["justification"] = "Bio is empty or too short for analysis."
        return default_response

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
            model="llama3-8b-8192",
            temperature=0.2,
            max_tokens=1500,
            stop=None,
        )

        response_content = chat_completion.choices[0].message.content

        score = -1
        justification = "Could not parse LLM response."
        
        score_match = re.search(r"Score:\s*(\d+)", response_content)
        just_match = re.search(r"Justification:\s*(.*)", response_content, re.DOTALL)

        if score_match:
            score = int(score_match.group(1))
            score = max(0, min(10, score))
            
        if just_match:
            justification = just_match.group(1).strip()

        return {"score": score, "justification": justification}

    except Exception as e:
        default_response["justification"] = f"API/Processing Error: {str(e)[:50]}..."
        return default_response

# --- Sidebar Setup ---
with st.sidebar:
    # Custom CSS for animations
    st.markdown("""
    <style>
        .sidebar-feature {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        .sidebar-feature:hover {
            background-color: rgba(79, 70, 229, 0.1);
            transform: translateX(5px);
        }
        .feature-title {
            font-weight: bold;
            color: #4F46E5;
            margin-bottom: 5px;
        }
        .feature-description {
            font-size: 0.9em;
            color: #6B7280;
        }
        .sidebar-header {
            text-align: center;
            margin-bottom: 20px;
            animation: fadeIn 1.5s;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Animated header
    st.markdown('<div class="sidebar-header"><h3>📋 App Features</h3></div>', unsafe_allow_html=True)
    
    # Feature list with hover animations
    st.markdown("""
    <div class="sidebar-feature">
        <div class="feature-title">Text Analysis</div>
        <div class="feature-description">Process and analyze text content</div>
    </div>
    
    <div class="sidebar-feature">
        <div class="feature-title">Entity Recognition</div>
        <div class="feature-description">Find important biological terms</div>
    </div>
    
    <div class="sidebar-feature">
        <div class="feature-title">Visualization</div>
        <div class="feature-description">See relationships in visual format</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a divider
    st.divider()
    
    # --- Groq API Client Setup ---
    st.markdown("### 🤖 Groq LLM Bio Analysis")
    try:
        GROQ_API_KEY = st.secrets["groq"]["api_key"]
        groq_client = Groq(api_key=GROQ_API_KEY)
        GROQ_ENABLED = True
        st.success("✅ Groq AI analysis enabled")
    except KeyError:
        st.error("❌ Groq API key not found")
        st.info("Add Groq API key to enable AI Bio Analysis")
        groq_client = None
        GROQ_ENABLED = False
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq client: {str(e)[:50]}...")
        groq_client = None
        GROQ_ENABLED = False
    
    # Cache clearing buttons with improved styling
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Clear Data Cache", key="clear_data_cache"):
            fetch_instagram_data.clear()
            st.success("✅ Cache cleared!")
    with col2:
        if st.button("🔄 Clear AI Cache", key="clear_ai_cache"):
            agent_analyze_bio_with_groq.clear()
            st.success("✅ Cache cleared!")

# --- Model Loading Function (Loads pre-trained model) ---
@st.cache_resource
def load_model(filename=MODEL_FILENAME):
    if not os.path.exists(filename):
        st.error(f"⚠️ Model file '{filename}' not found.")
        return None
    try:
        model = xgb.XGBClassifier()
        model.load_model(filename)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

xgb_model = load_model()  # Load the pre-trained model

# --- Data Loading Function ---
@st.cache_data(ttl=3600)
def load_dataset():
    dataset_path = "C:\\Users\\ANITHA\\Downloads\\fake_id\\fake_id\\datas.csv"
    dataset = pd.read_csv(dataset_path)
    return dataset

dataset = load_dataset()

# Ensure expected columns exist
expected_columns = ["followers_ratio", "followers", "following", "mutual_connections", "profile_completion"]
for col in expected_columns:
    if col not in dataset.columns:
        default_value = 1.0 if col == "profile_completion" else 0
        dataset[col] = default_value

# Define Feature Set
X = dataset[expected_columns]
y = dataset["label"]

# Train XGBoost Model
xgb_model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=150, objective='binary:logistic')
xgb_model.fit(X, y)

# Define GNN Model
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

# --- Agent Functions ---

# Extract Instagram Username from URL
def extract_instagram_username(url):
    instagram_pattern = r"https?://(?:www\.)?instagram\.com/([^/?]+)"
    match = re.search(instagram_pattern, url)
    return match.group(1) if match else None


# Analyze Profile Completion
def agent_analyze_profile_completion(profile_data):
    if not profile_data or not profile_data.get("success"): 
        return None
    
    features = {}
    followers = profile_data.get('followers', 0)
    following = profile_data.get('following', 0)
    post_count = profile_data.get('post_count', 0)
    pic_url = profile_data.get('profile_pic_url', '')

    features["followers_ratio"] = followers / (following + 1e-6)
    
    completion_score = 0.0
    if profile_data.get('bio', ''): 
        completion_score += 0.4
    
    if pic_url and ('cdninstagram.com' in pic_url or 'scontent' in pic_url) and 'profile_pic_anonymous' not in pic_url:
        completion_score += 0.4
        
    if post_count > 0: 
        completion_score += 0.2
        
    features["profile_completion"] = completion_score
    features["followers"] = followers
    features["following"] = following
    features["mutual_connections"] = 0.0

    return features

# Analyze Profile Picture
def agent_analyze_profile_picture(pic_url):
    features = {"has_profile_pic": 0, "is_default_pic": 1}
    if not pic_url: 
        return features
    
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


# Fake Account Prediction
def predict_fakeness(features_dict, model):
    if model is None: 
        return 0.0
    if not features_dict: 
        return 0.0
    
    try:
        feature_values = [features_dict.get(col, 0) for col in EXPECTED_COLUMNS]
        features_df = pd.DataFrame([feature_values], columns=EXPECTED_COLUMNS)
        xgb_pred_proba = model.predict_proba(features_df)[:, 1]
        final_score = xgb_pred_proba[0] * 100
        return final_score
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        return 0.0

# Network Graph Visualization with Improved Styling
def visualize_network_graph(username, followers_count, following_count):
    G = nx.Graph()
    G.add_node(username, size=800, color='red', label='Target')

    # Limit displayed connections for clarity
    max_connections = min(30, max(followers_count, following_count))
    
    for i in range(min(followers_count, max_connections)):
        follower_node = f'Follower_{i}'
        G.add_node(follower_node, size=200, color='blue', label='Follower')
        G.add_edge(username, follower_node)

    for i in range(min(following_count, max_connections)):
        following_node = f'Following_{i}'
        G.add_node(following_node, size=200, color='green', label='Following')
        G.add_edge(username, following_node)

    plt.figure(figsize=(10, 6), facecolor='#F9FAFB')
    pos = nx.spring_layout(G, k=0.3)
    
    node_colors = [G.nodes[n]['color'] for n in G.nodes]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes]
    
    nx.draw(
        G, pos, 
        node_color=node_colors, 
        node_size=node_sizes, 
        with_labels=False, 
        edge_color='#E5E7EB',
        width=1.5,
        alpha=0.9
    )
    
    # Add custom legend
    legend_labels = {
        'Target Profile': 'red', 
        'Followers': 'blue', 
        'Following': 'green'
    }
    legend_patches = [plt.Line2D(
        [0], [0], 
        marker='o', 
        color='w', 
        markerfacecolor=color, 
        markersize=10, 
        label=label
    ) for label, color in legend_labels.items()]
    
    plt.legend(
        handles=legend_patches, 
        title="Network Nodes", 
        loc='upper right',
        fontsize=9,
        frameon=True,
        facecolor='white',
        edgecolor='#E5E7EB'
    )
    
    plt.tight_layout()
    
    return plt.gcf()

# --- Main Analysis UI ---
st.markdown("### Enter Instagram Profile to Analyze")

with st.container():
    st.markdown("""
    <div class="stCardContainer">
        <p>Enter an Instagram profile URL below to analyze for signs of a fake account</p>
    """, unsafe_allow_html=True)
    
    url = st.text_input(
        "🔗 Instagram Profile URL", 
        key="url_input", 
        placeholder="https://www.instagram.com/username/",
        help="Enter the full URL of the Instagram profile you want to analyze"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_button = st.button(
            "🔍 Analyze Profile", 
            key="analyze_button", 
            disabled=(xgb_model is None),
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Analysis Process
if analyze_button:
    if not url:
        st.warning("⚠️ Please enter an Instagram URL to analyze.")
    elif xgb_model is None:
        st.error("⚠️ Analysis cannot proceed: Pre-trained model not loaded.")
    else:
        username = extract_instagram_username(url)

        if not username:
            st.error("⚠️ Invalid Instagram URL format. Please enter a valid profile URL.")
        else:
            st.markdown(f"""
            <div class="stCardContainer">
                <h3>Analyzing @{username}</h3>
            """, unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # --- Execute Analysis Pipeline ---
            all_features = {}
            groq_analysis_result = {}
            
            # Step 1: Fetch profile data
            status_text.text("📡 Connecting to Instagram...")
            progress_bar.progress(10)
            profile_data = fetch_instagram_data(username)
            
            if profile_data and profile_data.get("success"):
                # Step 2: Analyze profile structure
                status_text.text("🔍 Analyzing profile structure...")
                progress_bar.progress(30)
                basic_features = agent_analyze_profile_completion(profile_data)
                if basic_features: 
                    all_features.update(basic_features)
                
                # Step 3: Analyze profile picture
                status_text.text("🖼️ Analyzing profile picture...")
                progress_bar.progress(50)
                image_features = agent_analyze_profile_picture(profile_data.get('profile_pic_url'))
                if image_features: 
                    all_features.update(image_features)
                
                # Step 4: Analyze bio with Groq LLM
                status_text.text("🤖 Analyzing bio with AI...")
                progress_bar.progress(70)
                if GROQ_ENABLED:
                    groq_analysis_result = agent_analyze_bio_with_groq(profile_data.get("bio", ""))
                else:
                    groq_analysis_result = {"score": -1, "justification": "Groq analysis disabled."}
                
                # Step 5: Calculate fakeness score
                status_text.text("🧮 Calculating risk scores...")
                progress_bar.progress(90)
                fake_score = predict_fakeness(all_features, xgb_model)
                
                # Step 6: Complete
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                time.sleep(1)
                
                # Clear progress elements
                progress_bar.empty()
                status_text.empty()
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # --- Results Dashboard ---
                
                # Profile Overview Tab
                tab1, tab2, tab3 = st.tabs(["📊 Profile Assessment", "🔍 Analysis Details", "📡 Network Analysis"])
                
                with tab1:
                    # Profile Header
                    st.markdown("""
                    <div class="stCardContainer">
                        <h3>📌 Profile Overview</h3>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if profile_data.get('profile_pic_url'):
                            st.image(
                                profile_data['profile_pic_url'], 
                                width=150,
                                caption=f"@{username}"
                            )
                        else:
                            st.markdown("*No profile picture available*")
                    
                    with col2:
                        # Profile badges
                        badge_html = ""
                        if profile_data.get('is_verified', False):
                            badge_html += '<span class="verified-badge">✓ Verified</span> '
                        if profile_data.get('is_private', False):
                            badge_html += '<span class="private-badge">🔒 Private</span>'
                        
                        st.markdown(f"""
                        <h3 style="margin-bottom: 5px;">@{username}</h3>
                        {badge_html}
                        """, unsafe_allow_html=True)
                        
                        # Stats row
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        with metrics_col1:
                            st.metric("Followers", f"{profile_data.get('followers', 0):,}")
                        with metrics_col2:
                            st.metric("Following", f"{profile_data.get('following', 0):,}")
                        with metrics_col3:
                            st.metric("Posts", f"{profile_data.get('post_count', 0):,}")
                            
                        # Bio box
                        bio_text = profile_data.get('bio', '')
                        if bio_text:
                            st.markdown("""
                            <div class="bio-box">
                                <p style="margin-bottom: 5px; font-weight: 500;">Bio:</p>
                                <p style="white-space: pre-line;">{}</p>
                            </div>
                            """.format(bio_text), unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="bio-box"><p><em>No bio provided</em></p></div>', unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Fakeness Score Display
                    st.markdown("""
                    <div class="stCardContainer">
                        <h3>🛑 Fakeness Assessment</h3>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Create a custom gauge chart with better styling
                        if fake_score > 75:
                            gauge_color = "#EF4444"  # Red
                            result_text = "🚨 HIGH RISK - Likely FAKE Account"
                            result_class = "result-badge-high"
                        elif fake_score > 50:
                            gauge_color = "#F59E0B"  # Orange/Amber
                            result_text = "⚠️ MEDIUM RISK - Potentially Suspicious"
                            result_class = "result-badge-medium"
                        else:
                            gauge_color = "#10B981"  # Green
                            result_text = "✅ LOW RISK - Likely Authentic"
                            result_class = "result-badge-low"
                        
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=fake_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={
                                "text": "Fakeness Score",
                                "font": {"size": 24, "color": "#111827"}
                            },
                            gauge={
                                "axis": {
                                    "range": [0, 100],
                                    "tickwidth": 1,
                                    "tickcolor": "#6B7280"
                                },
                                "bar": {"color": gauge_color},
                                "bgcolor": "#F3F4F6",
                                "borderwidth": 2,
                                "bordercolor": "#E5E7EB",
                                "threshold": {
                                    "line": {"color": "#6B7280", "width": 4},
                                    "thickness": 0.75,
                                    "value": 50
                                }
                            },
                            number={
                                "suffix": "%",
                                "font": {"size": 26, "color": gauge_color, "family": "Arial"}
                            }
                        ))
                        
                        fig_gauge.update_layout(
                            height=400,
                            margin={'t': 40, 'b': 40, 'l': 40, 'r': 40},
                            paper_bgcolor="#FFFFFF",
                            plot_bgcolor="#FFFFFF"
                        )
                        
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                        # Display result badge
                        st.markdown(f"""
                        <div class="{result_class}">
                            <h4 style="margin: 0;">{result_text}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Key risk factors visualization
                        st.markdown("#### Key Risk Factors")
                        
                        # Calculate factor contributions
                        risk_factors = {
                            "Profile Completion": (1 - all_features.get('profile_completion', 0)) * 100,
                            "Followers/Following": (1 - min(1, all_features.get('followers_ratio', 0) / 5)) * 100,
                            "Profile Picture": all_features.get('is_default_pic', 1) * 100
                        }
                        
                        # Add Bio analysis if available
                        if groq_analysis_result and groq_analysis_result.get("score", -1) >= 0:
                            risk_factors["Bio Content"] = groq_analysis_result.get("score", 0) * 10
                        
                        # Create horizontal bar chart
                        risk_df = pd.DataFrame({
                            'Factor': list(risk_factors.keys()),
                            'Risk Level': list(risk_factors.values())
                        })
                        
                        fig = px.bar(
                            risk_df,
                            x='Risk Level',
                            y='Factor',
                            orientation='h',
                            color='Risk Level',
                            color_continuous_scale=['#10B981', '#FBBF24', '#EF4444'],
                            range_color=[0, 100],
                            text_auto='.0f'
                        )
                        
                        fig.update_traces(
                            texttemplate='%{text}%',
                            textposition='outside'
                        )
                        
                        fig.update_layout(
                            height=250,
                            margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
                            xaxis_title=None,
                            yaxis_title=None,
                            coloraxis_showscale=False,
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tab2:
                    # Detailed Analysis Tab
                    st.markdown("""
                    <div class="stCardContainer">
                        <h3>🔍 Detailed Analysis</h3>
                    """, unsafe_allow_html=True)
                    
                    # Analysis Timeline
                    
                    st.markdown("#### Feature Analysis")
                    
                    # Key metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        follower_ratio = all_features.get('followers_ratio', 0)
                        st.metric(
                            "Followers/Following Ratio", 
                            f"{follower_ratio:.2f}",
                            delta=f"{'Suspicious' if follower_ratio < 0.5 else 'Normal'}"
                        )
                        
                    with col2:
                        completion = all_features.get('profile_completion', 0) * 100
                        st.metric(
                            "Profile Completion", 
                            f"{completion:.0f}%",
                            delta=f"{'Incomplete' if completion < 50 else 'Complete'}"
                        )
                        
                    with col3:
                        has_pic = all_features.get('has_profile_pic', 0)
                        is_default = all_features.get('is_default_pic', 1)
                        
                        if has_pic and not is_default:
                            st.metric("Profile Picture", "Custom", delta="Normal")
                        elif has_pic and is_default:
                            st.metric("Profile Picture", "Default", delta="Suspicious")
                        else:
                            st.metric("Profile Picture", "None", delta="Suspicious")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # AI Bio Analysis
                    st.markdown("""
                    <div class="stCardContainer">
                        <h3>🤖 AI Bio Analysis</h3>
                    """, unsafe_allow_html=True)
                    
                    if groq_analysis_result and groq_analysis_result.get("score", -1) >= 0:
                        g_score = groq_analysis_result["score"]
                        g_just = groq_analysis_result["justification"]
                        
                        # Create score pill with conditional colors
                        if g_score >= 7:
                            score_color = "#EF4444"  # Red
                        elif g_score >= 4:
                            score_color = "#F59E0B"  # Orange
                        else:
                            score_color = "#10B981"  # Green
                            
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <div style="background-color: {score_color}; color: white; border-radius: 999px; padding: 3px 15px; font-weight: bold; margin-right: 10px;">
                                {g_score}/10
                            </div>
                            <span>Bio Suspicion Score</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                                <div style="
                                    background-color: #F9FAFB !important; 
                                    border-left: 4px solid {score_color}; 
                                    padding: 15px; 
                                    border-radius: 4px; 
                                    color: black !important;
                                    font-size: 1rem !important;
                                ">
                                    <p style="font-weight: 500; margin-bottom: 5px; color: black !important;">AI Analysis:</p>
                                    <p style="color: black !important;">{g_just}</p>
                                </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        st.info(f"Bio analysis not available: {groq_analysis_result.get('justification', 'Unknown reason')}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Account Recommendations
                    st.markdown("""
                    <div class="stCardContainer">
                        <h3>💡 Recommendations</h3>
                    """, unsafe_allow_html=True)
                    
                    if fake_score > 75:
                        st.markdown("""
                        <div class="danger-card">
                            <p><strong>⚠️ High Risk Account Detected</strong></p>
                            <p>This account shows multiple high-risk factors consistent with fake accounts. Exercise extreme caution.</p>
                            <ul>
                                <li>Do not share personal information with this account</li>
                                <li>Avoid clicking on any links sent by this account</li>
                                <li>Consider reporting this account to Instagram</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    elif fake_score > 50:
                        st.markdown("""
                        <div class="warning-card">
                            <p><strong>⚠️ Proceed with Caution</strong></p>
                            <p>This account shows some suspicious patterns. Verify identity before engaging.</p>
                            <ul>
                                <li>Look for verification through mutual connections</li>
                                <li>Check for consistent posting history</li>
                                <li>Be cautious with any requests or links</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="success-card">
                            <p><strong>✅ Low Risk Profile</strong></p>
                            <p>This account appears to be authentic based on our analysis.</p>
                            <ul>
                                <li>Normal engagement patterns detected</li>
                                <li>Profile shows typical authentic characteristics</li>
                                <li>Still follow standard social media safety practices</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tab3:
                    # Network Analysis Tab
                    st.markdown("""
                    <div class="stCardContainer">
                        <h3>📡 Network Visualization</h3>
                        <p>Simplified visual representation of the account's connection patterns</p>
                    """, unsafe_allow_html=True)
                    
                    # Create network graph
                    network_fig = visualize_network_graph(
                        username,
                        profile_data.get("followers", 0),
                        profile_data.get("following", 0)
                    )
                    
                    st.pyplot(network_fig)
                    
                    # Network metrics
                    st.markdown("#### Network Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        follower_count = profile_data.get("followers", 0)
                        st.metric("Total Followers", f"{follower_count:,}")
                        
                    with col2:
                        following_count = profile_data.get("following", 0)
                        st.metric("Total Following", f"{following_count:,}")
                        
                    with col3:
                        ratio = follower_count / max(following_count, 1)
                        st.metric("Network Ratio", f"{ratio:.2f}")
                    
                    # Network analysis explanation
                    if follower_count > following_count * 2:
                        network_type = "Broadcaster"
                        network_desc = "This account has significantly more followers than accounts they follow, typical of public figures, brands, or content creators."
                    elif following_count > follower_count * 2:
                        network_type = "Consumer"
                        network_desc = "This account follows many more accounts than follow them. This can be normal for new accounts but may indicate follow-farming behavior."
                    else:
                        network_type = "Balanced"
                        network_desc = "This account has a relatively balanced ratio of followers to following, common for regular personal accounts."
                    
                    st.markdown(f"""
                        <div style="
                            background-color: #F9FAFB !important;
                            color: black !important;
                            padding: 15px;
                            border-radius: 8px;
                            margin-top: 20px;
                            font-size: 1rem;
                        ">
                            <p style="font-weight: 600; margin-bottom: 5px; color: black !important;">Network Type: {network_type}</p>
                            <p style="color: black !important;">{network_desc}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Advanced Network Insights
                    st.markdown("""
                    <div class="stCardContainer">
                        <h3>📊 Engagement Analysis</h3>
                    """, unsafe_allow_html=True)
                    
                    # Simulated engagement metrics
                    post_count = profile_data.get("post_count", 0)
                    
                    if post_count > 0:
                        # Simulate engagement metrics for demonstration
                        avg_likes = min(int(follower_count * 0.05 + 10), follower_count)
                        avg_comments = min(int(avg_likes * 0.1 + 2), avg_likes)
                        
                        engagement_rate = (avg_likes + avg_comments) / max(follower_count, 1) * 100
                        
                        # Create engagement visualization
                        engagement_data = {
                            'Metric': ['Avg. Likes', 'Avg. Comments'],
                            'Value': [avg_likes, avg_comments]
                        }
                        
                        fig = px.bar(
                            engagement_data,
                            x='Metric',
                            y='Value',
                            color='Metric',
                            color_discrete_map={
                                'Avg. Likes': '#4F46E5',
                                'Avg. Comments': '#60A5FA'
                            },
                            text_auto=True
                        )
                        
                        fig.update_layout(
                            height=300,
                            showlegend=False,
                            xaxis_title=None,
                            yaxis_title='Count',
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display engagement rate
                        if engagement_rate < 1:
                            rate_assessment = "Very Low"
                            rate_color = "#EF4444"
                        elif engagement_rate < 3:
                            rate_assessment = "Low"
                            rate_color = "#F59E0B"
                        elif engagement_rate < 6:
                            rate_assessment = "Average"
                            rate_color = "#10B981"
                        else:
                            rate_assessment = "High"
                            rate_color = "#10B981"
                        
                        st.markdown(f"""
                            <div style="
                                background-color: #F9FAFB !important;
                                color: black !important;
                                border-left: 4px solid {rate_color};
                                padding: 15px;
                                border-radius: 4px;
                            ">
                                <p style="font-weight: 600; margin-bottom: 5px; color: black !important;">
                                    Estimated Engagement Rate: {engagement_rate:.2f}%
                                </p>
                                <p style="color: black !important;">
                                    Assessment: <span style="color: {rate_color}; font-weight: 500;">{rate_assessment}</span>
                                </p>
                                <p style="font-size: 0.9rem; color: black !important;">
                                    Note: These are estimated metrics based on account statistics.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                    else:
                        st.info("No posts found to analyze engagement metrics.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            else:  # Handle case where fetch_instagram_data failed
                st.error(f"""
                ⚠️ Could not analyze profile. 
                Error: {profile_data.get("error", "Unknown error occurred")}
                """)
                progress_bar.empty()

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; border-top: 1px solid #E5E7EB;">
    <p style="color: #6B7280; font-size: 0.9rem;">
        InstaGuard: Fake Account Detection Tool v2.2 | Powered by AI & Network Analysis
    </p>
    <p style="color: #6B7280; font-size: 0.8rem;">
        This tool is for educational purposes only. Always verify accounts through official channels.
    </p>
</div>
""", unsafe_allow_html=True)