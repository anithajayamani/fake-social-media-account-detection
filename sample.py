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

# 📂 Load Dataset
dataset_path = "C:\\Users\\ANITHA\\Downloads\\fake_id\\fake_id\\datas.csv"
dataset = pd.read_csv(dataset_path)

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
st.set_page_config(page_title="Instagram Fake Account Detector", layout="wide")
st.sidebar.image("C:\\Users\\ANITHA\\Downloads\\fake_id\\fake_id\\logo.png.webp", width=150) 
st.title("📸 Instagram Fake Account Detector")

# 🔍 Extract Instagram Username from URL
def extract_instagram_username(url):
    instagram_pattern = r"https?://(?:www\.)?instagram\.com/([^/?]+)"
    match = re.search(instagram_pattern, url)
    return match.group(1) if match else None

# 🔗 Fetch Instagram Profile Data
def fetch_instagram_data(username):
    loader = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        return profile.followers, profile.followees, profile.biography
    except Exception as e:
        st.error(f"⚠️ Error fetching Instagram data: {e}")
        return None, None, None

# 🔬 Fake Account Prediction using XGBoost & GNN
def predict_fakeness(followers_ratio, followers, following, mutual_connections, profile_completion):
    xgb_features = np.array([[followers_ratio, followers, following, mutual_connections, profile_completion]])
    xgb_pred = xgb_model.predict_proba(xgb_features)[:, 1][0]

    gnn_features = torch.tensor([[followers_ratio, followers, following, mutual_connections, profile_completion]], dtype=torch.float)
    gnn_pred = gnn_model(gnn_features).softmax(dim=1)[:, 1].item()

    final_score = (xgb_pred + gnn_pred) / 2 * 100
    return final_score

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

# 🎯 Streamlit Input
url = st.text_input("🔗 Enter Instagram Profile URL")

if st.button("🔍 Fetch Data & Predict"):
    username = extract_instagram_username(url)

    if username:
        followers_count, following_count, bio_text = fetch_instagram_data(username)

        if followers_count is not None and following_count is not None:
            followers_ratio = followers_count / (following_count + 1)  # Avoid division by zero
            fake_score = predict_fakeness(followers_ratio, followers_count, following_count, 0.5, 1.0)

            st.subheader(f"📌 Instagram User: @{username}")
            st.write(f"**Followers:** {followers_count}")
            st.write(f"**Following:** {following_count}")
            st.write(f"**Bio:** {bio_text}")

            # 📈 Display Fakeness Score
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fake_score,
                title={"text": "Fakeness Score"},
                gauge = {
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "red" if fake_score > 50 else "green"}
}

            ))
            st.plotly_chart(fig)

            # ✅ Display Final Result
            result_text = "🚨 FAKE ACCOUNT DETECTED" if fake_score > 50 else "✅ REAL ACCOUNT"
            st.success(f"### 🔍 Prediction: {result_text}")

            # 📡 Network Graph
            st.subheader("📡 Network Graph Visualization")
            visualize_network_graph(username, followers_count, following_count)
        else:
            st.error("⚠️ Failed to fetch Instagram profile data.")
    else:
        st.error("⚠️ Invalid Instagram URL. Please check the format.")