import torch
import numpy as np
import json
import streamlit as st
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from src.data_preprocessing import load_data
from src.model import APPNPModel, GPRGNN, GCN
import networkx as nx
import plotly.graph_objects as go

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="CiteSeer Graph Neural Network Explorer",
    layout="wide"
)

st.title("CiteSeer Graph Neural Network Explorer")

st.markdown("""
This demo shows how **Graph Neural Networks** classify research papers  
in the **CiteSeer citation network**.

• Each **paper is a node**  
• Each **citation is a connection (edge)**  
• The model predicts the **topic of a paper**

Select a paper ID to explore its neighborhood and prediction.
""")

# -----------------------------
# Device
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -----------------------------
# Load Dataset
# -----------------------------
dataset, data = load_data(device, split_type="random_80_10_10")

class_names = ["Agents", "AI", "DB", "IR", "ML", "HCI"]
# Compute class distribution
labels = data.y.cpu().numpy()
counts = np.bincount(labels)
fig_dist = px.bar(
    x=counts,
    y=class_names,
    orientation='h',
    # title="Class Distribution"
    labels={
        "x": "Number of Papers",
        "y": "Research Topics"
    }
)

fig_dist.update_layout(
    template="plotly_dark",
    height=250,
    margin=dict(l=10, r=10, t=40, b=10)
)

# -----------------------------
# Dataset Overview
# -----------------------------
st.markdown("### Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Number of Papers", data.num_nodes)
c2.metric("Citation Links", data.edge_index.shape[1])
c3.metric("Research Topics", dataset.num_classes)

# -----------------------------
# Model Loader
# -----------------------------
def load_model(model_name):

    if model_name == "APPNP":
        config_path = "outputs/appnp_final_summary.json"
        ModelClass = APPNPModel

    elif model_name == "GPRGNN":
        config_path = "outputs/gpr_final_summary.json"
        ModelClass = GPRGNN

    elif model_name == "GCN":
        config_path = "outputs/gcn_final_summary.json"
        ModelClass = GCN

    with open(config_path, "r") as f:
        config = json.load(f)

    hidden_dim = config["hidden_dim"]
    dropout = config["dropout"]
    checkpoint_path = config["best_checkpoint"]

    model = ModelClass(
        dataset.num_node_features,
        hidden_dim,
        dataset.num_classes
    ).to(device)

    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])

    # model.eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)

# Handle both checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model, config

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:

    st.header("Model Selection")

    model_choice = st.selectbox(
        "Select Graph Neural Network",
        ["APPNP", "GPRGNN", "GCN"]
    )
    st.markdown("### Dataset Class Distribution")
    st.plotly_chart(fig_dist, use_container_width=True)

    model, config = load_model(model_choice)

    st.metric("Model Accuracy", f"{config['mean_accuracy']:.4f}")

    st.markdown("---")

    st.header("Topic Color Legend")

    COLOR_MAP = {
        "Agents": "#1f77b4",
        "AI": "#2ca02c",
        "DB": "#ff7f0e",
        "IR": "#9467bd",
        "ML": "#8c564b",
        "HCI": "#FFD700"
    }

    with st.expander("Show Topic Colors"):

        for topic, color in COLOR_MAP.items():

            st.markdown(
                f"""
                <div style="display:flex;align-items:center;margin-bottom:5px;">
                <div style="width:15px;height:15px;background:{color};border-radius:50%;margin-right:10px;"></div>
                {topic}
                </div>
                """,
                unsafe_allow_html=True
            )

# -----------------------------
# Feature Similarity
# -----------------------------
features = data.x.cpu().numpy()
sim_matrix = cosine_similarity(features)

# -----------------------------
# Graph Visualization
# -----------------------------
def build_graph_visualization(node_idx):

    node_idx = int(node_idx)

    G = nx.Graph()
    G.add_node(node_idx)

    edge_index = data.edge_index.cpu().numpy()
    neighbors = edge_index[1][edge_index[0] == node_idx]

    for n in neighbors:
        G.add_node(int(n))
        G.add_edge(node_idx, int(n))

    pos = nx.spring_layout(G, dim=3, seed=42)

    class_colors = {
        0: "#1f77b4",
        1: "#2ca02c",
        2: "#ff7f0e",
        3: "#9467bd",
        4: "#8c564b",
        5: "#FFD700"
    }

    node_traces = []

    for class_id, color in class_colors.items():

        x_vals, y_vals, z_vals = [], [], []
        labels = []

        for node in G.nodes():

            if data.y[node].item() == class_id:

                x, y, z = pos[node]

                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(z)

                labels.append(str(node))

        if x_vals:
            node_traces.append(
                go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    marker=dict(size=8, color=color),
                    name=class_names[class_id]
                )
            )

    x, y, z = pos[node_idx]

    selected_trace = go.Scatter3d(
        x=[x],
        y=[y],
        z=[z],
        mode="markers+text",
        text=[str(node_idx)],
        textposition="top center",
        marker=dict(size=18, color="red"),
        name="Selected Paper"
    )

    edge_x, edge_y, edge_z = [], [], []

    for edge in G.edges():

        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(width=2, color="gray"),
        showlegend=False
    )

    fig = go.Figure(data=[edge_trace] + node_traces + [selected_trace])

    fig.update_layout(
        title="Citation Network Visualization",
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )

    return fig

# -----------------------------
# Prediction
# -----------------------------
def predict(node_idx):

    node_idx = int(node_idx)

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.exp(out)

    pred_class = probs[node_idx].argmax().item()
    confidence = probs[node_idx][pred_class].item()
    true_class = data.y[node_idx].item()

    edge_index = data.edge_index.cpu().numpy()
    neighbors = edge_index[1][edge_index[0] == node_idx]

    graph_neighbors = neighbors[:5].tolist()

    similar_idx = np.argsort(-sim_matrix[node_idx])[1:6].tolist()

    return pred_class, true_class, confidence, graph_neighbors, similar_idx

# -----------------------------
# Layout
# -----------------------------
col_graph, col_control = st.columns([2,1])

with col_control:

    st.subheader("Explore Papers")

    paper_id = st.number_input(
        "Enter Paper ID",
        min_value=0,
        max_value=data.num_nodes - 1,
        value=0
    )

    if st.button("Analyze Paper"):

        pred, true, conf, neighbors, similar = predict(paper_id)

        st.markdown("### Model Prediction")

        c1, c2, c3 = st.columns(3)

        c1.metric("Predicted Topic", class_names[pred])
        c2.metric("Actual Topic", class_names[true])
        c3.metric("Confidence", f"{conf:.2%}")

        st.markdown("### Citation Neighbors")
        neighbor_text = " • ".join(str(n) for n in neighbors)
        st.markdown(f"**{neighbor_text}**")

        st.markdown("### Papers With Similar Content")
        st.caption("These papers discuss similar research topics based on their text content.")
        similar_text = " • ".join(str(s) for s in similar)
        st.markdown(f"**{similar_text}**")

with col_graph:

    st.markdown("### Citation Network Visualization")

    fig = build_graph_visualization(paper_id)

    st.plotly_chart(fig, use_container_width=True)