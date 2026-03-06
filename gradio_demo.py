import torch
import numpy as np
import json
import gradio as gr
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from src.data_preprocessing import load_data
from src.model import APPNPModel, GPRGNN, GCN
import networkx as nx
import plotly.graph_objects as go

# -----------------------------
# Device
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -----------------------------
# Load Dataset
# -----------------------------
dataset, data = load_data(device, split_type="random_80_10_10")

class_names = ["Agents", "AI", "DB", "IR", "ML", "HCI"]

labels = data.y.cpu().numpy()
counts = np.bincount(labels)

fig_dist = px.bar(
    x=counts,
    y=class_names,
    orientation="h",
    labels={
        "x": "Number of Papers",
        "y": "Research Topics"
    }
)

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

    else:
        config_path = "outputs/gcn_final_summary.json"
        ModelClass = GCN

    with open(config_path, "r") as f:
        config = json.load(f)

    hidden_dim = config["hidden_dim"]
    checkpoint_path = config["best_checkpoint"]

    model = ModelClass(
        dataset.num_node_features,
        hidden_dim,
        dataset.num_classes
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model, config

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
        template="plotly_dark"
    )

    return fig

# -----------------------------
# Prediction Function
# -----------------------------
def analyze_paper(model_name, paper_id):

    model, config = load_model(model_name)

    node_idx = int(paper_id)

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

    fig = build_graph_visualization(node_idx)

    return (
        class_names[pred_class],
        class_names[true_class],
        f"{confidence:.2%}",
        ", ".join(map(str, graph_neighbors)),
        ", ".join(map(str, similar_idx)),
        fig
    )

# -----------------------------
# Gradio Interface
# -----------------------------
with gr.Blocks(title="CiteSeer Graph Neural Network Explorer") as demo:

    gr.Markdown("# CiteSeer Graph Neural Network Explorer")

    with gr.Row():

        with gr.Column(scale=1):

            model_choice = gr.Dropdown(
                ["APPNP", "GPRGNN", "GCN"],
                value="APPNP",
                label="Select Graph Neural Network"
            )

            paper_id = gr.Number(value=0, label="Paper ID")

            analyze_btn = gr.Button("Analyze Paper")

            pred_topic = gr.Textbox(label="Predicted Topic")
            actual_topic = gr.Textbox(label="Actual Topic")
            confidence = gr.Textbox(label="Confidence")

            neighbors = gr.Textbox(label="Citation Neighbors")
            similar = gr.Textbox(label="Similar Papers")

            gr.Plot(fig_dist, label="Dataset Distribution")

        with gr.Column(scale=2):

            graph_plot = gr.Plot(label="Citation Network")

    analyze_btn.click(
        analyze_paper,
        inputs=[model_choice, paper_id],
        outputs=[
            pred_topic,
            actual_topic,
            confidence,
            neighbors,
            similar,
            graph_plot
        ]
    )

demo.launch()