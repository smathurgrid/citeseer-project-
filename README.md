# 📚 CiteSeer Graph Neural Network Explorer

An interactive **Graph Neural Network (GNN) dashboard** for exploring and classifying research papers in the **CiteSeer citation network** using multiple GNN architectures.

This project compares **GCN, APPNP, and GPRGNN** models for node classification and provides a **Streamlit-based visualization interface** to explore the citation graph and predictions.

---

# 🚀 Project Overview

In citation networks:

- Each **paper is represented as a node**
- Each **citation is represented as an edge**
- Each node contains **text features describing the paper**

The goal is to **predict the research topic of each paper** using Graph Neural Networks.

This project:

- Trains multiple GNN architectures
- Compares their performance
- Visualizes citation relationships
- Provides an interactive UI for exploring predictions

---

# 🧠 Models Implemented

### 1️⃣ GCN (Graph Convolutional Network)

- Classic Graph Neural Network
- Aggregates information from neighboring nodes
- Two-layer propagation

### 2️⃣ APPNP (Approximate Personalized Propagation of Neural Predictions)

- Separates feature transformation and propagation
- Allows long-range propagation across the graph
- Achieves strong performance on citation networks

### 3️⃣ GPRGNN (Generalized PageRank Graph Neural Network)

- Learns optimal propagation weights
- Flexible propagation mechanism
- Captures different neighborhood ranges

---

# 📊 Dataset

Dataset used: **CiteSeer Citation Network**

| Metric | Value |
|------|------|
| Number of Papers | 3327 |
| Citation Links | 9104 |
| Research Topics | 6 |
| Node Features | 3703 |

Research categories:

- Agents
- AI
- DB
- IR
- ML
- HCI

---

# 📈 Experimental Results

| Model | Mean Accuracy |
|------|------|
| APPNP | ~0.80 |
| GPRGNN | ~0.77 |
| GCN | ~0.75 |

APPNP achieved the **best performance** due to its improved propagation mechanism.

---

# 🏗 Project Architecture

```
Dataset
   │
   ▼
Data Preprocessing
(src/data_preprocessing.py)
   │
   ▼
Graph Neural Network Models
(GCN / APPNP / GPRGNN)
   │
   ▼
Model Training + Evaluation
   │
   ▼
Saved Checkpoints + JSON Configs
(models/ and outputs/)
   │
   ▼
Streamlit Interactive Dashboard
(streamlit_demo.py)
```

# 📁 Project Structure

```
citeseer_project/
│
├── models/
│   ├── appnp_best_seed_42.pt
│   ├── gcn_seed_42.pt
│   ├── gpr_best_seed_999.pt
│
├── outputs/
│   ├── appnp_final_summary.json
│   ├── gcn_final_summary.json
│   ├── gpr_final_summary.json
│
├── notebooks/
│   ├── Experiments.ipynb
│   ├── Visualization.ipynb
│
├── src/
│   ├── model.py
│   ├── data_preprocessing.py
│   ├── train.py
│
├── streamlit_demo.py
│
└── README.md
```

# 🖥 Interactive Dashboard

The project includes a **Streamlit application** for interactive exploration.

Features:

### 🔹 Model Selection
Users can switch between:

- APPNP
- GCN
- GPRGNN

### 🔹 Paper Explorer
Users can enter a **paper ID** to inspect:

- predicted topic
- actual topic
- prediction confidence

### 🔹 Citation Graph Visualization
Interactive **3D citation network visualization** showing:

- selected paper
- citation neighbors
- topic clusters

### 🔹 Similar Papers
Displays papers with **similar textual features** using cosine similarity.

### 🔹 Dataset Insights
Sidebar includes **class distribution visualization**.

---

# ⚙️ Installation
How to Install Brew 
```bash 
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

```

Install Python 3.11
```bash 
brew install python@3.11
```

Clone the repository:

```bash
git clone <repo-link>
cd citeseer_project
```

Create virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch Geometric compiled libraries 

```bash 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
```
### Train Models via CLI
You can train different Graph Neural Network models using the command line interface provided in `main.py`.
```bash
python main.py --model appnp
python main.py --model gcn
python main.py --model gpr
```

## ▶️ Running the Streamlit Application

To launch the interactive dashboard locally, run:

```bash
streamlit run streamlit_demo.py
```
or 
```bash 
python -m streamlit run streamlit_demo.py
```


---


## 🧩 Technologies Used

The project was implemented using the following tools and libraries:

- **Python** – Core programming language
- **PyTorch** – Deep learning framework
- **PyTorch Geometric** – Graph Neural Network library
- **Streamlit** – Interactive dashboard for visualization
- **Plotly** – Interactive graphs and charts
- **NetworkX** – Graph structure handling and visualization
- **NumPy** – Numerical computations
- **Scikit-learn** – Feature similarity computation

## 🚀 Future Improvements

Possible extensions for this project include:

- Implement additional Graph Neural Network models such as **Graph Attention Networks (GAT)**
- Add **node embedding visualization** using t-SNE or UMAP
- Enable **search functionality for papers by topic or keyword**
- Deploy the dashboard publicly using **Streamlit Cloud or Hugging Face Spaces**
- Integrate **dynamic graph exploration with adjustable neighborhood depth**

## 👨‍💻 Author

**Siddhant Mathur**

Data Science Intern 
Hyderabad Office 

This project was developed as part of a **Graph Neural Network exploration and visualization study on the CiteSeer citation dataset**.