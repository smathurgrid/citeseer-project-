# Project 3: Productionizing Graph Neural Networks (GNNs)

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-View_Report-2563eb?style=for-the-badge)](https://smathurgrid.github.io/blog_post/)
*(Click the badge above to view the interactive engineering report)*

## 📌 Project Overview
In modern data ecosystems, information does not exist in isolation. Predicting the topic of a research paper requires a profound mathematical understanding of the complex web of citations connecting the ecosystem. 

This project is an end-to-end Machine Learning Engineering case study focused on the **CiteSeer network dataset**. Utilizing Graph Neural Networks—specifically the APPNP architecture—this project documents the complete lifecycle of deploying a graph model: from theoretical hardware profiling and latent space debugging to deploying a high-concurrency asynchronous inference API.

## 🛠️ Tech Stack
* **Deep Learning Framework:** PyTorch, PyTorch Geometric (PyG)
* **Hardware Acceleration:** Apple Silicon (MPS)
* **API & Deployment:** FastAPI, Asyncio, Uvicorn
* **Data Visualization:** Plotly, t-SNE, Matplotlib
* **Frontend Dashboard:** HTML5, CSS3, Vanilla JavaScript

## 🚀 Engineering Phases (Tasks)
* **Phase 1: Architectural Profiling:** Benchmarked GCN, APPNP, and GPR-GNN models. Established a Pareto frontier to identify **APPNP (K=5)** as the optimal low-latency baseline.
* **Phase 2: Activation Patching:** Applied layer-wise edge ablation to mathematically prove how cross-topic citations act as structural noise, corrupting information flow.
* **Phase 3: Ablation Study:** Executed a 240-model hyperparameter grid search. Identified a regularized baseline that sustains a peak accuracy of **78.05%** even under severe data starvation conditions.
* **Phase 4: Latent Manifold Analysis:** Bypassed terminal layers to extract 64-dimensional internal representations, applying **t-SNE dimensionality reduction** to visually map the model's logical decision boundaries.
* **Phase 5: High-Concurrency API:** Engineered an asynchronous `asyncio` waiting-room protocol to facilitate dynamic tensor batching. Sustained **1,378+ Requests Per Second (QPS)** under extreme load testing while preventing GPU Out-Of-Memory (OOM) failures.

## 🌐 Viewing the Dashboard
To view the full, interactive engineering report, navigate to the **[Live Dashboard](https://[YOUR-USERNAME].github.io/[YOUR-REPO-NAME]/)**.
<!-- ### 🚀 [Project 3: Productionizing Graph Neural Networks & Async APIs](https://smathurgrid.github.io/blog_post/)
**Technologies:** PyTorch, FastAPI, Asyncio, Plotly, HTML/CSS/JS

An end-to-end Machine Learning Engineering case study detailing the deployment of a Graph Neural Network (APPNP) on the CiteSeer dataset. This project goes beyond model training to tackle real-world deployment challenges. I conducted deep hardware profiling, visual latent space analysis (t-SNE), and designed a custom asynchronous batching queue in FastAPI. The final enterprise-grade REST API successfully decoupled client requests from GPU constraints, sustaining **1,378+ QPS** during stress testing with zero VRAM exhaustion.

👉 **[View the Interactive Engineering Dashboard Here](https://smathurgrid.github.io/blog_post/)** -->