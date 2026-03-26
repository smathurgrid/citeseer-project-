import sys
import os
# Use to handle file path 
# This tells Python to look one folder up for the 'src' module,
# allowing you to run this seamlessly from inside the task_3 folder.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
#torch → ML
#time → measure speed
#pandas → table
#matplotlib → plotting
#profiler → find bottlenecks

# Import your existing pipeline
from src.data_preprocessing import load_data
from src.model import GCN, APPNPModel, GPRGNN
from src.train import train, test
#load_data → dataset
#train → training
#test → accuracy
# 1. Setup Device and Data
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
#Using Apple GPU 

# We use the 80/10/10 split as our standard for this test
dataset, data = load_data(device, split_type="random_80_10_10")
hidden_dim = 64
epochs = 100 # Kept at 100 for faster profiling, adjust if needed

def get_current_mps_memory():# memory function returns how many GPU memory is currently used 
    """Fetches the current memory allocated by PyTorch on the MPS device in MB."""
    if device.type == "mps":
        # Returns current allocated memory in bytes, convert to Megabytes (MB)
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return 0.0
# This is the main function and the most important one 
# This function does everything -
    # train , measure time , measure memory , measure accuracy , measure inference , run profiler 
def profile_model(model_name, model, K_val="N/A"):
    """Runs training, measures time, memory, inference latency, and runs PyTorch Profiler."""
    print(f"\n--- Profiling {model_name} (K={K_val}) ---")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #standrad training optimizer 
        
    # 1. Reset the MPS memory tracker before training so we get a clean measurement
    if device.type == "mps":
        torch.mps.empty_cache()
    
    # 2. Track Training Time and Peak Memory
    start_train = time.time()
    peak_memory = 0

    for epoch in range(1, epochs + 1):
        train(model, data, optimizer)
        
        # Manually track the peak memory using current_allocated_memory()
        current_mem = get_current_mps_memory()
        if current_mem > peak_memory:
            peak_memory = current_mem

    train_time = time.time() - start_train
    time_per_epoch = (train_time / epochs) * 1000 # convert to milliseconds
    
    # 3. Get Accuracy
    train_acc, val_acc, test_acc = test(model, data)
    
    # 4. Track Inference Latency (How fast it predicts)
    model.eval()
    start_infer = time.time()
    with torch.no_grad():
        for _ in range(50): # Run 50 times to get a stable average
            _ = model(data.x, data.edge_index)
    infer_latency = ((time.time() - start_infer) / 50) * 1000 # convert to ms
    
    # 5. Run PyTorch Profiler to find bottlenecks
    print(f"Running deep PyTorch profiling for {model_name}...")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(data.x, data.edge_index)
            
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=3))
    
    return {
        "Model": f"{model_name} (K={K_val})" if K_val != "N/A" else model_name,
        "Base_Model": model_name,
        "K": K_val,
        "Test Accuracy": test_acc,
        "Train Time/Epoch (ms)": time_per_epoch,
        "Inference Latency (ms)": infer_latency,
        "Peak Memory (MB)": peak_memory
    }

# 3. Define the Sweep
results = []
k_sweep = [5, 10, 20, 40]

# Run GCN (No K value)
gcn_model = GCN(dataset.num_node_features, hidden_dim, dataset.num_classes).to(device)
results.append(profile_model("GCN", gcn_model))

# Run APPNP and GPR-GNN Sweeps
for k in k_sweep:
    appnp_model = APPNPModel(dataset.num_node_features, hidden_dim, dataset.num_classes, K=k, alpha=0.1, dropout=0.5).to(device)
    results.append(profile_model("APPNP", appnp_model, K_val=k))
    
    gpr_model = GPRGNN(dataset.num_node_features, hidden_dim, dataset.num_classes, K=k, alpha=0.1, dropout=0.5).to(device)
    results.append(profile_model("GPR-GNN", gpr_model, K_val=k))

# 4. Save and Print Results
df_results = pd.DataFrame(results)
print("\n=== FINAL PROFILING RESULTS ===")
print(df_results.to_string(index=False))

# Make sure outputs folder exists before saving csv
# creating output files and if not exisitn=ing creating a new one 
os.makedirs("../outputs", exist_ok=True)
df_results.to_csv("../outputs/profiling_results.csv", index=False)

# 5. Plot the Pareto Frontier (Latency vs Accuracy)
plt.figure(figsize=(10, 6))

colors = {"GCN": "blue", "APPNP": "red", "GPR-GNN": "green"}

for idx, row in df_results.iterrows():
    plt.scatter(row["Inference Latency (ms)"], row["Test Accuracy"], 
                color=colors[row["Base_Model"]], s=100, alpha=0.7)
    plt.annotate(row["Model"], 
                 (row["Inference Latency (ms)"], row["Test Accuracy"]),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.title("Pareto Frontier: Inference Latency vs Test Accuracy")
plt.xlabel("Inference Latency (ms) - Lower is Better")
plt.ylabel("Test Accuracy - Higher is Better")
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
# Note: Saving to ../plots/ since the script runs from inside task_3
os.makedirs("../plots", exist_ok=True)
plt.savefig("../plots/pareto_frontier.png", dpi=300)
print("\n✅ Pareto plot saved to ../plots/pareto_frontier.png")