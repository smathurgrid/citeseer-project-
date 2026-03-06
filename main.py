import torch
import numpy as np
from src.data_preprocessing import load_data
from src.model import GCN , APPNPModel , GPRGNN
from src.train import train, test
import argparse
# Creating the CLI for which model to run as per requirements 
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["gcn", "appnp","gpr"],
    help="Choose which model to run: gcn or appnp"
)

args = parser.parse_args()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# -------------------------
# Hyperparameters (GLOBAL)
# -------------------------

hidden_dim = 64
K_value = 5
alpha_value = 0.5
dropout_value = 0.6
weight_decay_value = 5e-4
split_type = "random_80_10_10"   # or "random_60_20_20" or "random_80_10_10"

dataset, data = load_data(device, split_type=split_type)

def run_GCN(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GCN(
        dataset.num_node_features,
        16,
        dataset.num_classes
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=weight_decay_value
    )
    #create lists to store metrics 
    losses=[]
    train_accs=[]
    val_accs=[]

    for epoch in range(1, 301):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)
        losses.append(loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        

        if epoch % 50 == 0:
            
            print(f"Epoch {epoch}")
            print(train_acc, val_acc, test_acc)
    #save matrics to numpy files
    np.save(f"outputs/losses_seed_{seed}.npy", np.array(losses))
    np.save(f"outputs/train_acc_seed_{seed}.npy", np.array(train_accs))
    np.save(f"outputs/val_acc_seed_{seed}.npy", np.array(val_accs))
    
    #storing the weights value after training 
    torch.save(model.state_dict(), f"models/gcn_seed_{seed}.pt")
          

    return test(model, data)[2]
    
def run_APPNP(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = APPNPModel(
        dataset.num_node_features,
        hidden_dim,
        dataset.num_classes,
        K=K_value,
        alpha=alpha_value,
        dropout=dropout_value
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=weight_decay_value
    )

    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(1, 301):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)

        # 🔥 Print accuracy every epoch
        print(f"Seed {seed} | Epoch {epoch} | "
              f"Train: {train_acc:.4f} | "
              f"Val: {val_acc:.4f} | "
              f"Test: {test_acc:.4f}")

        # 🔥 Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc
            }, f"models/appnp_best_seed_{seed}.pt")

            print(f"✅ New Best Model Saved (Seed {seed}) at Epoch {epoch}")

    return best_test_acc
def run_GPR(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GPRGNN(
        dataset.num_node_features,
        hidden_dim,
        dataset.num_classes,
        K=K_value,
        alpha=alpha_value,
        dropout=dropout_value
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=weight_decay_value
    )

    best_val_acc = 0

    for epoch in range(1, 301):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)

        print(f"Seed {seed} | Epoch {epoch} | "
              f"Train: {train_acc:.4f} | "
              f"Val: {val_acc:.4f} | "
              f"Test: {test_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc
            }, f"models/gpr_best_seed_{seed}.pt")

            print(f"✅ New Best GPR Model Saved (Seed {seed})")

    return best_val_acc

#------GCN--------
# gcn_results =[]  

# for seed in seeds:
#     acc = run_GCN(seed)
#     gcn_results.append(acc)

# print("Mean accuracy:", np.mean(gcn_results))

#-------APPNP-------
# appnp_results = []

# for seed in seeds:
#     acc = run_APPNP(seed)
#     appnp_results.append(acc)

# print("APPNP Mean accuracy:", np.mean(appnp_results))
# best_seed_index = np.argmax(appnp_results)
# best_seed = seeds[best_seed_index]

# print("Best Seed:", best_seed)
# print("Best Validation Accuracy:", appnp_results[best_seed_index])



# Seeding with CLI.
seeds = [0, 42, 7, 123, 999]
results = []

for seed in seeds:
    if args.model == "gcn":
        acc = run_GCN(seed)
    elif args.model == "appnp":
        acc = run_APPNP(seed)
    elif args.model == "gpr":
        acc = run_GPR(seed)

    results.append(acc)
mean_acc = np.mean(results)

print(f"\n{args.model.upper()} Mean accuracy:", mean_acc)

best_seed_index = np.argmax(results)
best_seed = seeds[best_seed_index]

print("Best Seed:", best_seed)
print("Best Validation Accuracy:", results[best_seed_index])
# saving best mean accuracy at checkpoint
# import json
# import os

# mean_accuracy = float(np.mean(appnp_results))

# summary_path = "outputs/appnp_best_mean.json"

# # If file already exists → compare with old best
# if os.path.exists(summary_path):
#     with open(summary_path, "r") as f:
#         old_data = json.load(f)

#     old_best = old_data["best_mean_accuracy"]

#     if mean_accuracy > old_best:
#         print("🔥 New BEST Mean Accuracy! Updating file...")

#         new_data = {
#             "best_mean_accuracy": mean_accuracy
#         }

#         with open(summary_path, "w") as f:
#             json.dump(new_data, f, indent=4)

#     else:
#         print("Old best mean accuracy is higher. Keeping previous best.")

# else:
#     print("First experiment. Saving mean accuracy.")

#     new_data = {
#         "best_mean_accuracy": mean_accuracy
#     }

#     with open(summary_path, "w") as f:
#         json.dump(new_data, f, indent=4)

import json
import os
mean_acc = float(np.mean(results))
std_acc = float(np.std(results))

best_seed_index = np.argmax(results)
best_seed = seeds[best_seed_index]

summary = {
    "model": args.model,
    "hidden_dim": hidden_dim,
    "K": K_value,
    "alpha": alpha_value,
    "dropout": dropout_value,
    "weight_decay": weight_decay_value,
    "split_type": split_type,
    "epochs": 300,
    "seeds": seeds,
    "mean_accuracy": mean_acc,
    "std_accuracy": std_acc,
    "best_seed": int(best_seed),
    "best_checkpoint": f"models/{args.model}_best_seed_{best_seed}.pt"
}

summary_path = f"outputs/{args.model}_final_summary.json"

if os.path.exists(summary_path):
    with open(summary_path, "r") as f:
        old_data = json.load(f)

    old_best = old_data["mean_accuracy"]

    if mean_acc > old_best:
        print("🔥 New BEST configuration found! Updating summary file...")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
    else:
        print("Old configuration is better. Keeping previous best.")
else:
    print("First experiment. Saving summary file.")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

print("\nCurrent Mean Accuracy:", mean_acc)
print("Current Std Accuracy:", std_acc)
print("Best Seed in This Run:", best_seed)