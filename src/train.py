import torch
import torch.nn.functional as F
#This function makes the model learn 
def train(model, data, optimizer):
    #tells pytorch we are in training mode 
    # Dropouts are on and model are allowed to change the weight 
    model.train()
    # This clears old gradients 
    # Pytorch stores gradients from prev steps 
    # We reset them before new update
    optimizer.zero_grad()
    #Forward Pass 
    # Most important line 
    # Models sees ALL 3327 Papers 
    # Model sees ALL citation edges 
    # Model predicts class scores for every paper 
    # Soo outpt shape is [3327,6]
    # Each row one paper 
    # Each Coumn one class score 
    out = model(data.x, data.edge_index)
    #Compute Loss 
    #Looking at only training nodes 
    #compare predicted labels with true labels 
    #Compute error 

    # Even though model predicts for all nodes loss is calcualted only for training loss 
    # THis is how solit works 
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # Backpropagation 
    # How should weight change to reduce error 
    loss.backward()
    # Update weights 
    optimizer.step()
    #Retrun Loss value for printing 
    return loss.item()

# Only Checkss the performance 

def test(model, data):
    # This tell pytorch to off the dropout and no randomness
    # Stable predictictions 
    model.eval()
    #Model predicts for all papers 
    out = model(data.x, data.edge_index)
    #Pick class with highest score
    pred = out.argmax(dim=1)

    accs = []
    #Accuracy calcualtion 
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]

        acc = int(correct.sum()) / int(mask.sum())
        accs.append(acc)

    return accs
# ## SUMMARY ##
# 1. Take Full graph 
# 2. Predict class for every nodes 
# 3. Compute loss only on train nodes 
# 4. Update weights
# 5. Repeat for 300 Epochs 
# 6. After training -> check val and test accuracy 