import asyncio
import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Import your custom modules
from src.data_preprocessing import load_data
from src.model import APPNPModel

# ==========================================
# 1. SETUP: The Restaurant Cashier
# ==========================================
app = FastAPI(title="CiteSeer APPNP Inference Server")

# Global variables for our Model (Chef) and Data
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = None
data = None

# Batching variables (The Bus Stop)
BATCH_TIMEOUT = 0.100  # 100 milliseconds
queue = asyncio.Queue()

# Define what the incoming JSON request should look like
class PredictRequest(BaseModel):
    node_ids: List[int]

# ==========================================
# 2. STARTUP: Prepping the Kitchen
# ==========================================
@app.on_event("startup")
async def startup_event():
    global model, data
    print("⏳ Starting server and loading model into memory...")
    
    # Load CiteSeer data
    dataset, data = load_data(device, split_type="random_80_10_10")
    
    # Initialize the APPNP Model
    model = APPNPModel(
        dataset.num_node_features, 64, dataset.num_classes,
        K=5, alpha=0.5, dropout=0.6
    ).to(device)
    
    # Load your best trained weights
    checkpoint = torch.load("models/appnp_best_seed_42.pt", map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Start the background "Bus Stop" worker
    asyncio.create_task(batch_processor())
    print("✅ Model loaded! Server is ready to take orders.")

# ==========================================
# 3. THE BUS STOP: 100ms Dynamic Batching
# ==========================================
async def batch_processor():
    while True:
        batch = []
        
        # Wait until at least ONE customer arrives
        req = await queue.get()
        batch.append(req)
        
        # Start a 100ms timer to let other customers join the line
        start_time = time.time()
        while time.time() - start_time < BATCH_TIMEOUT:
            try:
                # Check if anyone else gets in line during this 100ms
                next_req = await asyncio.wait_for(queue.get(), timeout=0.01)
                batch.append(next_req)
            except asyncio.TimeoutError:
                # If no one joins, just keep looping until 100ms is up
                continue
                
        # Time is up! Send the whole batch to the Chef
        if batch:
            await process_batch(batch)

async def process_batch(batch):
    # Track Mac MPS Memory before processing
    if torch.backends.mps.is_available():
        mem_before = torch.mps.current_allocated_memory() / (1024**2) # Convert to MB
    
    # THE CHEF: Run the model once for everyone
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        predictions = out.argmax(dim=1)
        
    # Track Mac MPS Memory after processing to see the spike
    if torch.backends.mps.is_available():
        mem_after = torch.mps.current_allocated_memory() / (1024**2)
        print(f"🚀 Processed batch of {len(batch)} requests | 🧠 MPS Memory Spike: {mem_after - mem_before:.2f} MB")

    # Hand the correct predictions back to each specific customer
    for req in batch:
        node_ids = req['node_ids']
        future = req['future']
        
        # Grab only the answers this specific customer asked for
        res = [predictions[n].item() for n in node_ids]
        future.set_result(res)  # This triggers the endpoint to return the response

# ==========================================
# 4. THE ENDPOINT: Taking the Order
# ==========================================
@app.post("/predict")
async def predict(request: PredictRequest):
    # Create an empty "ticket" to hold the result when it's done
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    
    # Put the customer's order and their ticket in the queue
    await queue.put({
        'node_ids': request.node_ids,
        'future': future
    })
    
    # Wait here until the batch processor finishes and fills the ticket
    result = await future
    
    # Return the final JSON to the customer!
    return {"predictions": result}