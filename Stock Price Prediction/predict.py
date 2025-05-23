import torch
import pandas as pd
import numpy as np
from train_model import PatchTST, load_data
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
def load_model():
    model = PatchTST(input_dim=1, hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load("patchtst_stock_model.pth"))
    model.eval()
    return model

# Predict future prices
def predict_future(days=10):
    close_prices, scaler = load_data()
    model = load_model()

    # Use last 30 days as input
    last_sequence = close_prices[-30:].reshape(1, -1, 1)  # Shape: (1, 30, 1)
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32)

    predictions = []
    for _ in range(days):
        with torch.no_grad():
            pred = model(last_sequence).item()
        predictions.append(pred)

        # Update sequence
        last_sequence = torch.cat((last_sequence[:, 1:, :], torch.tensor([[[pred]]], dtype=torch.float32)), dim=1)

    # Convert predictions back to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    
    return predictions

if __name__ == "__main__":
    future_prices = predict_future()
    print("📈 Future Stock Prices:", future_prices)

