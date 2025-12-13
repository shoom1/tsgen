import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (Batch, Seq, Feat)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

def train_and_evaluate_tstr(synthetic_data, real_data, epochs=10, device='cpu'):
    """
    Train on Synthetic, Test on Real (TSTR) Metric.
    
    Task: Predict the NEXT step return given the previous steps.
    
    Args:
        synthetic_data: (N_syn, Seq_Len, Features)
        real_data: (N_real, Seq_Len, Features)
    
    Returns:
        float: Mean Squared Error on Real Data.
    """
    # Task setup: Input = X[t:t+L-1], Target = X[t+L-1] (Next step prediction roughly, 
    # or rather, we use the sequence to predict the FINAL value's future or something sim.
    # To keep it simple and matching the data shape:
    # Input:  (Batch, Seq_Len-1, Features)
    # Target: (Batch, Features) - The last return in the sequence
    
    seq_len = synthetic_data.shape[1]
    features = synthetic_data.shape[2]
    
    if seq_len < 2:
        return 0.0 # Cannot do prediction
        
    # Prepare Datasets
    def prepare_xy(data):
        # X: First L-1 steps
        # y: Last step
        X = torch.FloatTensor(data[:, :-1, :])
        y = torch.FloatTensor(data[:, -1, :])
        return X, y
    
    X_syn, y_syn = prepare_xy(synthetic_data)
    X_real, y_real = prepare_xy(real_data)
    
    # Train on Synthetic
    train_ds = TensorDataset(X_syn, y_syn)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    model = SimpleLSTM(input_dim=features, output_dim=features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
    # Test on Real
    model.eval()
    with torch.no_grad():
        real_pred = model(X_real.to(device))
        tstr_mse = criterion(real_pred, y_real.to(device)).item()
        
    return tstr_mse
