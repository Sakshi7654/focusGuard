import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from fuzzy_engine import evaluate_behavior

# 1. Prepare Feature Sequences
print("[1/4] Loading master dataset...")
df = pd.read_csv("data/master_custom_dataset.csv")

print("[2/4] Generating fuzzy soft-state probabilities...")
fuzzy_records = [evaluate_behavior(row.to_dict()) for _, row in df.iterrows()]
fuzzy_df = pd.DataFrame(fuzzy_records)

# Use fuzzy probabilities as inputs to temporal LSTM[cite: 2, 4, 18]
features = fuzzy_df[["focus_score", "distraction_score", "uncertain_score"]].values

# Extract ground-truth labels if labeled; otherwise determine dominant state[cite: 2, 17]
if "label" in df.columns:
    labels = df["label"].values
else:
    labels = np.argmax(features, axis=1)

# Sequence Generation: Window size = 6 steps (30 seconds)[cite: 11, 17]
SEQ_LEN = 6
X_seq, y_seq = [], []
for i in range(len(features) - SEQ_LEN):
    X_seq.append(features[i:i + SEQ_LEN])
    y_seq.append(labels[i + SEQ_LEN])

X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
y_tensor = torch.tensor(np.array(y_seq), dtype=torch.long)

class SequenceData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

loader = DataLoader(SequenceData(X_tensor, y_tensor), batch_size=16, shuffle=True)

# 2. Define LSTM Architecture[cite: 2, 5]
class FocusLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_classes=3):
        super(FocusLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

print("[3/4] Training Temporal LSTM Network...")
model = FocusLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train across 15 epochs
for epoch in range(1, 16):
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:02d}/15 | Loss: {total_loss / len(loader):.4f}")

# 3. Save Model Weights
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/temporal_lstm.pth")
print("[4/4] [SUCCESS] Trained weights saved to 'models/temporal_lstm.pth'!")