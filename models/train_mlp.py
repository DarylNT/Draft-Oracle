import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # silence loky Windows warning - adjust to system capabilities

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)

class MatchDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(prefix):
    train = torch.load(f"splits/{prefix}_train.pt")
    valid = torch.load(f"splits/{prefix}_valid.pt")
    test = torch.load(f"splits/{prefix}_test.pt")
    return train, valid, test

def train(model_type, model, train_loader, valid_loader, epochs, lr, weight_decay):
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    best_valid_loss = float('inf')
    ctr = 0
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            train_logits = model(batch_X).squeeze(dim=1)
            loss = loss_fn(train_logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        model.eval()
        valid_losses = []
        all_probs, all_targets = [], []
        with torch.inference_mode():
            for batch_X, batch_y in valid_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                valid_logits = model(batch_X).squeeze(dim=1)
                valid_probs = torch.sigmoid(valid_logits)

                valid_loss = loss_fn(valid_logits, batch_y)
                valid_losses.append(valid_loss.item())

                all_probs.append(valid_probs)
                all_targets.append(batch_y)

        avg_valid_loss = np.mean(valid_losses)
        all_probs = torch.cat(all_probs)
        all_targets = torch.cat(all_targets)

        valid_auc = roc_auc_score(all_targets.cpu().numpy(), all_probs.cpu().numpy())
        valid_preds = (all_probs >= 0.5).int()
        valid_acc = (valid_preds == all_targets.int()).float().mean().item() * 100


        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Valid Loss: {avg_valid_loss:.4f} | Valid Accuracy: {valid_acc:.2f} |"
              f"Valid AUC: {valid_auc:.4f}")

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            ctr = 0
            torch.save(model.state_dict(), f"best_models/best_{model_type}_model.pth")
        else:
            ctr += 1
            if ctr >= 5:
                print("Not improving, ending training")
                break

def make_model_train_and_test(model_type, input_features, y_train, y_valid, y_test):
    model = MLP(input=input_features).to(device)
    X_train, X_valid, X_test = load_data(f"X_{model_type}")

    train_dataset = MatchDataset(X_train, y_train)
    valid_dataset = MatchDataset(X_valid, y_valid)
    test_dataset = MatchDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    train(model_type, model, train_loader, valid_loader, epochs=50, lr=1e-2, weight_decay=1e-4)

    best_model = MLP(input=input_features)
    best_model.load_state_dict(torch.load(f"best_models/best_{model_type}_model.pth"))
    best_model.to(device)
    best_model.eval()

    all_probs, all_targets = [], []

    with torch.inference_mode():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            test_logits = best_model(batch_X).squeeze(dim=1)
            test_probs = torch.sigmoid(test_logits)

            all_probs.append(test_probs)
            all_targets.append(batch_y)

    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    test_auc = roc_auc_score(all_targets.cpu().numpy(), all_probs.cpu().numpy())
    test_preds = (all_probs >= 0.5).int()
    test_acc = (test_preds == all_targets.int()).float().mean().item() * 100

    return test_acc, test_auc

class MLP(nn.Module):
    def __init__(self, input):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=input, out_features=256), nn.ReLU(),
            nn.Dropout(p=0.2), nn.Linear(in_features=256, out_features=128),
            nn.ReLU(), nn.Linear(in_features=128, out_features=64),
            nn.ReLU(), nn.Linear(in_features=64, out_features=1))
        
    def forward(self, x):
        return self.layers(x)

"""
    MLP_A (Control group) - input_features = 142
        (5 roles * 14 subclasses + 1 winner) * 2 teams = 142

    MLP_B (PCA coordinates) - input_features = 148
        3 coordinates * 2 teams + control group features = 148
    
    MLP_C (Cluster + reliability) - input_features = 148
        (1 cluster label + 1 silhouette score + 1 margin) * 2 teams + control group features = 148
"""
y_train, y_valid, y_test = load_data("y")

model_A_acc, model_A_auc = make_model_train_and_test("A", 142, y_train, y_valid, y_test)
model_B_acc, model_B_auc = make_model_train_and_test("B", 148, y_train, y_valid, y_test)
model_C_acc, model_C_auc = make_model_train_and_test("C", 148, y_train, y_valid, y_test)

print(f"Model A:\n\tTest Accuracy: {model_A_acc:.2f}% | Test AUC: {model_A_auc:.4f}")
print(f"Model B:\n\tTest Accuracy: {model_B_acc:.2f}% | Test AUC: {model_B_auc:.4f}")
print(f"Model C:\n\tTest Accuracy: {model_C_acc:.2f}% | Test AUC: {model_C_auc:.4f}")
