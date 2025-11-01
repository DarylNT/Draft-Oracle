import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

X_A = np.load("data/MLP_A_X.npy")
X_B = np.load("data/MLP_B_X.npy")
X_C = np.load("data/MLP_C_X.npy")
y = np.load("data/y.npy")

RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = "splits"

X_A_train, X_A_temp, y_train, y_temp = train_test_split(X_A, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_A_valid, X_A_test, y_valid, y_test = train_test_split(X_A_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)

X_B_train, X_B_temp = train_test_split(X_B, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_B_valid, X_B_test = train_test_split(X_B_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)

X_C_train, X_C_temp = train_test_split(X_C, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_C_valid, X_C_test = train_test_split(X_C_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)

def save_splits(prefix, train, valid, test):
    torch.save(train, os.path.join(OUTPUT_DIR, f"{prefix}_train.pt"))
    torch.save(valid, os.path.join(OUTPUT_DIR, f"{prefix}_valid.pt"))
    torch.save(test, os.path.join(OUTPUT_DIR, f"{prefix}_test.pt"))

save_splits("X_A", X_A_train, X_A_valid, X_A_test)
save_splits("X_B", X_B_train, X_B_valid, X_B_test)
save_splits("X_C", X_C_train, X_C_valid, X_C_test)
save_splits("y", y_train, y_valid, y_test)

print("Data split and saved, go next")