import os
import json
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

INPUT_JSONL = "data/matches.jsonl"
OUTPUT_DIR = "data"
SUBCLASSES = [
        "Juggernaut", "Burst", "Skirmisher", "Marksman", "Catcher",
        "Artillery", "Battlemage", "Diver", "Slayer", "Specialist",
        "Warden", "Vanguard", "Enchanter", "Assassin"
        ]

ROLES = ["Top", "Jungle", "Mid", "ADC", "Support"]

def load_matches(filepath):
    matches = []
    with open(filepath, "r") as f:
        for line in f:
            matches.append(json.loads(line))
    return matches

print("Loading matches...")
matches = load_matches(INPUT_JSONL)
print(f"Loaded {len(matches)} matches\n")

enc = OneHotEncoder(categories=[SUBCLASSES], sparse_output=False, handle_unknown="ignore")
enc.fit(np.array(SUBCLASSES).reshape(-1, 1))
joblib.dump(enc, os.path.join(OUTPUT_DIR, "subclass_enc.pkl"))
print("Saved subclass encoder")

X_A = []
y = []

for m in tqdm(matches, desc="Processing matches"):
    row = []

    y.append(1.0 if m["winner"] == "blue" else 0.0)

    for side in ["blue", "red"]:
        row.append(0.0 if side == "blue" else 1.0)

        for role in ROLES:
            subclass = m[side][role]["subclass"]
            row.extend(enc.transform([[subclass]]).flatten())
        
    X_A.append(row)

X_A = np.array(X_A, dtype=np.float32)
y = np.array(y, dtype=np.float32)

np.save(os.path.join(OUTPUT_DIR, "MLP_A_X.npy"), X_A)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
print(f"Saved MLP_A_X.npy with shape {X_A.shape}")

# PCA ------------------------------------------------

print("\nCreating per-side PCA...")
"""
    Nesting hurts the brain...
    blue side: 5 roles * 14 subclasses, red side: 5 roles * 14 subclasses
    5 * 14 = 70
"""
NUM_ROLES = len(ROLES)                      # 5
NUM_SUBCLASSES = len(SUBCLASSES)            # 14
PER_SIDE_FEATS = NUM_ROLES * NUM_SUBCLASSES # 70

blue_feats = X_A[:, 1:1+PER_SIDE_FEATS]
red_feats = X_A[:, 1+PER_SIDE_FEATS+1:1+PER_SIDE_FEATS+1+PER_SIDE_FEATS] # Skipping all the unnecessary parent flags
both_sides_feats = np.vstack([blue_feats, red_feats])

pca = PCA(n_components=3, random_state=42)
pca.fit(both_sides_feats)

blue_pca = pca.transform(blue_feats)
red_pca = pca.transform(red_feats)

joblib.dump(pca, os.path.join(OUTPUT_DIR, "pca_model.pkl"))
joblib.dump(blue_pca, os.path.join(OUTPUT_DIR, "blue_pca_model.pkl"))
joblib.dump(red_pca, os.path.join(OUTPUT_DIR, "red_pca_model.pkl"))
print("Saved PCA model")

X_B = np.hstack([X_A, blue_pca, red_pca])
np.save(os.path.join(OUTPUT_DIR, "MLP_B_X.npy"), X_B)
print(f"Saved MLP_B_X.npy with shape {X_B.shape}")

# Clustering ------------------------------------------------

print("\nClustering per-side PCA...")
kmeans = KMeans(n_clusters=4,n_init="auto", random_state=42).fit(both_sides_feats)
joblib.dump(kmeans, os.path.join(OUTPUT_DIR, "cluster_model.pkl"))
print("Saved KMeans model")

blue_labels = kmeans.predict(blue_feats)
red_labels = kmeans.predict(red_feats)

both_labels = kmeans.labels_
both_sils = silhouette_samples(both_sides_feats, both_labels)

blue_sils = both_sils[:len(blue_feats)]
red_sils = both_sils[len(blue_feats):]

blue_dists = kmeans.transform(blue_feats)
red_dists = kmeans.transform(red_feats)

blue_margins = np.partition(blue_dists, 1, axis=1)[:, 1] - blue_dists.min(axis=1)
red_margins = np.partition(red_dists, 1, axis=1)[:, 1] - red_dists.min(axis=1)

blue_cluster_info = np.vstack([blue_labels, blue_sils, blue_margins]).T
red_cluster_info = np.vstack([red_labels, red_sils, red_margins]).T
X_C = np.hstack([X_A, blue_cluster_info, red_cluster_info])
np.save(os.path.join(OUTPUT_DIR, "MLP_C_X.npy"), X_C)
print(f"Saved MLP_C_X.npy with shape {X_C.shape}\n")

print("All Done :)")