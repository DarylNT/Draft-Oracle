import os, json, gzip, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # silence loky Windows warning - adjust to system capabilities
sns.set_theme(style="whitegrid")

INPUT = "matches_25_21.jsonl"
REPORT_DIR = "eda_output"
os.makedirs(REPORT_DIR, exist_ok=True)

def load_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    records = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception as e:
                print("Bad line:", e)
    return records

data = load_jsonl(INPUT)
print(f"Loaded {len(data)} matches")

rows = []
for m in data:
    row = {"patch": m.get("patch"), "winner": m.get("winner"), "duration": m.get("duration")}
    for side in ["blue", "red"]:
        for role in ["Top", "Jungle", "Mid", "ADC", "Support"]:
            champ = m[side].get(role, {}).get("champion", "Unknown")
            major = m[side].get(role, {}).get("class", "Unknown")
            sub   = m[side].get(role, {}).get("subclass", "Unknown")
            row[f"{side}_{role}_champion"] = champ
            row[f"{side}_{role}_class"]    = major
            row[f"{side}_{role}_subclass"] = sub
    rows.append(row)

df = pd.DataFrame(rows)
print(f"Flattened shape: {df.shape}")

winrates = df["winner"].value_counts(normalize=True)
patches  = df["patch"].value_counts()
print("\nWin rate distribution:\n", winrates)
print("\nPatch distribution:\n", patches)

def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table)[0]
    n = table.sum().sum()
    r, k = table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

cat_cols = [c for c in df.columns if c.endswith("_class") or c.endswith("_subclass")]
X = df[cat_cols].astype("category").apply(lambda c: c.cat.codes)
y = (df["winner"] == "blue").astype(int)

mi = mutual_info_classif(X, y, discrete_features=True, random_state=42)
mi_series = pd.Series(mi, index=cat_cols).sort_values(ascending=False)
mi_series.to_csv(os.path.join(REPORT_DIR, "mutual_info.csv"))

cramers = {col: cramers_v(df[col], df["winner"]) for col in cat_cols}
cramers_series = pd.Series(cramers).sort_values(ascending=False)
cramers_series.to_csv(os.path.join(REPORT_DIR, "cramers_v.csv"))

plt.figure(figsize=(8,6))
sns.barplot(y=mi_series.head(15).index, x=mi_series.head(15).values)
plt.title("Top Mutual Information vs Winner")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "MI_top15.png"))
plt.close()

plt.figure(figsize=(8,6))
sns.barplot(y=cramers_series.head(15).index, x=cramers_series.head(15).values)
plt.title("Cramer's V Association Strength (vs Winner)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "CramersV_top15.png"))
plt.close()

# Heat maps ---------------------------------------------------------------- 

roles = ["Top","Jungle","Mid","ADC","Support"]
for role in roles:
    cross = pd.crosstab(df[f"blue_{role}_class"], df[f"red_{role}_class"])
    plt.figure(figsize=(6,5))
    sns.heatmap(cross, cmap="crest", annot=True, fmt="d")
    plt.title(f"Role-Class Frequency Blue vs Red {role}")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f"heatmap_freq_{role}.png"))
    plt.close()

    pivot = pd.crosstab(
        df[f"blue_{role}_class"],
        df[f"red_{role}_class"],
        values=(df["winner"]=="blue"),
        aggfunc="mean"
    ).fillna(0)
    plt.figure(figsize=(6,5))
    sns.heatmap(pivot, cmap="coolwarm", center=0.5, annot=True, fmt=".2f")
    plt.title(f"Win-Rate Heatmap Blue vs Red {role} (blue wins)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f"heatmap_winrate_{role}.png"))
    plt.close()

# PCA + Clustering ------------------------------------------------

blue_cols = [c for c in df.columns if c.startswith("blue_") and c.endswith("_subclass")]
red_cols  = [c for c in df.columns if c.startswith("red_")  and c.endswith("_subclass")]

enc = LabelEncoder()
blue_enc = df[blue_cols].apply(lambda c: enc.fit_transform(c.astype(str)))
red_enc  = df[red_cols].apply(lambda c: enc.fit_transform(c.astype(str)))

pca = PCA(n_components=3, random_state=42)
blue_coords = pca.fit_transform(blue_enc)
red_coords  = pca.fit_transform(red_enc)

print("\nFinding optimal cluster count...")
inertias, silhouettes = [], []
K_RANGE = range(2, 9)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(blue_coords)
    inertias.append(km.inertia_)
    sil = silhouette_score(blue_coords, labels)
    silhouettes.append(sil)
    print(f"k={k}: inertia={km.inertia_:.0f}, silhouette={sil:.3f}")

# plot elbow + silhouette ----------------------------------------------------

fig, ax = plt.subplots(1, 2, figsize=(10,4))
sns.lineplot(x=list(K_RANGE), y=inertias, marker="o", ax=ax[0])
ax[0].set_title("Elbow Method (Inertia vs K)")
ax[0].set_xlabel("k")
ax[0].set_ylabel("Inertia")
sns.lineplot(x=list(K_RANGE), y=silhouettes, marker="o", ax=ax[1])
ax[1].set_title("Silhouette Score vs K")
ax[1].set_xlabel("k")
ax[1].set_ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "cluster_k_evaluation.png"))
plt.close()

best_k = K_RANGE[np.argmax(silhouettes)]
print(f"\nOptimal cluster count (by silhouette): k={best_k}")

print(f"Clustering with {best_k} clusters")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
df["blue_cluster"] = kmeans.fit_predict(blue_coords)
df["red_cluster"]  = kmeans.fit_predict(red_coords)

pivot_clusters = pd.crosstab(
    df["blue_cluster"],
    df["red_cluster"],
    values=(df["winner"]=="blue"),
    aggfunc="mean"
).fillna(0)

plt.figure(figsize=(6,5))
sns.heatmap(pivot_clusters, annot=True, cmap="coolwarm", center=0.5, fmt=".2f")
plt.title("Cluster vs Cluster Win Rate (Blue Wins)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "cluster_vs_cluster_winrate_fixed.png"))
pivot_clusters.to_csv(os.path.join(REPORT_DIR, "cluster_vs_cluster_winrate_fixed.csv"))
plt.close()

# Cluster size distribution ------------------------------------------------------------------

cluster_sizes = df["blue_cluster"].value_counts().sort_index()
print("\nCluster sizes (Blue side):")
print(cluster_sizes)
print(f"Average size: {cluster_sizes.mean():.1f}, StdDev: {cluster_sizes.std():.1f}")

# Top subclasses per cluster -----------------------------------------------------------------

sub_cols = [c for c in df.columns if c.endswith("_subclass")]
blue_subs = df[sub_cols].copy()
blue_subs["cluster"] = df["blue_cluster"]

top_subclasses = {}
for k in sorted(df["blue_cluster"].unique()):
    subset = blue_subs[blue_subs["cluster"] == k]
    counts = (
        subset.drop(columns="cluster")
        .melt()["value"]
        .value_counts(normalize=True)
        .head(10)
        .mul(100)
        .round(2)
    )
    top_subclasses[k] = counts
    print(f"\nTop subclasses in Cluster {k}:")
    print(counts.to_string())

match_counts = pd.crosstab(df["blue_cluster"], df["red_cluster"])
winrates = pd.crosstab(
    df["blue_cluster"], df["red_cluster"], values=(df["winner"]=="blue"), aggfunc="mean"
).fillna(0)

# compute simple Wilson-like CI band
def proportion_ci(p, n):
    if n == 0: return (np.nan, np.nan)
    z = 1.96  # 95% CI
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half_width = (z * np.sqrt((p*(1-p)/n) + (z**2/(4*n**2)))) / denom
    return (round(center-half_width,3), round(center+half_width,3))

ci_table = pd.DataFrame(index=winrates.index, columns=winrates.columns)
for i in winrates.index:
    for j in winrates.columns:
        p, n = winrates.loc[i,j], match_counts.loc[i,j]
        lo, hi = proportion_ci(p, n)
        ci_table.loc[i,j] = f"{p:.2f} [{lo:.2f}-{hi:.2f}]"

ci_table.to_csv(os.path.join(REPORT_DIR, "cluster_winrate_confidence.csv"))

sub_cols = [c for c in df.columns if c.endswith("_subclass")]
class_cols = [c for c in df.columns if c.endswith("_class")]
comp_long = pd.DataFrame()

for side, cluster_col in [("blue", "blue_cluster"), ("red", "red_cluster")]:
    for cols, label in [(sub_cols, "subclass"), (class_cols, "class")]:
        relevant = [c for c in cols if c.startswith(side)]
        temp = df[relevant + [cluster_col]].melt(
            id_vars=[cluster_col],
            var_name="role_side",
            value_name=label
        )[[cluster_col, label]].rename(columns={cluster_col: "cluster"})
        comp_long = pd.concat([comp_long, temp], ignore_index=True)

composition = (
    comp_long.groupby(["cluster", "subclass"])
    .size()
    .groupby(level=0)
    .apply(lambda x: 100 * x / x.sum())
    .to_frame("pct")
)

composition_pivot = composition.pivot_table(
    index="cluster",
    columns="subclass",
    values="pct",
    aggfunc="sum",
    fill_value=0
).round(3)

composition_pivot = composition_pivot.div(composition_pivot.sum(axis=1), axis=0).mul(100).round(2)
composition_pivot.to_csv(os.path.join(REPORT_DIR, "global_composition_readable.csv"))

flat_enc = pd.concat([blue_enc, red_enc], axis=1)
coords2 = PCA(n_components=2, random_state=42).fit_transform(flat_enc)
df["PCA1"], df["PCA2"] = coords2[:,0], coords2[:,1]

# --- 2D scatter
plt.figure(figsize=(7,6))
sns.scatterplot(x="PCA1", y="PCA2", hue="blue_cluster", data=df, palette="tab10", alpha=0.6)
plt.title("2D PCA of Team Compositions (colored by cluster)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "PCA_compositions_2D.png"))
plt.close()

# --- 3D scatter
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
colors = sns.color_palette("tab10", n_colors=best_k)
for k in range(best_k):
    subset = blue_coords[df["blue_cluster"]==k]
    ax.scatter(subset[:,0], subset[:,1], subset[:,2],
               s=30, label=f"Cluster {k}", alpha=0.7, color=colors[k])
ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2"); ax.set_zlabel("PCA 3")
ax.set_title("3D PCA of Blue Team Compositions")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "PCA_compositions_3D.png"))
plt.close()

report_path = os.path.join(REPORT_DIR, "EDA_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"# EDA Report – {datetime.now():%Y-%m-%d %H:%M}\n")
    f.write(f"Analyzed {len(df)} matches.\n\n")
    f.write("## Basic Statistics\n")
    f.write(df.describe(include='all').to_markdown() + "\n\n")
    f.write("## Win Rate Distribution\n")
    f.write(winrates.to_markdown() + "\n\n")
    f.write("## Top 15 Mutual Information Features\n")
    f.write(mi_series.head(15).to_markdown() + "\n\n")
    f.write("## Top 15 Cramer's V Associations\n")
    f.write(cramers_series.head(15).to_markdown() + "\n\n")
    f.write("## Generated Plots\n")
    for img in sorted(os.listdir(REPORT_DIR)):
        if img.endswith(".png"):
            f.write(f"![]({img})\n")

print("\nFin")
