# Draft-Oracle

ReadMe still a work in progress :)

## Table of Contents

[Introduction](###introduction)  
[Considerations, Warnings, and Licensing]()

###Introduction

The goal of this project is to learn more about data science, and machine/deep learning.  
With this in mind the topic I explored was whether or not it's possible to predict the win rate of a given team composition versus another in the pre-game lobby of a League of Legends game. By proxy determining inherit advantages or patterns in drafting a team.

### Considerations, Warnings, and Licensing

Giving credit where credit is due, not all of this work is my own. This project is meant as a learning experience and not a piece entirely of my own creation, guidance was sought for on the internet. 
Sources' information was often adapted, toyed with, broken, fixed, or manipulated; however, full attribution may not be complete.  
I received assistance from multiple sources including but not limited to:  
- "[*PyTorch for Deep Learning & Machine Learning - Full Course*](https://www.youtube.com/watch?v=V_xro1bcAuA)" on the freeCodeCamp.org youtube channel which is taught by Daniel Bourke
- "[*PyTorch documentation — PyTorch 2.9 documentation*](https://docs.pytorch.org/docs/stable/index.html)"
- "[*scikit-learn: machine learning in Python — scikit-learn 1.7.2 ...*](https://scikit-learn.org/stable/)"
  
Links to the many unlisted sources of information and assistance come from random Google searches and generative AI models like OpenAI's ChatGPT-5 model and DeepSeek's DeepSeek-V3.2 model. If you wish to use any of the information, programs, or files I provide in this repository, 
do so with the mindfulness of where this came from and make an effort to provide credit where you can. Otherwise, I'm not going to be creating any licensing aggreements, but anyone may use programs, or files within this repository as they see fit **as long as they follow any guidelines necessary** such as the ones set by Riot Games.  
Draft Oracle is not endorsed by Riot Games and does not reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties

## Project Overview

- **Data:** ~22,400 games in solo queue masters+ lobbies in patch 25.21 gathered using Riot's LoL API
- **Goal:** Find patterns and perform predictions on the win rate of a team using only the champion roles and subclasses
- **Models Used:** Multi-Layer Perceptrons (MLP), PCA, KMeans, Silhouette scores, Elbow charts, several heatmaps, CramersV, and Mutual Information charts (MI)
- **Insights:** Truth be told more work needs done, team composition clusters tend to fluctuate with more data although seem to converge on 4-5 major archetypes. Cluster v Cluster matchups show all round ~50% winrates making accurate predictions, beyond a coin toss, impossible

## Repository Structure
```
├── README.md
├── champ_class_map.json
├── collector.py
├── deep_analysis.py
├── eda_output/  <-- Data visualization
│   ├── CramersV_top15.png
│   ├── EDA_report.md
│   ├── MI_top15.png
│   ├── PCA_compositions_2D.png
│   ├── PCA_compositions_3D.png
│   ├── cluster_k_evaluation.png
│   ├── cluster_vs_cluster_winrate_fixed.csv
│   ├── cluster_vs_cluster_winrate_fixed.png
│   ├── cluster_winrate_confidence.csv
│   ├── cramers_v.csv
│   ├── global_composition_readable.csv
│   ├── heatmap_freq_ADC.png
│   ├── heatmap_freq_Jungle.png
│   ├── heatmap_freq_Mid.png
│   ├── heatmap_freq_Support.png
│   ├── heatmap_freq_Top.png
│   ├── heatmap_winrate_ADC.png
│   ├── heatmap_winrate_Jungle.png
│   ├── heatmap_winrate_Mid.png
│   ├── heatmap_winrate_Support.png
│   ├── heatmap_winrate_Top.png
│   └── mutual_info.csv
├── matches_25_21.jsonl
├── models/  <-- MLP models and training
│   ├── best_models/  <-- Their home :)
│   │   ├── best_A_model.pth
│   │   ├── best_B_model.pth
│   │   └── best_C_model.pth
│   ├── data/  <-- Data processed for MLP ingestion
│   │   ├── MLP_A_X.npy
│   │   ├── MLP_B_X.npy
│   │   ├── MLP_C_X.npy
│   │   ├── blue_pca_model.pkl
│   │   ├── cluster_model.pkl
│   │   ├── matches.jsonl
│   │   ├── pca_model.pkl
│   │   ├── red_pca_model.pkl
│   │   ├── subclass_enc.pkl
│   │   └── y.npy
│   ├── process_data.py
│   ├── split_data.py
│   ├── splits/  <-- Where MLPs pull training, validation, and testing data
│   │   ├── X_A_test.pt
│   │   ├── X_A_train.pt
│   │   ├── X_A_valid.pt
│   │   ├── X_B_test.pt
│   │   ├── X_B_train.pt
│   │   ├── X_B_valid.pt
│   │   ├── X_C_test.pt
│   │   ├── X_C_train.pt
│   │   ├── X_C_valid.pt
│   │   ├── y_test.pt
│   │   ├── y_train.pt
│   │   └── y_valid.pt
│   └── train_mlp.py
└── seen_ids.json
```

---

## Analysis Summary

### Data Exploration Insights

- PCA plot visually shows 4 distinct groupings
- Optimal cluster count on 3D PCA via KMeans and Silhouette score assessment: **5 clusters**
- Cluster vs Cluster win rates hover 50% regardless of matchup with the most notable being cluster 4 vs 0 with a 55% win rate

**Conclusions:** Cluster counts fluctuate with increasing data size but seem to converge to 4 major clusters. Cluster vs Cluster data shows peculiar results with 4 vs 0 having 55% winrate conversly 0 vs 4 having a 53% win rate. 
Either due to sample size still being too small given data complexity, or cluster win rates need to be normalized against blue vs red side win rate variance. Overall, data shows it is difficult to distinguish win rate probability from team composition.

### MLP Feature Sets

| Model | Features | Feature Count |
|-------|----------|--------------------|
| `MLP_A` | Champion subclasses per role + Winner | Control group with 142 Features |
| `MLP_B` | Control + 3D PCA coordinates | Self Identifies Clusters, 148 Features |
| `MLP_C` | Control + KMeans clustering + Margin & Silhouette score | Predefined clusters, 148 Features |

### Model Performance

| Model | Best Validation Accuracy | AUC score | Test Accuracy | AUC score |
|-------|--------------------------|-----------|--------------------|-----------|
| `MLP_A` | 51.70% | 0.5000 | 51.72% | 0.5000 |
| `MLP_B` | 51.74% | 0.4880 | 51.72% | 0.4995 |
| `MLP_C` | 51.70% | 0.5000 | 51.72% | 0.5016 |

**Conclusions:** No model was able to do better than random guessing and nearly immediately found the overall blue win rate percentage of the given dataset which is 11585/22405 = **51.707%**  
I stopped model training after 5 epochs of no score improvements, model A trained for 11 epochs, B trained for 7, and C trained for 10. Removing random seeds barely changes these results, with all models halting their training within a couple epochs of eachother.  
This result corroborates the findings from data exploration/analysis. Little to no predictive pattern emerges from the given data features, which is why each model seems to solely rely on overall win rate as its main feature.

---

## Epilogue

At the end of the day, this project was an excuse for me to learn about data science and engineering as well as exploring machine/deep learning practices and algorithms

### What I Learned

- Different real world data collection, cleaning, and analysis techniques
- Data analysis and visualization techniques (take a while) provide opportunities to recognize patterns and rationalize results
- Reinforcement on structuring data and evaluating machine learning pipelines using PyTorch

### Future Work

- Go into a deeper analysis of the given data, PCA plots per cluster, different classification, and clustering algorithms
- Slowly integrate more pre-game and in-game features to evaluate their strength in predicting win rates
- Try different neural networks like Graph Neural Networks (GNNs) or even temporal evaluation like live drafting using transformers
- Gathering more data; I'd like to avoid introducing unnecassary variables when possible but I'd also like to see this data across regions and even patches as well as wholelistically
- Complete full technical analysis report and post it in this repo :D

### Tech Stack & Dependencies

- **Python 3.12**
- Riot Games API
- Python libraries and imports can be found in requirements.txt

### Usage Pipeline
```bash
python collector.py  <-- Collects matches using Riot API
python deep_analysis.py  <-- Creates plots, charts, and tables for human analysis and visualization
python models/process_data.py  <-- Processes data for MLP ingestion
python models/split_data.py  <-- Splits processed data into train/valid/test directories 80/10/10
python models/train_mlp.py  <-- Creates, trains, saves the MLPs
```
---

### Author

Me... aspiring software engineer studying at the University of Central Florida  
