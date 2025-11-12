
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------- Load & basic prep ----------
df = pd.read_csv("/mnt/data/synthetic_nursing_notes.csv")
df['race'] = df['race'].astype(str).str.strip().str.title()

# Identify key columns
num_cols = ['age']
cat_cols = [c for c in ['gender', 'ethnicity', 'marital_status'] if c in df.columns]

#
race_counts = df['race'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(9,5))
race_counts.plot(kind="bar")
plt.title("Count of Patients by Race")
plt.xlabel("Race")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#Age Distribution by Race 
groups = [g['age'].dropna().values for name, g in df.groupby('race')]
labels = [name for name, g in df.groupby('race')]

plt.figure(figsize=(10,5))
plt.boxplot(groups, labels=labels, showmeans=True)
plt.title("Age Distribution by Race")
plt.xlabel("Race")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

# Gener Proportions
if 'gender' in df.columns:
    prop_gender = pd.crosstab(df['race'], df['gender'], normalize='index').reindex(race_counts.index)
    plt.figure(figsize=(10,5))
    x = np.arange(len(prop_gender.index))
    width = 0.8 / max(1, prop_gender.shape[1])
    for i, col in enumerate(prop_gender.columns):
        plt.bar(x + i*width, prop_gender[col].values, width=width, label=col)
    plt.xticks(x + (prop_gender.shape[1]-1)*width/2, prop_gender.index, rotation=0)
    plt.title("Gender Proportions by Race")
    plt.ylabel("Proportion")
    plt.legend()
    plt.tight_layout()
    plt.show()

#  Marital Status Chart
if 'marital_status' in df.columns:
    ct_mar = pd.crosstab(df['race'], df['marital_status'])
    top_mar = ct_mar.sum().sort_values(ascending=False).head(4).index
    prop_mar = pd.crosstab(df['race'], df['marital_status'], normalize='index')[top_mar].reindex(race_counts.index)
    plt.figure(figsize=(11,5))
    x = np.arange(len(prop_mar.index))
    width = 0.8 / max(1, prop_mar.shape[1])
    for i, col in enumerate(prop_mar.columns):
        plt.bar(x + i*width, prop_mar[col].values, width=width, label=col)
    plt.xticks(x + (prop_mar.shape[1]-1)*width/2, prop_mar.index, rotation=0)
    plt.title("Marital Status (Top 4) Proportions by Race")
    plt.ylabel("Proportion")
    plt.legend()
    plt.tight_layout()
    plt.show()
#--Ethnicity chart
if 'ethnicity' in df.columns:
    prop_eth = pd.crosstab(df['race'], df['ethnicity'], normalize='index').reindex(race_counts.index)
    top_eth = prop_eth.sum().sort_values(ascending=False).head(4).index
    prop_eth2 = prop_eth[top_eth]
    plt.figure(figsize=(10,5))
    x = np.arange(len(prop_eth2.index))
    width = 0.8 / max(1, prop_eth2.shape[1])
    for i, col in enumerate(prop_eth2.columns):
        plt.bar(x + i*width, prop_eth2[col].values, width=width, label=col)
    plt.xticks(x + (prop_eth2.shape[1]-1)*width/2, prop_eth2.index, rotation=0)
    plt.title("Ethnicity (Top 4) Proportions by Race")
    plt.ylabel("Proportion")
    plt.legend()
    plt.tight_layout()
    plt.show()
