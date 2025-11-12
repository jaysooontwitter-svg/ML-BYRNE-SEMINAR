
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

# ---------- 1) Count of Patients by Race (Bar) ----------
race_counts = df['race'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(9,5))
race_counts.plot(kind="bar")
plt.title("Count of Patients by Race")
plt.xlabel("Race")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ---------- 2) Age Distribution by Race (Boxplot) ----------
groups = [g['age'].dropna().values for name, g in df.groupby('race')]
labels = [name for name, g in df.groupby('race')]

plt.figure(figsize=(10,5))
plt.boxplot(groups, labels=labels, showmeans=True)
plt.title("Age Distribution by Race")
plt.xlabel("Race")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

# ---------- 3) Gender Proportions by Race (Grouped Bar) ----------
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

# ---------- 4) Marital Status (Top 4) Proportions by Race (Grouped Bar) ----------
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

# ---------- 5) Ethnicity Proportions by Race (Grouped Bar; Top 4) ----------
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

# ---------- Statistical tests (Numeric and Categorical) ----------
# Age: ANOVA and Kruskal-Wallis
groups_for_tests = [g['age'].dropna().values for _, g in df.groupby('race') if len(g) > 1]
anova_F = anova_p = kw_H = kw_p = np.nan
if len(groups_for_tests) > 1:
    anova_F, anova_p = stats.f_oneway(*groups_for_tests)
    kw_H, kw_p = stats.kruskal(*groups_for_tests)

numeric_tests = pd.DataFrame({
    "metric": ["age"],
    "anova_F": [anova_F],
    "anova_p": [anova_p],
    "kruskal_H": [kw_H],
    "kruskal_p": [kw_p]
})
numeric_tests.to_csv("/mnt/data/numeric_tests.csv", index=False)

# Categorical: Chi-square + Cramer's V
def chi_square_test(df, target_col, cat_col):
    ct = pd.crosstab(df[target_col], df[cat_col])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    cramer_v = np.sqrt((chi2/n) / (min(k-1, r-1))) if min(k-1, r-1) > 0 else np.nan
    return {"feature": cat_col, "chi2": chi2, "p_value": p, "cramers_v": cramer_v, "rows": r, "cols": k}

cat_rows = []
for c in cat_cols:
    cat_rows.append(chi_square_test(df, 'race', c))
cat_tests = pd.DataFrame(cat_rows)
cat_tests.to_csv("/mnt/data/categorical_tests.csv", index=False)

# Save age summary by race
age_summary = df.groupby('race')['age'].agg(['count','mean','std','median','min','max']).reset_index()
age_summary.to_csv("/mnt/data/age_by_race_summary.csv", index=False)

print("Saved:", "/mnt/data/numeric_tests.csv", "/mnt/data/categorical_tests.csv", "/mnt/data/age_by_race_summary.csv")
