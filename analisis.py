import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# -----------------------------
# Load dataset
# -----------------------------
data_dir = ""  # <-- change this
train_file = os.path.join(data_dir, "competition_data/train_data.txt")

df = pd.read_csv(train_file, sep="\t", low_memory=False)
print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

# -----------------------------
# Derive binary 'skipped' target
# -----------------------------
# Define: trackdone = not skipped, fwdbtn/unknown = skipped
df["skipped"] = df["reason_end"].apply(lambda x: 1 if x in ["fwdbtn", "unknown"] else 0)
print("Skip target distribution:")
print(df["skipped"].value_counts(normalize=True))

# -----------------------------
# 1. Skip Rate by Artist
# -----------------------------
artist_skip = (
    df.groupby("master_metadata_album_artist_name")["skipped"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(8,5))
sns.barplot(x=artist_skip.values, y=artist_skip.index, palette="mako")
plt.title("Top 10 Artists by Skip Rate")
plt.xlabel("Average Skip Rate")
plt.ylabel("Artist")
plt.tight_layout()
plt.show()

# -----------------------------
# 2. Skip Rate by Hour of Day
# -----------------------------
df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
df["hour"] = df["ts"].dt.hour

plt.figure(figsize=(8,5))
sns.lineplot(data=df, x="hour", y="skipped", estimator="mean", errorbar=None)
plt.title("Skip Rate by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Skip Rate")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# 3. Skip Rate by Shuffle Mode
# -----------------------------
plt.figure(figsize=(5,4))
sns.barplot(data=df, x="shuffle", y="skipped", estimator="mean", errorbar=None, palette="viridis")
plt.title("Skip Rate by Shuffle Mode")
plt.xlabel("Shuffle Enabled")
plt.ylabel("Average Skip Rate")
plt.tight_layout()
plt.show()

# -----------------------------
# 4. User Skip Rate Distribution
# -----------------------------
user_skip = df.groupby("username")["skipped"].mean()

plt.figure(figsize=(7,5))
sns.histplot(user_skip, bins=20, kde=True)
plt.title("Distribution of User Skip Rates")
plt.xlabel("User Skip Rate")
plt.ylabel("Count of Users")
plt.tight_layout()
plt.show()


# -----------------------------
# Define 'skipped' target
# -----------------------------
df["skipped"] = df["reason_end"].apply(lambda x: 1 if x in ["fwdbtn", "unknown"] else 0)
df["shuffle"] = df["shuffle"].astype(bool)
df["offline"] = df["offline"].astype(bool)
df["incognito_mode"] = df["incognito_mode"].astype(bool)

# -----------------------------
# Extract hour from timestamp
# -----------------------------
df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
df["hour"] = df["ts"].dt.hour

# =====================================================
# CORRELATION ANALYSIS 1: CATEGORICAL VARIABLES
# =====================================================

def cramers_v(confusion_matrix):
    """Compute Cramér’s V statistic for categorical-categorical association."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

categorical_cols = ["platform", "offline", "incognito_mode", "shuffle", "skipped"]
cramers_results = pd.DataFrame(index=categorical_cols, columns=categorical_cols, dtype=float)

for col1 in categorical_cols:
    for col2 in categorical_cols:
        if col1 == col2:
            cramers_results.loc[col1, col2] = 1.0
        else:
            confusion = pd.crosstab(df[col1], df[col2])
            cramers_results.loc[col1, col2] = cramers_v(confusion)

plt.figure(figsize=(7,5))
sns.heatmap(cramers_results.astype(float), annot=True, cmap="coolwarm", center=0)
plt.title("Cramér’s V Correlation between Categorical Variables")
plt.tight_layout()
plt.show()

# =====================================================
# CORRELATION ANALYSIS 2: HOUR VS SKIP RATE BY PLATFORM
# =====================================================

# Only keep top 5 most common platforms for readability
top_platforms = df["platform"].value_counts().nlargest(5).index
subset = df[df["platform"].isin(top_platforms)]

plt.figure(figsize=(8,5))
sns.lineplot(data=subset, x="hour", y="skipped", hue="platform", estimator="mean", errorbar=None)
plt.title("Skip Rate vs Hour of Day by Platform")
plt.xlabel("Hour of Day")
plt.ylabel("Average Skip Rate")
plt.legend(title="Platform", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()