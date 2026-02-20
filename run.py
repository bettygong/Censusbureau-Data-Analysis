# ===============================
# Census Income Classification + Customer Segmentation
# Author: Ziling Gong
# ===============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, average_precision_score
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from xgboost import XGBClassifier


# 1. LOAD DATA
print("Loading data...")
cols = open("census-bureau.columns").read().splitlines()
df = pd.read_csv("census-bureau.data", names=cols)

df["encode_label"] = (df["label"] != "- 50000.").astype(int)
df["hispanic origin"] = df["hispanic origin"].fillna("Do not know")

print("Dataset shape:", df.shape)


# 2. FEATURE SELECTION
numerical_cols=['age',
       'wage per hour', 'capital gains', 'capital losses',
       'dividends from stocks', 'num persons worked for employer',
       'weeks worked in year']

categorical_cols=['class of worker', 'education', 'enroll in edu inst last wk',
       'marital stat', 'major industry code', 'major occupation code', 'race',
       'hispanic origin', 'sex', 'member of a labor union',
       'reason for unemployment', 'full or part time employment stat',
       'tax filer stat', 'region of previous residence',
      #  'state of previous residence', # have region of previous residence
       'detailed household and family stat',
       'detailed household summary in household',
       'migration code-change in msa', 'migration code-change in reg',
       'migration code-move within reg', 'live in this house 1 year ago',
       'migration prev res in sunbelt', 'family members under 18',
      #  'country of birth father', 'country of birth mother','country of birth self',
       'citizenship',
       'fill inc questionnaire for veteran\'s admin',
       'detailed industry recode', 'detailed occupation recode',
       'own business or self employed', 'veterans benefits', 'year']

df[categorical_cols] = df[categorical_cols].astype("category")

X = df[numerical_cols + categorical_cols]
y = df["encode_label"]
weights = df["weight"]

# ===============================
# 3. PREPROCESSING PIPELINE
# ===============================
print("Building preprocessing pipeline...")

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse=True), categorical_cols) # newer version sklearn should use sparse_output=True
])

# Train/val/test split
X_temp, X_test, y_temp, y_test, w_temp, w_test = train_test_split(
    X, y, weights, test_size=0.2, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
    X_temp, y_temp, w_temp, test_size=0.2, random_state=42
)

X_train_p = preprocess.fit_transform(X_train)
X_val_p = preprocess.transform(X_val)
X_test_p = preprocess.transform(X_test)

print("Processed feature shape:", X_train_p.shape)


# ===============================
# 4. LOGISTIC REGRESSION (BASELINE)
# ===============================
print("\nTraining Logistic Regression...")

lg = LogisticRegression(
    class_weight="balanced",
    penalty="l1",
    solver="saga",
    max_iter=1000,
    n_jobs=-1
)

lg.fit(X_train_p, y_train)

val_prob = lg.predict_proba(X_val_p)[:, 1]
print("Logistic validation PR-AUC:", average_precision_score(y_val, val_prob))


# ===============================
# 5. XGBOOST MODEL
# ===============================
print("\nTraining XGBoost...")

scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos,
    eval_metric="aucpr",
)

xgb.fit(X_train_p, y_train, sample_weight=w_train)

test_pred = xgb.predict(X_test_p)
test_prob = xgb.predict_proba(X_test_p)[:, 1]

print("\nXGBoost test Results")
print(classification_report(y_test, test_pred))
print("Test PR-AUC:", average_precision_score(y_test, test_prob))


# ===============================
# 6. CUSTOMER SEGMENTATION
# ===============================

print("\nRunning Customer Segmentation...")

# Use FULL dataset for clustering
X_full = preprocess.fit_transform(X)

# Dimensionality reduction
svd = TruncatedSVD(n_components=10, random_state=42)
X_reduced = svd.fit_transform(X_full)

# Choose k=5
kmeans = KMeans(n_clusters=5, n_init=20, random_state=42)
clusters = kmeans.fit_predict(X_reduced)

# Evaluate silhouette on sample
sample_idx = np.random.choice(len(X_reduced), size=10000, replace=False)
score = silhouette_score(X_reduced[sample_idx], clusters[sample_idx])
print("Silhouette Score (k=5):", score)

df["cluster"] = clusters


# ===============================
# 7. SAVE RESULTS
# # ===============================

# df[["encode_label", "cluster"]].to_csv("customer_segments_output.csv", index=False)

# print("\nSaved cluster assignments to customer_segments_output.csv")
# print("Pipeline complete.")
