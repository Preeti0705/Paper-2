"""
Data Preprocessing Module for CSDL-SEN-XAI Framework
Reference: "An Interpretable Analytical Intelligence Architecture Delivering Reliable Detection of Software Defect Instances"
(IJACSA, Vol. 16, No. 12, 2025)

Steps:
1. Missing Value Imputation (Mean/Median for continuous, Mode for categorical)
2. Pearson Correlation Pruning (Remove highly collinear/redundant metrics)
3. Min-Max Normalization to [0, 1] range (Eq. 1)
4. Stratified 80:20 Train-Test Partitioning
5. SMOTE Balancing (Eq. 2, Eq. 3) applied STRICTLY to training data only
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os


def load_and_preprocess_data(
    file_path="cm1.csv",
    correlation_threshold=0.98,
    test_size=0.20,
    random_state=42,
    smote_k_neighbors=5,
):
    """
    Loads NASA CM1 dataset, cleans missing values, eliminates highly collinear features,
    applies Min-Max scaling, partitions into 80:20 train-test sets, and balances
    the training set using SMOTE.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path)

    # Drop non-predictive id column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Identify target column
    target_col = None
    for candidate in ["defects", "Defect", "defect", "label", "problems"]:
        if candidate in df.columns:
            target_col = candidate
            break

    if target_col is None:
        target_col = df.columns[-1]

    # Map target column to binary integers (0 = non-defective, 1 = defective)
    y_raw = df[target_col]
    if y_raw.dtype == bool or y_raw.dtype == object:
        y = y_raw.map(lambda x: 1 if str(x).strip().lower() in ["true", "1", "y", "yes", "defective"] else 0).values
    else:
        y = (y_raw.values > 0).astype(int)

    X_df = df.drop(columns=[target_col])

    # 1. Missing Value Imputation
    for col in X_df.columns:
        if X_df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(X_df[col]):
                X_df[col] = X_df[col].fillna(X_df[col].median())
            else:
                X_df[col] = X_df[col].fillna(X_df[col].mode()[0])

    # Convert all feature columns to float
    X_df = X_df.astype(float)

    # 2. Pearson Correlation Pruning (drop highly collinear metrics)
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    
    if len(to_drop) > 0:
        X_df = X_df.drop(columns=to_drop)

    feature_names = list(X_df.columns)
    X = X_df.values

    # 3. Stratified 80:20 Train-Test Split (Section III-B & Table II)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 4. Min-Max Normalization to [0, 1] (Eq. 1: x' = (x - x_min)/(x_max - x_min))
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train_raw)
    X_test_norm = scaler.transform(X_test_raw)

    # 5. SMOTE Imbalance Handling (Eq. 2-3) on Training Set ONLY
    # N_synthetic = ((n_majority / n_minority) - 1) * n_minority
    smote = SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_norm, y_train)

    return {
        "X_train": X_train_balanced,
        "y_train": y_train_balanced,
        "X_test": X_test_norm,
        "y_test": y_test,
        "X_train_unbalanced": X_train_norm,
        "y_train_unbalanced": y_train,
        "scaler": scaler,
        "feature_names": feature_names,
        "dropped_features": to_drop,
        "raw_df": df,
    }


def save_preprocessor(scaler, feature_names, save_path="models/preprocessor.joblib"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"scaler": scaler, "feature_names": feature_names}, save_path)


def load_preprocessor(load_path="models/preprocessor.joblib"):
    data = joblib.load(load_path)
    return data["scaler"], data["feature_names"]


if __name__ == "__main__":
    data = load_and_preprocess_data()
    print("=== Preprocessing Summary ===")
    print(f"Original Features: {len(data['feature_names']) + len(data['dropped_features'])}")
    print(f"Retained Features ({len(data['feature_names'])}): {data['feature_names']}")
    print(f"Dropped collinear features ({len(data['dropped_features'])}): {data['dropped_features']}")
    print(f"Train Balanced shape: {data['X_train'].shape}, Class 0: {(data['y_train']==0).sum()}, Class 1: {(data['y_train']==1).sum()}")
    print(f"Test shape: {data['X_test'].shape}, Class 0: {(data['y_test']==0).sum()}, Class 1: {(data['y_test']==1).sum()}")