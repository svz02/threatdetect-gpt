import os
import glob
import logging
from typing import Tuple, List

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import joblib

logger = logging.getLogger(__name__)

# 1) Load & merge dataset
def load_and_merge_csvs(data_dir: str, pattern: str = "UNSW-NB15_*.csv") -> pd.DataFrame:
    column_names = [
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
        "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth",
        "response_body_len", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
        "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "srcip_duplicated", "dstip_duplicated",
        "attack_cat", "label"
    ]
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files found for pattern {pattern} in {data_dir}")
    logger.info("Found %d files, first few: %s", len(files), files[:3])
    df_list = [pd.read_csv(f, header=None, names=column_names) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    logger.info("Merged DataFrame shape: %s", df.shape)
    return df

# 2) Prepare labels & quick EDA
def prepare_target(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    if label_col not in df.columns:
        raise KeyError(f"{label_col} not in dataframe columns: {df.columns.tolist()[:10]}")
    df = df.copy()
    df["target"] = df[label_col].astype(int)
    logger.info("Target distribution:\n%s", df["target"].value_counts(dropna=False))
    return df

# 3) Build preprocessing pipeline
def build_preprocessor(df: pd.DataFrame,
                       text_columns: List[str] = None,
                       max_tfidf_features: int = 500) -> Tuple[ColumnTransformer, List[str]]:
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_features:
        df[col] = df[col].astype(str)
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ["label", "target", "attack_cat"]:
        if col in numeric_features:
            numeric_features.remove(col)
        if col in categorical_features:
            categorical_features.remove(col)
    # Remove text columns from categorical if specified
    text_columns = text_columns or []
    for col in text_columns:
        if col in categorical_features:
            categorical_features.remove(col)
    transformers = []
    if categorical_features:
        transformers.append(("cat_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                             categorical_features))
    if numeric_features:
        transformers.append(("num", "passthrough", numeric_features))
    for i, text_col in enumerate(text_columns):
        vect = TfidfVectorizer(max_features=max_tfidf_features)
        transformers.append((f"tfidf_{i}", Pipeline([("selector", FunctionSelector(text_col)), ("tfidf", vect)]), [text_col]))
    preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    feature_list = numeric_features + categorical_features + text_columns
    logger.info("Preprocessor built. Numeric: %d, Categorical: %d, Text: %d",
                len(numeric_features), len(categorical_features), len(text_columns))
    return preprocessor, feature_list

from sklearn.base import BaseEstimator, TransformerMixin
class FunctionSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.key].astype(str).values
        return X

# 4) Train pipeline + evaluate
def train_and_evaluate(df: pd.DataFrame,
                       preprocessor,
                       classifier=None,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       model_out_path: str = "models/rf_pipeline.joblib"):
    if classifier is None:
        classifier = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=random_state)
    X = df.drop(columns=["label", "target","attack_cat"], errors="ignore")
    y = df["target"].values
    #debug prints
    #print("Columns in X:", X.columns.tolist())
    print("Number of duplicate rows:", df.duplicated().sum())
    #rint(X.head())
    #rint(y[:5])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)
    logger.info("Train/test sizes: %d / %d", X_train.shape[0], X_test.shape[0])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", classifier)
    ])
    logger.info("Fitting the pipeline...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred, digits=4))
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))
    if y_proba is not None:
        try:
            roc = roc_auc_score(y_test, y_proba)
            logger.info("ROC AUC: %.4f", roc)
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall, precision)
            logger.info("Precision-Recall AUC: %.4f", pr_auc)
        except Exception as e:
            logger.warning("Could not compute ROC/PR AUC: %s", e)
    os.makedirs(os.path.dirname(model_out_path) or ".", exist_ok=True)
    joblib.dump(pipeline, model_out_path)
    logger.info("Saved trained pipeline to: %s", model_out_path)
    return pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_dir = "data"
    df = load_and_merge_csvs(data_dir)
    df = df.drop_duplicates()
    df.to_csv("data/merged.csv", index=False)
    print("Merged CSV saved as data/merged.csv")
    print("Columns in merged DataFrame:", df.columns.tolist())
    df = prepare_target(df)

    # Diagnostic: Check for features that perfectly separate the target


    preprocessor, feature_list = build_preprocessor(df)
    train_and_evaluate(df, preprocessor)


