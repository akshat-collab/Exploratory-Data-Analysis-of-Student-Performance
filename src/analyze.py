"""Run EDA and a leakage-aware baseline model on UCI student performance data."""
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/student-mat.csv"
OUT = ROOT / "reports"
FIG = ROOT / "figures"
TARGET = "G3"


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df = pd.read_csv(DATA, sep=";")
    # Save descriptive statistics for all numeric and categorical variables.
    numeric = df.select_dtypes(include=np.number)
    numeric.describe().T.to_csv(OUT / "descriptive_statistics.csv")
    df.describe(include="all").T.to_csv(OUT / "all_variable_summary.csv")
    missing = df.isna().sum().sort_values(ascending=False)
    pd.DataFrame({"missing_count": missing, "missing_percent": missing / len(df) * 100}).to_csv(OUT / "missing_values.csv")

    sns.set_theme(style="whitegrid", palette="deep")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET], bins=np.arange(-0.5, 21.5, 1), kde=True, ax=ax, color="#2878B5")
    ax.set(title="Final grade (G3) distribution", xlabel="Final grade (0–20)", ylabel="Students")
    fig.tight_layout(); fig.savefig(FIG / "grade_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df[["G1", "G2", "G3"]], palette="Set2", ax=ax)
    ax.set(title="Grade distributions across periods", ylabel="Grade (0–20)")
    fig.tight_layout(); fig.savefig(FIG / "grade_boxplots.png", dpi=160); plt.close(fig)

    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax, square=True, linewidths=.25)
    ax.set_title("Numeric variable correlations")
    fig.tight_layout(); fig.savefig(FIG / "correlation_heatmap.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="absences", y=TARGET, alpha=.65, ax=ax, color="#E07A5F")
    ax.set(title="Absences vs final grade", xlabel="Absences", ylabel="Final grade (0–20)")
    fig.tight_layout(); fig.savefig(FIG / "absences_vs_grade.png", dpi=160); plt.close(fig)

    # IQR flags for every numeric variable; preserve observations in model/EDA.
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    iqr = q3 - q1
    flags = ((numeric.lt(q1 - 1.5 * iqr)) | (numeric.gt(q3 + 1.5 * iqr)))
    outlier_summary = pd.DataFrame({"outlier_count": flags.sum(), "outlier_percent": flags.mean() * 100})
    outlier_summary.to_csv(OUT / "outlier_summary_iqr.csv")
    outlier_rows = df.loc[flags.any(axis=1)].copy()
    outlier_rows.to_csv(OUT / "outlier_rows_iqr.csv", index=False)

    # G1/G2 are excluded: they are strong proxies for G3 and unavailable in an early-risk setting.
    X = df.drop(columns=[TARGET, "G1", "G2"])
    y = df[TARGET]
    cats = X.select_dtypes(exclude=np.number).columns.tolist()
    nums = X.select_dtypes(include=np.number).columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), nums),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats),
    ])
    model = Pipeline([("prep", prep), ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cv_mae = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=1).mean()
    metrics = {
        "dataset": "UCI Student Performance - Mathematics",
        "rows": int(len(df)), "features_used": int(X.shape[1]),
        "target": TARGET, "target_scale": "0-20",
        "model": "RandomForestRegressor (400 trees, min_samples_leaf=2)",
        "split": "80/20 random_state=42", "cross_validation": "5-fold",
        "test_mae": float(mean_absolute_error(y_test, pred)),
        "test_rmse": float(mean_squared_error(y_test, pred) ** .5),
        "test_r2": float(r2_score(y_test, pred)), "cv_mae": float(cv_mae),
        "excluded_features": ["G1", "G2"],
        "interpretation": "Predictive association, not causal effect; small single-school-context dataset.",
    }
    (OUT / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"actual_G3": y_test.values, "predicted_G3": pred}).to_csv(OUT / "predictions.csv", index=False)
    # Feature importance after fitting the preprocessing pipeline.
    feature_names = model.named_steps["prep"].get_feature_names_out()
    importance = model.named_steps["model"].feature_importances_
    pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)
    print(json.dumps({"rows": len(df), "columns": len(df.columns), "g3_mean": float(y.mean()), "g3_median": float(y.median()), "g3_std": float(y.std()), "zero_final_grades": int((y == 0).sum()), "absences_iqr_outliers": int(flags["absences"].sum()), "model": metrics}, indent=2))

if __name__ == "__main__":
    main()
