import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. CONFIGURATION
# ============================================================

GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

RANDOM_STATE = 42

# ============================================================
# 2. LOAD AND CLEAN DATA
# ============================================================

print("Loading data from price.json...")
df = pd.read_json("price.json")
df.columns = df.columns.str.strip().str.lower()

required = ["area", "location", "bedrooms", "price"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column in price.json: {col}")

df = df[required].copy()

df["area"] = pd.to_numeric(df["area"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

df["bedrooms"] = pd.to_numeric(
    df["bedrooms"].astype(str).str.extract(r"(\d+)", expand=False),
    errors="coerce"
)

df["location"] = (
    df["location"]
    .astype("string")
    .str.strip()
    .replace(["", "nan", "None", "<NA>"], pd.NA)
)

df.dropna(subset=required, inplace=True)

df = df[
    (df["area"] > 0) &
    (df["bedrooms"] > 0) &
    (df["price"] > 0)
].copy()

if len(df) < 50:
    raise ValueError("Not enough valid rows for model training and optimization.")

print(f"Data cleaned successfully. Total valid records: {len(df)}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

df["area_per_bedroom"] = df["area"] / df["bedrooms"]

X = df[["area", "bedrooms", "area_per_bedroom", "location"]].copy()
y = df["price"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

numeric_features = ["area", "bedrooms", "area_per_bedroom"]
categorical_features = ["location"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            numeric_features
        ),
        (
            "location",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ============================================================
# 4. MODEL HYPERPARAMETER SEARCH SPACES
# ============================================================

models = [
    (
        "Random Forest",
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        {
            "model__regressor__n_estimators": [150, 250, 400],
            "model__regressor__max_depth": [None, 10, 20],
            "model__regressor__min_samples_leaf": [1, 2, 4],
            "model__regressor__min_samples_split": [2, 5, 10],
            "model__regressor__max_features": [0.6, 0.8, 1.0]
        }
    ),
    (
        "Extra Trees",
        ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1),
        {
            "model__regressor__n_estimators": [150, 250, 400],
            "model__regressor__max_depth": [None, 10, 20],
            "model__regressor__min_samples_leaf": [1, 2, 4],
            "model__regressor__min_samples_split": [2, 5, 10],
            "model__regressor__max_features": [0.6, 0.8, 1.0]
        }
    ),
    (
        "Gradient Boosting",
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {
            "model__regressor__n_estimators": [100, 200],
            "model__regressor__learning_rate": [0.03, 0.07, 0.1],
            "model__regressor__max_depth": [2, 3, 4],
            "model__regressor__min_samples_leaf": [2, 5],
            "model__regressor__loss": ["squared_error", "huber"]
        }
    )
]

# ============================================================
# 5. BASELINE EVALUATION
# ============================================================

baseline_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DummyRegressor(strategy="mean"))
])
baseline_pipeline.fit(X_train, y_train)
baseline_pred = baseline_pipeline.predict(X_test)
print("Baseline Dummy R²:", round(r2_score(y_test, baseline_pred), 4))

# ============================================================
# 6. TUNE MODELS WITH CROSS-VALIDATION
# ============================================================

best_name = None
best_pipeline = None
best_cv_r2 = -np.inf
comparison = []

for name, estimator, params in models:
    print(f"\nOptimizing {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            TransformedTargetRegressor(
                regressor=estimator,
                func=np.log1p,
                inverse_func=np.expm1
            )
        )
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        n_iter=15,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        error_score="raise"
    )

    search.fit(X_train, y_train)

    print(f"Best CV R² for {name}: {round(search.best_score_, 4)}")
    comparison.append({"Model": name, "CV R2": search.best_score_})

    if search.best_score_ > best_cv_r2:
        best_cv_r2 = search.best_score_
        best_name = name
        best_pipeline = search.best_estimator_

# ============================================================
# 7. FINAL TEST EVALUATION
# ============================================================

y_pred = best_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test.to_numpy() - y_pred) / y_test.to_numpy())) * 100

metrics = {
    "model_name": best_name,
    "cv_r2": float(best_cv_r2),
    "mae": float(mae),
    "mse": float(mse),
    "rmse": float(rmse),
    "r2": float(r2),
    "mape": float(mape),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test))
}

print("\n========== BEST MODEL ==========")
print(f"Selected Model: {best_name}")
print(f"Cross-validation R²: {round(best_cv_r2, 4)}")
print(f"Test R²: {round(r2, 4)}")
print(f"MAE: {round(mae, 2)}")
print(f"RMSE: {round(rmse, 2)}")
print(f"MAPE: {round(mape, 2)}%")

# ============================================================
# 8. SAVE MODEL & METADATA (CRITICAL FIX FOR APP.PY)
# ============================================================

print("\nSaving model files...")

with open("house_price_model.pkl", "wb") as file:
    pickle.dump(best_pipeline, file)

model_columns = ["area", "bedrooms", "area_per_bedroom", "location"]
with open("model_columns.pkl", "wb") as file:
    pickle.dump(model_columns, file)

with open("model_metrics.pkl", "wb") as file:
    pickle.dump(metrics, file)

# Save unique locations for app.py dropdown
unique_locations = sorted(df["location"].dropna().unique().tolist())
with open("locations.pkl", "wb") as file:
    pickle.dump(unique_locations, file)

print("Saved: house_price_model.pkl, model_columns.pkl, model_metrics.pkl, locations.pkl")

# ============================================================
# 9. GENERATE AND SAVE ALL VISUALIZATION GRAPHS
# ============================================================

print("\nGenerating visualization graphs...")

results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred
})
results["Residual"] = results["Actual"] - results["Predicted"]

# 1. Model comparison
comparison_df = pd.DataFrame(comparison).sort_values("CV R2", ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(data=comparison_df, x="CV R2", y="Model", hue="Model", palette="viridis", legend=False, ax=ax)
ax.axvline(0.70, color="red", linestyle="--", label="Target R² = 0.70")
ax.set_title("Model Performance Comparison")
ax.set_xlabel("Cross-Validation R²")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(GRAPH_DIR, "model_comparison.png"), dpi=180)
plt.close(fig)

# 2. Actual vs Predicted
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=results, x="Actual", y="Predicted", alpha=0.6, ax=ax)
low = min(results["Actual"].min(), results["Predicted"].min())
high = max(results["Actual"].max(), results["Predicted"].max())
ax.plot([low, high], [low, high], color="red", linestyle="--")
ax.set_title(f"Actual vs Predicted ({best_name})")
fig.tight_layout()
fig.savefig(os.path.join(GRAPH_DIR, "actual_vs_predicted.png"), dpi=180)
plt.close(fig)

# 3. Residual Plot
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=results, x="Predicted", y="Residual", alpha=0.6, color="purple", ax=ax)
ax.axhline(0, color="red", linestyle="--")
ax.set_title("Prediction Error (Residuals)")
ax.set_xlabel("Predicted Price")
ax.set_ylabel("Residual (Actual - Predicted)")
fig.tight_layout()
fig.savefig(os.path.join(GRAPH_DIR, "residual_plot.png"), dpi=180)
plt.close(fig)

# 4. Error distribution
fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(results["Residual"], kde=True, color="coral", ax=ax)
ax.set_title("Prediction Error Distribution")
ax.set_xlabel("Error")
fig.tight_layout()
fig.savefig(os.path.join(GRAPH_DIR, "error_distribution.png"), dpi=180)
plt.close(fig)

# 5. Price distribution
fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(df["price"], kde=True, color="teal", ax=ax)
ax.set_title("House Price Distribution")
ax.set_xlabel("Price (Lakhs)")
fig.tight_layout()
fig.savefig(os.path.join(GRAPH_DIR, "price_distribution.png"), dpi=180)
plt.close(fig)

# 6 & 7. Covariance and Correlation heatmaps
numeric_df = df[["area", "bedrooms", "price"]]
for matrix, filename, title, cmap in [
    (numeric_df.cov(), "covariance_heatmap.png", "Covariance Matrix Heatmap", "YlGnBu"),
    (numeric_df.corr(), "correlation_heatmap.png", "Correlation Heatmap", "coolwarm")
]:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, linewidths=1, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, filename), dpi=180)
    plt.close(fig)

# 8. Feature Importance
try:
    inner_regressor = best_pipeline.named_steps["model"].regressor_
    if hasattr(inner_regressor, "feature_importances_"):
        ohe = best_pipeline.named_steps["preprocessor"].named_transformers_["location"]
        loc_feature_names = ohe.get_feature_names_out(["location"]).tolist()
        all_feature_names = numeric_features + loc_feature_names
        importances = inner_regressor.feature_importances_
        
        fi_df = pd.DataFrame({
            "Feature": all_feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=fi_df, x="Importance", y="Feature", palette="magma", hue="Feature", legend=False, ax=ax)
        ax.set_title("Top 15 Feature Importances")
        fig.tight_layout()
        fig.savefig(os.path.join(GRAPH_DIR, "feature_importance.png"), dpi=180)
        plt.close(fig)
except Exception:
    pass

# 9. Location comparison (Top 12 frequent locations)
top_locations = df["location"].value_counts().nlargest(12).index
df_top_loc = df[df["location"].isin(top_locations)]
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(
    data=df_top_loc,
    x="price",
    y="location",
    estimator=np.median,
    errorbar=None,
    palette="Blues_r",
    hue="location",
    legend=False,
    ax=ax
)
ax.set_title("Median House Price by Top Locations")
ax.set_xlabel("Median Price (Lakhs)")
fig.tight_layout()
fig.savefig(os.path.join(GRAPH_DIR, "location_comparison.png"), dpi=180)
plt.close(fig)

print("All graphs successfully generated in the 'graphs/' folder.")
print("Training process finished.")