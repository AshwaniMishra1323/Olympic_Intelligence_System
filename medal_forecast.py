
import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.dummy import DummyClassifier


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

DATA_PATH = "data/athlete_events_cleaned.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

required = {
    "NOC", "Sport", "Games", "Year", "Season", "Medal", "ID"
}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(
    subset=["NOC", "Sport", "Games", "Year", "Season"]
).copy()

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)


# --------------------------------------------------
# 2. COUNTRY-SPORT-EDITION DATA
# --------------------------------------------------

groups = (
    df.groupby(
        ["NOC", "Sport", "Games", "Year", "Season"],
        as_index=False,
    )
    .agg(
        medal_count=("Medal", lambda x: x.notna().sum()),
        entries=("ID", "count"),
    )
)

groups["won_medal"] = (
    groups["medal_count"] > 0
).astype(int)


# --------------------------------------------------
# 3. HISTORICAL FEATURES FOR SPORT MODEL
# --------------------------------------------------

groups = groups.sort_values(
    ["NOC", "Sport", "Season", "Year", "Games"]
).copy()

history_group = groups.groupby(
    ["NOC", "Sport", "Season"],
    sort=False,
)

for lag in (1, 2, 3):
    groups[f"prev_medals_{lag}"] = (
        history_group["medal_count"].shift(lag)
    )
    groups[f"prev_entries_{lag}"] = (
        history_group["entries"].shift(lag)
    )

groups["prev_medal_rate"] = (
    groups["prev_medals_1"]
    / groups["prev_entries_1"].replace(0, float("nan"))
)

history_features = [
    "prev_medals_1",
    "prev_medals_2",
    "prev_medals_3",
    "prev_entries_1",
    "prev_entries_2",
    "prev_entries_3",
    "prev_medal_rate",
]

groups[history_features] = groups[history_features].fillna(0)

# Exclude first-ever observed appearances.
model_df = groups[
    groups["prev_entries_1"] > 0
].copy()

years = sorted(model_df["Year"].unique())

if len(years) < 5:
    raise ValueError(
        "Not enough Olympic years to train and test reliably."
    )

cutoff_index = min(
    len(years) - 1,
    max(1, int(len(years) * 0.8)),
)
cutoff_year = years[cutoff_index]

train = model_df[
    model_df["Year"] < cutoff_year
].copy()

test = model_df[
    model_df["Year"] >= cutoff_year
].copy()

if train.empty or test.empty:
    raise ValueError(
        "Chronological train/test split produced an empty set."
    )

print("\n========== DATA SUMMARY ==========")
print("Country-sport-edition rows:", len(groups))
print("Rows with historical features:", len(model_df))
print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Test begins in year:", cutoff_year)


# --------------------------------------------------
# 4. MODEL 1: TOTAL MEDAL FORECAST
# --------------------------------------------------

# Host NOC for Olympic editions present in this dataset.
# These are NOC codes, not city names.
host_noc_by_edition = {
    (1896, "Summer"): "GRE",
    (1900, "Summer"): "FRA",
    (1904, "Summer"): "USA",
    (1906, "Summer"): "GRE",
    (1908, "Summer"): "GBR",
    (1912, "Summer"): "SWE",
    (1920, "Summer"): "BEL",
    (1924, "Summer"): "FRA",
    (1924, "Winter"): "FRA",
    (1928, "Summer"): "NED",
    (1928, "Winter"): "SUI",
    (1932, "Summer"): "USA",
    (1932, "Winter"): "USA",
    (1936, "Summer"): "GER",
    (1936, "Winter"): "GER",
    (1948, "Summer"): "GBR",
    (1948, "Winter"): "SUI",
    (1952, "Summer"): "FIN",
    (1952, "Winter"): "NOR",
    (1956, "Summer"): "AUS",
    (1956, "Winter"): "ITA",
    (1960, "Summer"): "ITA",
    (1960, "Winter"): "USA",
    (1964, "Summer"): "JPN",
    (1964, "Winter"): "AUT",
    (1968, "Summer"): "MEX",
    (1968, "Winter"): "FRA",
    (1972, "Summer"): "GER",
    (1972, "Winter"): "JPN",
    (1976, "Summer"): "CAN",
    (1976, "Winter"): "AUT",
    (1980, "Summer"): "URS",
    (1980, "Winter"): "USA",
    (1984, "Summer"): "USA",
    (1984, "Winter"): "YUG",
    (1988, "Summer"): "KOR",
    (1988, "Winter"): "CAN",
    (1992, "Summer"): "ESP",
    (1992, "Winter"): "FRA",
    (1994, "Winter"): "NOR",
    (1996, "Summer"): "USA",
    (1998, "Winter"): "JPN",
    (2000, "Summer"): "AUS",
    (2002, "Winter"): "USA",
    (2004, "Summer"): "GRE",
    (2006, "Winter"): "ITA",
    (2008, "Summer"): "CHN",
    (2010, "Winter"): "CAN",
    (2012, "Summer"): "GBR",
    (2014, "Winter"): "RUS",
    (2016, "Summer"): "BRA",
}

# Aggregate medal-winning athlete-event records per country edition.
country = (
    groups.groupby(
        ["NOC", "Games", "Year", "Season"],
        as_index=False,
    )
    .agg(total_medals=("medal_count", "sum"))
)

# Host-country indicator.
country["is_host"] = country.apply(
    lambda row: int(
        host_noc_by_edition.get(
            (row["Year"], row["Season"])
        ) == row["NOC"]
    ),
    axis=1,
)

country = country.sort_values(
    ["NOC", "Season", "Year", "Games"]
).copy()

country_group = country.groupby(
    ["NOC", "Season"],
    sort=False,
)

for lag in (1, 2, 3):
    country[f"prev_total_{lag}"] = (
        country_group["total_medals"].shift(lag)
    )

country["trend"] = (
    (country["prev_total_1"] - country["prev_total_2"])
    / country["prev_total_2"].replace(0, 1)
)

country_features = [
    "NOC",
    "Season",
    "is_host",
    "prev_total_1",
    "prev_total_2",
    "prev_total_3",
    "trend",
]

country_model_df = country.dropna(
    subset=[
        "prev_total_1",
        "prev_total_2",
        "prev_total_3",
    ]
).copy()

country_years = sorted(country_model_df["Year"].unique())

if len(country_years) < 5:
    raise ValueError(
        "Not enough country-level Olympic years for evaluation."
    )

country_cut_index = min(
    len(country_years) - 1,
    max(1, int(len(country_years) * 0.8)),
)
country_cutoff = country_years[country_cut_index]

country_train = country_model_df[
    country_model_df["Year"] < country_cutoff
].copy()

country_test = country_model_df[
    country_model_df["Year"] >= country_cutoff
].copy()

if country_train.empty or country_test.empty:
    raise ValueError(
        "Country medal model needs a valid chronological split."
    )


def make_regressor(model):
    """Build a consistent preprocessing and regression pipeline."""
    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["NOC", "Season"],
        ),
        (
            "numeric",
            "passthrough",
            [
                "is_host",
                "prev_total_1",
                "prev_total_2",
                "prev_total_3",
                "trend",
            ],
        ),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# Compare candidate models on the same chronological holdout.
candidates = {
    "Random Forest (150 trees)": RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Random Forest (300 trees)": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees (200 trees)": ExtraTreesRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
best_mae = float("inf")
medal_regressor = None
best_name = None

for name, estimator in candidates.items():
    candidate = make_regressor(estimator)

    candidate.fit(
        country_train[country_features],
        country_train["total_medals"],
    )

    predictions = candidate.predict(
        country_test[country_features]
    ).clip(min=0)

    mae = mean_absolute_error(
        country_test["total_medals"],
        predictions,
    )

    results.append({
        "Model": name,
        "Test MAE": round(mae, 3),
    })

    print(f"{name} MAE: {mae:.3f}")

    if mae < best_mae:
        best_mae = mae
        medal_regressor = candidate
        best_name = name

baseline_pred = country_test["prev_total_1"].to_numpy()

baseline_mae = mean_absolute_error(
    country_test["total_medals"],
    baseline_pred,
)

print("\n========== MODEL COMPARISON ==========")
print(
    pd.DataFrame(results)
    .sort_values("Test MAE")
    .to_string(index=False)
)
print("Best model:", best_name)
print("Best model MAE:", round(best_mae, 3))
print("Previous-edition baseline MAE:", round(baseline_mae, 3))

if best_mae < baseline_mae:
    print("Result: Best tested model beats the baseline.")
else:
    print(
        "Result: Baseline still performs better. "
        "Do not claim ML is superior."
    )

joblib.dump(
    medal_regressor,
    os.path.join(MODEL_DIR, "total_medal_model.pkl"),
)

country_model_df.to_csv(
    os.path.join(MODEL_DIR, "country_medal_history.csv"),
    index=False,
)


# --------------------------------------------------
# 5. MODEL 2: SPORT-WISE MEDAL PROBABILITY
# --------------------------------------------------

sport_features = [
    "NOC",
    "Sport",
    "Season",
    "prev_medals_1",
    "prev_medals_2",
    "prev_medals_3",
    "prev_entries_1",
    "prev_entries_2",
    "prev_entries_3",
    "prev_medal_rate",
]

sport_preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        ["NOC", "Sport", "Season"],
    ),
    (
        "numeric",
        "passthrough",
        history_features,
    ),
])

sport_classifier = Pipeline([
    ("preprocessor", sport_preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )),
])

sport_classifier.fit(
    train[sport_features],
    train["won_medal"],
)


sport_pred = sport_classifier.predict(
    test[sport_features]
)

sport_accuracy = accuracy_score(
    test["won_medal"], sport_pred
)

sport_precision = precision_score(
    test["won_medal"],
    sport_pred,
    zero_division=0
)

sport_recall = recall_score(
    test["won_medal"],
    sport_pred,
    zero_division=0
)

sport_f1 = f1_score(
    test["won_medal"],
    sport_pred,
    zero_division=0
)

print("\n========== MODEL 2: SPORT-WISE PROBABILITY ==========")
print("Test cutoff year:", cutoff_year)
print(
    "Accuracy:",
    round(accuracy_score(test["won_medal"], sport_pred), 4),
)
print(
    "Precision:",
    round(
        precision_score(
            test["won_medal"], sport_pred, zero_division=0
        ),
        4,
    ),
)
print(
    "Recall:",
    round(
        recall_score(
            test["won_medal"], sport_pred, zero_division=0
        ),
        4,
    ),
)
print(
    "F1-score:",
    round(
        f1_score(
            test["won_medal"], sport_pred, zero_division=0
        ),
        4,
    ),
)

baseline_classifier = DummyClassifier(
    strategy="most_frequent"
)
baseline_classifier.fit(
    train[sport_features],
    train["won_medal"],
)

baseline_sport_pred = baseline_classifier.predict(
    test[sport_features]
)

print(
    "Baseline F1-score:",
    round(
        f1_score(
            test["won_medal"],
            baseline_sport_pred,
            zero_division=0,
        ),
        4,
    ),
)

evaluation_metrics = pd.DataFrame([{
    "Test cutoff year": cutoff_year,
    "Accuracy": sport_accuracy,
    "Precision": sport_precision,
    "Recall": sport_recall,
    "F1-score": sport_f1,
    "Baseline F1-score": f1_score(
        test["won_medal"],
        baseline_sport_pred,
        zero_division=0
    )
}])

evaluation_metrics.to_csv(
    os.path.join(MODEL_DIR, "sport_model_evaluation.csv"),
    index=False
)

print("Sport model evaluation saved.")


joblib.dump(
    sport_classifier,
    os.path.join(MODEL_DIR, "country_sport_medal_model.pkl"),
)

groups.to_csv(
    os.path.join(MODEL_DIR, "country_sport_history.csv"),
    index=False,
)


# --------------------------------------------------
# 6. COMPLETION
# --------------------------------------------------

print("\n========== SAVED FILES ==========")
print("models/total_medal_model.pkl")
print("models/country_medal_history.csv")
print("models/country_sport_medal_model.pkl")
print("models/country_sport_history.csv")
print("\nTraining complete. Review test metrics before using forecasts.")
