import pandas as pd

# 1. Load Cleaned Dataset
df = pd.read_csv("data/athlete_events_cleaned.csv")

# 2. Create Target Column
# 1 = Medal Won, 0 = No Medal
df["Won_Medal"] = df["Medal"].notna().astype(int)

# 3. Select Input Features
features = [
    "Sex",
    "Age",
    "Height",
    "Weight",
    "Sport",
    "Season",
    "NOC"
]

X = df[features]
y = df["Won_Medal"]

# 4. Check Dataset
print("Feature Dataset Shape:", X.shape)
print("\nTarget Distribution:")
print(y.value_counts())

print("\nInput Features:")
print(X.head())
# 5. Split Data Based on Olympic Year

train_data = df[df["Year"] <= 2008]
test_data = df[df["Year"] >= 2010]

X_train = train_data[features]
y_train = train_data["Won_Medal"]

X_test = test_data[features]
y_test = test_data["Won_Medal"]

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Years:")
print(train_data["Year"].min(), "to", train_data["Year"].max())

print("\nTesting Years:")
print(test_data["Year"].min(), "to", test_data["Year"].max())
# 6. Encode Categorical Features

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_features = [
    "Sex",
    "Sport",
    "Season",
    "NOC"
]

numerical_features = [
    "Age",
    "Height",
    "Weight"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print("\nEncoded Training Shape:", X_train_encoded.shape)
print("Encoded Testing Shape:", X_test_encoded.shape)


# 7. Train Improved Random Forest Model

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=3,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Random Forest Model...", flush=True)
model.fit(X_train_encoded, y_train)
print("Random Forest Model Training Completed!", flush=True)


# 8. Make Predictions

y_pred = model.predict(X_test_encoded)

print("\nPredictions Completed!")
print("Total Predictions:", len(y_pred))

print("\nFirst 20 Predictions:")
print(y_pred[:20])
# 9. Evaluate Random Forest Model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nModel Evaluation Results:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 10. Compare Different Prediction Thresholds

y_prob = model.predict_proba(X_test_encoded)[:, 1]

print("\nThreshold Comparison:")

for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
    y_threshold_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_test, y_threshold_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_threshold_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_threshold_pred, zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )


# 11. Compare Scaled Logistic Regression Model

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

lr_model = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ))
])

print("\nTraining Scaled Logistic Regression Model...", flush=True)

lr_model.fit(X_train_encoded, y_train)

print("Scaled Logistic Regression Training Completed!", flush=True)

lr_pred = lr_model.predict(X_test_encoded)

print("\nScaled Logistic Regression Evaluation:")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred, zero_division=0))
print("Recall:", recall_score(y_test, lr_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, lr_pred, zero_division=0))

# Save trained model and preprocessor

import os
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/olympic_medal_model.pkl")
joblib.dump(preprocessor, "models/olympic_preprocessor.pkl")

print("Model and preprocessor saved successfully!")
