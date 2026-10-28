
import pandas as pd

# Load the Olympic dataset
df = pd.read_csv("data/athlete_events.csv")

# Basic dataset information
print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nFirst 5 Rows:")
print(df.head())

# Inspect duplicate rows
duplicate_rows = df[df.duplicated(keep=False)]

print("\nDuplicate Rows Preview:")
print(duplicate_rows.head(10))

print("\nTotal Duplicate Rows Including Original Copies:")
print(len(duplicate_rows))


# Remove exact duplicate rows
df_clean = df.drop_duplicates()

print("\nOriginal Dataset Shape:", df.shape)
print("Cleaned Dataset Shape:", df_clean.shape)

print("\nRemaining Duplicate Rows:")
print(df_clean.duplicated().sum())

# Missing value analysis on cleaned dataset
missing_count = df_clean.isnull().sum()

missing_percentage = (
    missing_count / len(df_clean) * 100
)

missing_summary = pd.DataFrame({
    "Missing Values": missing_count,
    "Missing Percentage": missing_percentage
})

missing_summary = missing_summary[
    missing_summary["Missing Values"] > 0
].sort_values(by="Missing Percentage", ascending=False)

print("\nMissing Value Summary:")
print(missing_summary)
# Statistical summary of numerical columns
print("\nNumerical Data Summary:")

print(
    df_clean[["Age", "Height", "Weight"]].describe()
)

print("\nMedian Values:")
print(
    df_clean[["Age", "Height", "Weight"]].median()
)

# Missing Value Imputation using Median
df_clean["Age"] = df_clean["Age"].fillna(
    df_clean["Age"].median()
)

df_clean["Height"] = df_clean["Height"].fillna(
    df_clean["Height"].median()
)

df_clean["Weight"] = df_clean["Weight"].fillna(
    df_clean["Weight"].median()
)

# Verify missing values after imputation
print("\nMissing Values After Imputation:")

print(
    df_clean[["Age", "Height", "Weight"]].isnull().sum()
)

# Save cleaned dataset
df_clean.to_csv(
    "data/athlete_events_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
print("Final Dataset Shape:", df_clean.shape)

# Exploratory Data Analysis (EDA)

# 1. Total medal records
medal_data = df_clean.dropna(subset=["Medal"])

print("\nTotal Medal Records:")
print(len(medal_data))


# 2. Medal type distribution
print("\nMedal Type Distribution:")
print(medal_data["Medal"].value_counts())


# 3. Top 10 countries by medal records
top_countries = (
    medal_data["NOC"]
    .value_counts()
    .head(10)
)

print("\nTop 10 Countries by Medal Records:")
print(top_countries)


# 4. India medal performance
india_medals = medal_data[medal_data["NOC"] == "IND"]

print("\nIndia Medal Distribution:")
print(india_medals["Medal"].value_counts())

import matplotlib.pyplot as plt

# 1. Medal Type Distribution
medal_counts = medal_data["Medal"].value_counts()

plt.figure(figsize=(8, 5))
medal_counts.plot(kind="bar")

plt.title("Olympic Medal Records by Medal Type")
plt.xlabel("Medal Type")
plt.ylabel("Number of Medal Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 2. Top 10 Countries by Medal Records
plt.figure(figsize=(10, 6))
top_countries.plot(kind="bar")

plt.title("Top 10 Countries by Olympic Medal Records")
plt.xlabel("Country NOC")
plt.ylabel("Number of Medal Records")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# 3. Year-wise Olympic Medal Records

yearly_medals = medal_data.groupby("Year")["Medal"].count()

print("\nYear-wise Olympic Medal Records:")
print(yearly_medals)
# 4. Year-wise Olympic Medal Records Chart

plt.figure(figsize=(12, 6))

yearly_medals.plot(kind="line", marker="o")

plt.title("Year-wise Olympic Medal Records")
plt.xlabel("Olympic Year")
plt.ylabel("Number of Medal Records")
plt.grid(True)
plt.tight_layout()

plt.show()
# 5. India Year-wise Medal Records

india_yearly_medals = (
    india_medals.groupby(["Season", "Year"])["Medal"]
    .count()
)

print("\nIndia Year-wise Medal Records:")
print(india_yearly_medals)
# 6. India Summer Olympics Medal Trend

india_summer_medals = (
    india_medals[india_medals["Season"] == "Summer"]
    .groupby("Year")["Medal"]
    .count()
)

plt.figure(figsize=(12, 6))

india_summer_medals.plot(kind="line", marker="o")

plt.title("India's Summer Olympics Medal Records")
plt.xlabel("Olympic Year")
plt.ylabel("Number of Medal Records")
plt.grid(True)
plt.tight_layout()

plt.show()
# 7. India Medal Records by Sport

india_sport_medals = (
    india_medals.groupby("Sport")["Medal"]
    .count()
    .sort_values(ascending=False)
)

print("\nIndia Medal Records by Sport:")
print(india_sport_medals)

print("\nTop 10 Sports for India:")
print(india_sport_medals.head(10))

import pandas as pd

file_path = "data/athlete_events_cleaned.csv"
df = pd.read_csv(file_path)

print("\n========== CLEANED DATA AUDIT ==========")
print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nColumns:")
print(df.columns.tolist())
