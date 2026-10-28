# Olympic Intelligence System

## Project Overview
The Olympic Intelligence System is a data science project that analyzes historical Olympic Games data through an interactive Streamlit dashboard. It provides insights into countries, sports, athletes, medal trends, and machine-learning-based medal predictions.

## Features
- **Overview Dashboard:** Displays key statistics and historical Olympic trends.
- **Country Analysis:** Explores medal performance by country.
- **Sport Analysis:** Examines sport-wise participation and medal records.
- **Athlete Analysis:** Analyzes athlete-related information and performance.
- **ML Predictions:** Estimates future medal totals and the probability of a country winning medals in selected sports.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## Project Structure
```text
Olympic_Intelligence_System/
├── app.py
├── main.py
├── medal_forecast.py
├── requirements.txt
├── README.md
├── data/
│   ├── athlete_events.csv
│   ├── athlete_events_cleaned.csv
│   └── noc_regions.csv
└── models/
    ├── total_medal_model.pkl
    ├── country_medal_history.csv
    ├── country_sport_medal_model.pkl
    ├── country_sport_history.csv
    └── sport_model_evaluation.csv
```

## Setup and Execution

1. Install Python.
2. Open a terminal in the project directory.
3. Install the required packages:

   `python -m pip install -r requirements.txt`

4. Launch the dashboard:

   `python -m streamlit run app.py`

5. Open the local URL displayed in the terminal.

## Machine Learning Evaluation
The project compares a Random Forest medal prediction model with a previous-edition baseline. In the current evaluation, the baseline achieved a lower MAE than the Random Forest model, so the machine-learning model should not be described as superior.

The sport medal classification model achieved an F1-score of approximately 0.693 on the current test split. Results are based on historical data and should be interpreted as estimates, not guarantees.

## Data Limitations
Historical Olympic records may contain changes in country codes, event structures, and medal-recording practices. Predictions depend on the available historical features and cannot guarantee future results.

## Future Scope
- Integrate updated Olympic data.
- Add more interactive visualizations.
- Improve prediction features and model evaluation.
- Explore additional historical trends and country comparisons.
