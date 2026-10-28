# 🏅 Olympic Intelligence System

## 1. Project Overview

Olympic Intelligence System is a Python-based Data Science and Machine Learning project designed to analyze historical Olympic data and provide insights into countries, sports, athletes, and medal distributions.

The project combines data analysis, interactive visualization, and machine learning techniques to explore Olympic trends and perform medal forecasting and sport prediction.

## 2. Objectives

- Analyze historical Olympic data.
- Explore country-wise and sport-wise medal distributions.
- Study athlete participation and performance.
- Identify historical Olympic trends.
- Develop machine learning models for medal forecasting.
- Build an interactive dashboard for data exploration.
- Demonstrate the practical application of Data Science and Machine Learning.

## 3. Features

- Interactive Olympic dashboard using Streamlit.
- Olympic data overview and statistics.
- Country-wise medal analysis.
- Sport-wise medal analysis.
- Athlete analysis.
- Historical medal trends.
- Machine learning-based medal forecasting.
- Sport prediction.
- Model evaluation using appropriate performance metrics.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Streamlit | Interactive dashboard |
| Matplotlib | Data visualization |
| Joblib | Model serialization |

## 5. Project Structure

```text
Olympic_Intelligence_System/
├── app.py
├── main.py
├── medal_forecast.py
├── ml_model.py
├── README.md
├── .gitignore
├── data/
│   ├── README.md
│   ├── requirements.txt
│   └── noc_regions.csv
└── models/
    ├── country_medal_history.csv
    ├── country_sport_history.csv
    ├── country_sport_medal_model.pkl
    ├── olympic_medal_model.pkl
    ├── olympic_preprocessor.pkl
    ├── sport_model_evaluation.csv
    └── total_medal_model.pkl
```

## 6. Dataset

The project uses historical Olympic athlete-event data containing information about:

- Athlete identification and name
- Gender
- Age, height, and weight
- Country and National Olympic Committee
- Olympic Games and year
- Season and host city
- Sport and event
- Medal information

The project also uses NOC region information for country-related analysis.

The original and cleaned athlete datasets are not included in this repository because of their file size.

## 7. Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/AshwaniMishra1323/Olympic_Intelligence_System.git
```

### Step 2: Navigate to the project directory

```bash
cd Olympic_Intelligence_System
```

### Step 3: Install dependencies

```bash
pip install -r data/requirements.txt
```

### Step 4: Prepare the dataset

Download the required Olympic athlete dataset and place the original and cleaned CSV files inside the `data` directory.

Ensure that the filenames and locations match the paths expected by the application.

## 8. Running the Application

Start the Streamlit dashboard using:

```bash
streamlit run app.py
```

The application provides the following sections:

- Overview
- Country Analysis
- Sport Analysis
- Athlete Analysis
- ML Predictions

## 9. Machine Learning

The project includes machine learning components for medal forecasting and sport prediction.

### Medal Forecasting

The medal forecasting component uses historical Olympic information to estimate medal outcomes.

The models are evaluated using Mean Absolute Error (MAE).

A previous-edition baseline is also used to compare forecasting performance.

### Sport Prediction

The sport prediction component uses machine learning to classify sport-related outcomes.

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

### Model Evaluation

The forecasting experiments showed that the previous-edition baseline achieved lower MAE than the evaluated Random Forest and Extra Trees models.

Therefore, the current medal forecasting models require further improvement before they can be considered more accurate than the baseline.

The sport prediction model achieved an F1-score of approximately 0.6925 in the reported evaluation.

These results represent the current experimental performance and may change with improvements to the dataset, features, and model configuration.

## 10. Testing

The project was tested through the following activities:

- Python syntax compilation.
- Application execution using Streamlit.
- Dashboard navigation.
- Country analysis functionality.
- Sport analysis functionality.
- Athlete analysis functionality.
- Machine learning prediction functionality.
- Model evaluation and output verification.

The Python files were checked using:

```bash
python -m py_compile app.py main.py medal_forecast.py ml_model.py
```

The Streamlit application was also executed to verify the dashboard functionality.

## 11. Limitations

The current project has the following limitations:

- Medal forecasting depends on historical Olympic data.
- Future Olympic performance cannot be predicted with certainty.
- Historical trends may not represent future sporting outcomes.
- Model performance depends on the quality and availability of data.
- Some athlete records contain missing information.
- The medal forecasting models currently do not outperform the previous-edition baseline.
- The application requires the dataset files to be available locally.
- Sport prediction performance may vary depending on the data and classification task.

## 12. Future Scope

The Olympic Intelligence System can be enhanced in several ways:

### 12.1 Advanced Machine Learning Models

Future work can explore additional machine learning algorithms, ensemble methods, and improved model configurations to improve forecasting performance.

### 12.2 Feature Engineering

Additional historical features can be developed, including:

- Previous Olympic medal performance.
- Country-wise historical trends.
- Sport-specific medal patterns.
- Athlete participation trends.
- Historical performance changes across Olympic editions.

### 12.3 Improved Medal Forecasting

Future versions can investigate more reliable forecasting approaches and compare them against historical baselines using consistent evaluation methods.

### 12.4 Real-Time Data Integration

The project can be extended to incorporate updated Olympic information and results when reliable data sources are available.

### 12.5 Advanced Data Visualization

Additional visualizations can be introduced to explore:

- Long-term medal trends.
- Country performance comparisons.
- Sport participation patterns.
- Athlete performance history.
- Changes across Olympic editions.

### 12.6 Interactive Prediction Features

The dashboard can be enhanced with additional prediction controls, configurable inputs, and more detailed explanations of model outputs.

### 12.7 Deployment

The application can be deployed on a cloud platform to make the dashboard accessible through a web browser without requiring local installation.

### 12.8 Explainable Machine Learning

Future development can include feature importance analysis and explainability techniques to help users understand which historical factors influence model predictions.

## 13. Conclusion

Olympic Intelligence System demonstrates the application of Data Science and Machine Learning techniques to historical Olympic data.

The project provides an interactive dashboard for exploring Olympic statistics, analyzing country and sport performance, studying athlete information, and experimenting with machine learning-based predictions.

The project also highlights the importance of model evaluation and baseline comparison in forecasting tasks.

Further development can improve prediction performance, visualization capabilities, and the overall usefulness of the system.

## 14. Author

**Ashwani Mishra**

B.Tech – Computer Science and Engineering (Data Science)

IPS Academy, Institute of Engineering & Science, Indore

## 15. Repository

GitHub Repository:

https://github.com/AshwaniMishra1323/Olympic_Intelligence_System