import os
import joblib
import streamlit as st
import pandas as pd

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Olympic Intelligence System",
    page_icon="🏅",
    layout="wide"
)

st.title("🏅 Olympic Intelligence System")
st.write(
    "Explore Olympic history, country performance, sports, athletes "
    "and machine-learning-based medal forecasts."
)

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv("data/athlete_events_cleaned.csv")


@st.cache_data
def load_history(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


athlete_df = load_data()
athlete_df["Year"] = pd.to_numeric(athlete_df["Year"], errors="coerce")

# ==========================================
# SUMMARY METRICS
# ==========================================
total_athletes = athlete_df["ID"].nunique()
total_editions = athlete_df["Games"].nunique()
total_sports = athlete_df["Sport"].nunique()
total_medals = athlete_df["Medal"].notna().sum()
total_countries = athlete_df["NOC"].nunique()

# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.title("Olympic Filters")

countries = sorted(athlete_df["NOC"].dropna().unique())

selected_country = st.sidebar.selectbox(
    "Select Country",
    countries,
    index=countries.index("IND") if "IND" in countries else 0
)

country_df = athlete_df[athlete_df["NOC"] == selected_country]
years = sorted(country_df["Year"].dropna().unique())

selected_year = st.sidebar.selectbox(
    "Select Olympic Year",
    years,
    index=len(years) - 1
)

year_df = country_df[country_df["Year"] == selected_year]
available_sports = sorted(year_df["Sport"].dropna().unique())

selected_sport = (
    st.sidebar.selectbox("Select Sport", available_sports)
    if available_sports else None
)

# ==========================================
# NAVIGATION
# ==========================================
overview_tab, country_tab, sport_tab, athlete_tab, ml_tab = st.tabs([
    "🏅 Overview",
    "🌍 Country Analysis",
    "🏆 Sport Analysis",
    "👤 Athlete Analysis",
    "🤖 ML Predictions"
])

medal_types = ["Gold", "Silver", "Bronze"]

# ==========================================
# TAB 1: OVERVIEW
# ==========================================
with overview_tab:
    st.subheader("Olympic Games Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Unique Athletes", f"{total_athletes:,}")
    c2.metric("Olympic Editions", f"{total_editions:,}")
    c3.metric("Sports", f"{total_sports:,}")

    c4, c5 = st.columns(2)
    c4.metric("Medal Records", f"{total_medals:,}")
    c5.metric("NOC Codes", f"{total_countries:,}")

    st.subheader("Dataset Preview")
    st.dataframe(athlete_df.head(10), use_container_width=True)

    st.caption(
        "A medal record represents an athlete-event record and may not "
        "equal one unique physical medal."
    )

# ==========================================
# TAB 2: COUNTRY ANALYSIS
# ==========================================
with country_tab:
    st.subheader(f"Country Analysis: {selected_country}")

    country_medals = (
        country_df["Medal"]
        .value_counts()
        .reindex(medal_types, fill_value=0)
    )

    st.write("Historical medal records")
    st.bar_chart(country_medals)

    st.subheader(f"Performance in {selected_year}")

    year_medals = (
        year_df["Medal"]
        .value_counts()
        .reindex(medal_types, fill_value=0)
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("🥇 Gold", int(year_medals["Gold"]))
    c2.metric("🥈 Silver", int(year_medals["Silver"]))
    c3.metric("🥉 Bronze", int(year_medals["Bronze"]))

    st.bar_chart(year_medals)

    st.subheader("Medal Trend Over the Years")

    medal_trend = (
        country_df[country_df["Medal"].notna()]
        .groupby("Year")
        .size()
        .sort_index()
    )

    if not medal_trend.empty:
        st.line_chart(medal_trend)
    else:
        st.info("No medal records available for this country.")

    st.subheader("Medal Records by Gender")

    gender_medals = (
        year_df[year_df["Medal"].notna()]
        .groupby("Sex")
        .size()
    )

    if not gender_medals.empty:
        st.bar_chart(gender_medals)
    else:
        st.info("No medal records for this selection.")

    st.subheader("Country Comparison")

    default_countries = [c for c in ["IND", "USA"] if c in countries]

    compare_countries = st.multiselect(
        "Choose countries",
        countries,
        default=default_countries
    )

    if len(compare_countries) >= 2:
        comparison_df = athlete_df[
            athlete_df["NOC"].isin(compare_countries)
            & athlete_df["Medal"].notna()
        ]

        comparison = (
            comparison_df.groupby(["NOC", "Medal"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=medal_types, fill_value=0)
        )

        st.bar_chart(comparison)
        st.dataframe(comparison, use_container_width=True)
    else:
        st.info("Select at least two countries.")

# ==========================================
# TAB 3: SPORT ANALYSIS
# ==========================================
with sport_tab:
    st.subheader("Sport-wise Medal Analysis")

    if selected_sport is not None:
        sport_df = year_df[
            (year_df["Sport"] == selected_sport)
            & year_df["Medal"].notna()
        ]

        st.write(
            f"Country: {selected_country} | "
            f"Year: {selected_year} | Sport: {selected_sport}"
        )

        sport_medals = (
            sport_df["Medal"]
            .value_counts()
            .reindex(medal_types, fill_value=0)
        )

        st.bar_chart(sport_medals)

        st.subheader("Medal-Winning Athlete Records")
        st.dataframe(
            sport_df[["Name", "Sex", "Age", "Event", "Medal"]],
            use_container_width=True
        )

        if sport_df.empty:
            st.info("No medal records for this selection.")
    else:
        st.info("No sports available for this selection.")

# ==========================================
# TAB 4: ATHLETE ANALYSIS
# ==========================================
with athlete_tab:
    st.subheader(f"Athlete Records: {selected_country} - {selected_year}")

    medal_athletes = year_df[
        year_df["Medal"].notna()
    ][["Name", "Sex", "Age", "Sport", "Event", "Medal"]]

    st.metric("Medal-Winning Records", len(medal_athletes))
    st.dataframe(medal_athletes, use_container_width=True)

    st.subheader("Search Athlete")
    search_name = st.text_input("Enter athlete name")

    if search_name.strip():
        matching = year_df[
            year_df["Name"].str.contains(
                search_name.strip(),
                case=False,
                na=False
            )
        ]

        st.dataframe(
            matching[["Name", "Sex", "Age", "Sport", "Event", "Medal"]],
            use_container_width=True
        )

# =========================
# ML PREDICTIONS
# =========================

with ml_tab:
    st.header("Olympic Medal Predictions")
    st.write(
        "Explore historical trends and machine-learning estimates "
        "for a selected country, Olympic season, and sport."
    )

    forecast_tab, probability_tab = st.tabs([
        "Total Medal Forecast",
        "Sport-wise Medal Probability"
    ])

    # ---------------------------------
    # 1. TOTAL MEDAL FORECAST
    # ---------------------------------
    with forecast_tab:
        st.subheader("Total Medal Forecast")

        total_model_path = "models/total_medal_model.pkl"
        country_history_path = "models/country_medal_history.csv"

        if os.path.exists(total_model_path) and os.path.exists(country_history_path):
            total_model = joblib.load(total_model_path)
            country_history = pd.read_csv(country_history_path)

            forecast_country = st.selectbox(
                "Select country (NOC)",
                sorted(country_history["NOC"].dropna().unique()),
                key="forecast_country"
            )

            forecast_season = st.selectbox(
                "Select Olympic season",
                sorted(country_history["Season"].dropna().unique()),
                key="forecast_season"
            )

            hist = country_history[
                (country_history["NOC"] == forecast_country)
                & (country_history["Season"] == forecast_season)
            ].sort_values("Year")

            if len(hist) < 3:
                st.warning(
                    "Not enough historical records to generate this forecast. "
                    "At least three historical editions are required."
                )
            else:
                latest = float(hist.iloc[-1]["total_medals"])
                latest_year = int(hist.iloc[-1]["Year"])
                
                # Find the next scheduled Olympic edition for the selected season
                scheduled_years = {
                "Summer": [2028, 2032, 2036],
                "Winter": [2030, 2034, 2038]
            }

                future_years = [
                    year for year in scheduled_years[forecast_season]
                    if year > latest_year
                ]

                forecast_year = (
                    min(future_years) if future_years
                    else latest_year + (4 if forecast_season == "Summer" else 4)
                )

                # Host countries for editions after the current dataset
                host_noc_by_edition = {
                    (2018, "Winter"): "KOR",
                    (2020, "Summer"): "JPN",
                    (2022, "Winter"): "CHN",
                    (2024, "Summer"): "FRA",
                    (2026, "Winter"): "ITA",
                    (2028, "Summer"): "USA",
                    (2030, "Winter"): "FRA",
                    (2032, "Summer"): "AUS",
                }
                second = float(hist.iloc[-2]["total_medals"])
                third = float(hist.iloc[-3]["total_medals"])

                trend = (latest - second) / (second if second != 0 else 1)

                features = pd.DataFrame([{
                    "NOC": forecast_country,
                    "Season": forecast_season,
                    "is_host": int(
                        host_noc_by_edition.get(
                            (forecast_year, forecast_season)
                        ) == forecast_country
                    ),
                    "prev_total_1": latest,
                    "prev_total_2": second,
                    "prev_total_3": third,
                    "trend": trend
                }])

                prediction = max(0, float(total_model.predict(features)[0]))
                
                host_country = host_noc_by_edition.get(
                    (forecast_year, forecast_season),
                    "Not mapped"
                )

                is_host = int(host_country == forecast_country)

                st.markdown("#### Forecast Details")

                detail1, detail2, detail3 = st.columns(3)

                detail1.metric("Forecast Year", forecast_year)
                detail2.metric("Host Country (NOC)", host_country)
                detail3.metric("Host Feature (is_host)", is_host)

                st.markdown(
                f"### {forecast_country} — Estimated {forecast_season} "
                f"{forecast_year} Olympics"
                      )
                st.caption(
                    "The model estimate is based on historical medal-record totals. "
                    "It is an estimate, not a guaranteed result."
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Latest historical total",
                        f"{latest:.0f}",
                        help="Total medal-winning athlete-event records in the latest available edition."
                    )

                with col2:
                    st.metric(
                        "Random Forest estimate",
                        f"{prediction:.1f}",
                        delta=f"{prediction - latest:+.1f} vs. latest historical total"
                    )

                if prediction > latest:
                    interpretation = (
                        f"The model estimates approximately {prediction:.1f} medal records, "
                        f"which is {prediction - latest:.1f} higher than the latest historical total."
                    )
                elif prediction < latest:
                    interpretation = (
                        f"The model estimates approximately {prediction:.1f} medal records, "
                        f"which is {latest - prediction:.1f} lower than the latest historical total."
                    )
                else:
                    interpretation = (
                        "The model estimate is equal to the latest historical total."
                    )

                st.info("**What does this mean?** " + interpretation)

               
                # Actual vs Predicted: Historical Backtest
                st.markdown("#### Actual vs Predicted Medal Records")

                comparison_rows = []

                # Generate test-period predictions using earlier editions only
                for i in range(3, len(hist)):
                    target_row = hist.iloc[i]
                    target_year = int(target_row["Year"])

                    # Model evaluation uses a 2004 test-period cutoff
                    if target_year < 2004:
                        continue

                    previous = hist.iloc[i-3:i].sort_values(
                        ["Year", "Games"], ascending=False
                    )

                    totals = previous["total_medals"].astype(float).tolist()
                    latest, second, third = totals

                    input_features = pd.DataFrame([{
                        "NOC": forecast_country,
                        "Season": forecast_season,
                        "prev_total_1": latest,
                        "prev_total_2": second,
                        "prev_total_3": third,
                        "trend": (latest - second) / (second if second != 0 else 1),
                        "is_host": int(target_row.get("is_host", 0))
                    }])

                    predicted = max(
                        0.0,
                        float(total_model.predict(input_features)[0])
                    )
                    actual = float(target_row["total_medals"])
                    baseline = max(0.0, latest)

                    comparison_rows.append({
                        "Year": target_year,
                        "Actual": actual,
                        "Random Forest": predicted,
                        "Previous-edition baseline": baseline,
                        "RF absolute error": abs(actual - predicted),
                        "Baseline absolute error": abs(actual - baseline)
                    })

                if comparison_rows:
                    comparison_df = pd.DataFrame(comparison_rows)
                    comparison_df = comparison_df.sort_values("Year")

                    st.write("Actual vs predicted values by Olympic year")

                    chart_df = comparison_df.tail(8).copy()
                    chart_df["Year"] = chart_df["Year"].astype(str)
                    chart_df = chart_df.set_index("Year")[
                        ["Actual", "Random Forest", "Previous-edition baseline"]
                    ]

                    st.line_chart(chart_df)

                    st.caption(
                        "Actual values are historical dataset records. Predictions "
                        "use the three preceding editions, without using the target "
                        "edition's actual medal total as an input."
                    )

                    st.write("Year-wise prediction table")
                    st.dataframe(
                        comparison_df.sort_values("Year", ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )

                    rf_mae = comparison_df["RF absolute error"].mean()
                    baseline_mae = comparison_df["Baseline absolute error"].mean()

                    
                    col1, col2 = st.columns(2)
                    col1.metric(
                        "Country-specific RF MAE",
                        f"{rf_mae:.2f}"
                    )
                    col2.metric(
                        "Country-specific baseline MAE",
                        f"{baseline_mae:.2f}"
                    )

                    if rf_mae < baseline_mae:
                        st.success(
                            "Random Forest had lower error than the baseline "
                            "for this country's displayed test-period editions."
                        )
                    else:
                        st.warning(
                            "The previous-edition baseline performed as well as "
                            "or better than Random Forest for this selection."
                        )

                else:
                    st.info(
                        "No test-period editions with three preceding historical "
                        "editions are available for this country and season."
                    )

                st.markdown("#### Overall model evaluation")

                st.write("- Random Forest test MAE: **3.745**")
                st.write("- Previous-edition baseline MAE: **3.641**")

                st.warning(
                    "The Random Forest model did not outperform the "
                    "previous-edition baseline on the overall test set. "
                    "Use the forecast as an estimate, not a guaranteed result."
                )

                st.caption(
                    "Dataset note: these counts represent medal-winning athlete-event "
                    "records, not necessarily the number of unique physical medals."
                )

        else:
            st.error(
                "Required model or history file is missing. "
                "Check the files inside the models folder."
            )

    
    # ---------------------------------
    # 2. SPORT-WISE MEDAL PROBABILITY
    # ---------------------------------
    with probability_tab:
        st.subheader("Sport-wise Medal Probability")

        st.write(
            "Estimate the likelihood of a country winning at least one "
            "medal in a selected sport, using historical Olympic data."
        )

        sport_model_path = "models/country_sport_medal_model.pkl"
        sport_history_path = "models/country_sport_history.csv"

        if (
            os.path.exists(sport_model_path)
            and os.path.exists(sport_history_path)
        ):
            sport_model = load_model(sport_model_path)
            sport_history = load_history(sport_history_path)

            required_columns = [
                "NOC", "Sport", "Season", "Year", "Games",
                "entries", "medal_count",
                "prev_medals_1", "prev_medals_2", "prev_medals_3",
                "prev_entries_1", "prev_entries_2", "prev_entries_3",
                "prev_medal_rate"
            ]

            missing_columns = [
                col for col in required_columns
                if col not in sport_history.columns
            ]

            if missing_columns:
                st.error(
                    "The sport history file is missing required columns: "
                    + ", ".join(missing_columns)
                )
            else:
                col1, col2, col3 = st.columns(3)

                with col1:
                    probability_country = st.selectbox(
                        "Select country (NOC)",
                        sorted(sport_history["NOC"].dropna().unique()),
                        key="probability_country"
                    )

                with col2:
                    probability_season = st.selectbox(
                        "Select Olympic season",
                        sorted(sport_history["Season"].dropna().unique()),
                        key="probability_season"
                    )

                country_season = sport_history[
                    (sport_history["NOC"] == probability_country)
                    & (sport_history["Season"] == probability_season)
                ]

                with col3:
                    country_sports = sorted(
                        country_season["Sport"].dropna().unique()
                    )

                    probability_sport = st.selectbox(
                        "Select sport",
                        country_sports,
                        key="probability_sport"
                    )

                hist = sport_history[
                    (sport_history["NOC"] == probability_country)
                    & (sport_history["Season"] == probability_season)
                    & (sport_history["Sport"] == probability_sport)
                ].sort_values(["Year", "Games"])

                if hist.empty:
                    st.warning(
                        "No historical records were found for this "
                        "country, sport and season combination."
                    )
                else:
                    # Latest historical row contains the saved lag features
                    latest_row = hist.iloc[-1]

                    feature_columns = [
                        "NOC",
                        "Sport",
                        "Season",
                        "prev_medals_1",
                        "prev_medals_2",
                        "prev_medals_3",
                        "prev_entries_1",
                        "prev_entries_2",
                        "prev_entries_3",
                        "prev_medal_rate"
                    ]

                    input_df = pd.DataFrame([{
                        col: latest_row[col]
                        for col in feature_columns
                    }])

                    classes = list(sport_model.classes_)

                    if 1 not in classes:
                        st.error(
                            "The trained model does not contain "
                            "the medal-positive class."
                        )
                    else:
                        positive_index = classes.index(1)

                        probability = float(
                            sport_model.predict_proba(input_df)[0][
                                positive_index
                            ]
                        )

                        probability = max(0.0, min(1.0, probability))
                        probability_percent = probability * 100

                        scheduled_years = {
                            "Summer": [2020, 2024, 2028, 2032],
                            "Winter": [2018, 2022, 2026, 2030]
                        }

                        latest_year = int(latest_row["Year"])

                        future_years = [
                            year
                            for year in scheduled_years.get(
                                probability_season, []
                            )
                            if year > latest_year
                        ]

                        forecast_year = (
                            min(future_years)
                            if future_years
                            else latest_year + 4
                        )

                        st.markdown(
                            f"### {probability_country} — "
                            f"{probability_sport}"
                        )

                        st.caption(
                            f"Next estimated edition: "
                            f"{probability_season} {forecast_year}"
                        )

                        # Main probability display
                        st.markdown("#### Medal Probability")

                        p1, p2 = st.columns([2, 1])

                        with p1:
                            st.metric(
                                "Estimated probability of winning a medal",
                                f"{probability_percent:.1f}%"
                            )
                            st.progress(probability)
                            st.markdown("#### How to Interpret This Prediction")

                            st.write(
                                "The predicted probability represents the model's estimated "
                                "likelihood that the selected country will win at least one "
                                "medal in the selected sport. A higher probability indicates "
                                "a stronger prediction by the model, while a lower probability "
                                "indicates a weaker prediction."
                            )

                            st.caption(
                                "This is a machine-learning estimate based on historical data, "
                                "not a guarantee of future medal performance. The probability "
                                "may not be calibrated."
                            )
                        with p2:
                            st.metric(
                                "Latest historical medal records",
                                f"{float(latest_row['medal_count']):.0f}"
                            )
                            st.metric(
                                "Latest athlete-event records",
                                f"{float(latest_row['entries']):.0f}"
                            )

                        if probability_percent >= 70:
                            st.success(
                                "The model assigns a relatively high "
                                "medal probability for this selection."
                            )
                        elif probability_percent >= 40:
                            st.info(
                                "The model assigns a moderate medal "
                                "probability for this selection."
                            )
                        else:
                            st.info(
                                "The model assigns a relatively low "
                                "medal probability for this selection."
                            )

                        st.caption(
                            "These labels describe the model score only; "
                            "they are not guarantees of future performance."
                        )

                        
                        st.info(
                            "**How to interpret this prediction:** The model estimates "
                            "the likelihood that the selected country will record at least "
                            "one medal-winning result in the selected sport, assuming "
                            "participation. It is not an individual athlete prediction "
                            "or a prediction of the total medal count."
                        )

                        st.caption(
                            "The probability is an uncalibrated machine-learning estimate, "
                            "not a guarantee. Reliability depends on the historical data "
                            "and the model's performance on future Olympic editions."
                        )


                        # Historical trend
                        st.markdown("#### Historical Sport Performance")

                        trend_df = (
                            hist.groupby("Year", as_index=False)
                            .agg({
                                "medal_count": "sum",
                                "entries": "sum"
                            })
                            .sort_values("Year")
                        )

                        trend_chart = trend_df.set_index("Year")[
                            ["medal_count"]
                        ].rename(columns={
                            "medal_count": "Medal-winning records"
                        })
                        st.markdown("#### Historical Medal Performance")
                        st.caption(
                            "This chart shows how the selected country's medal-winning "
                            "records in this sport have changed across Olympic editions. "
                            "It helps identify improvements, declines, and performance trends."
                        )
                        st.line_chart(trend_chart)

                        entries_chart = trend_df.set_index("Year")[
                            ["entries"]
                        ].rename(columns={
                            "entries": "Athlete-event records"
                        })
                        st.markdown("#### Historical Participation Trend")
                        st.caption(
                            "This chart shows the number of athlete-event records for the "
                            "selected country and sport across Olympic editions. "
                            "It helps track changes in participation over time."
                        )
                        st.line_chart(entries_chart)

                        # Recent historical records
                        st.markdown("#### Recent Olympic Editions")

                        display_history = (
                            hist.sort_values("Year", ascending=False)
                            .head(5)[
                                ["Year", "Games", "entries", "medal_count"]
                            ]
                            .rename(columns={
                                "Year": "Olympic Year",
                                "Games": "Edition",
                                "entries": "Athlete-event records",
                                "medal_count": "Medal-winning records"
                            })
                        )

                        st.dataframe(
                            display_history,
                            use_container_width=True,
                            hide_index=True
                        )

                        # Show the actual features used by the model
                        with st.expander(
                            "View model input features"
                        ):
                            st.write(
                                "The prediction uses historical lag "
                                "features saved in the training history file."
                            )

                            st.dataframe(
                                input_df,
                                use_container_width=True,
                                hide_index=True
                            )

                        st.caption(
                            "Model limitation: this is an uncalibrated "
                            "probability estimate. Actual performance can "
                            "differ, and reliability depends on the "
                            "training data and future conditions."
                        )

                        # Sport model evaluation metrics
                        evaluation_path = "models/sport_model_evaluation.csv"

                        st.markdown("### Sport Model Evaluation")

                        if os.path.exists(evaluation_path):
                            evaluation_df = pd.read_csv(evaluation_path)

                            if not evaluation_df.empty:
                                metrics = evaluation_df.iloc[0]

                                metric1, metric2, metric3, metric4 = st.columns(4)

                                metric1.metric(
                                    "Accuracy",
                                    f"{metrics['Accuracy']:.2%}"
                                )
                                metric2.metric(
                                    "Precision",
                                    f"{metrics['Precision']:.2%}"
                                )
                                metric3.metric(
                                    "Recall",
                                    f"{metrics['Recall']:.2%}"
                                )
                                metric4.metric(
                                    "F1-score",
                                    f"{metrics['F1-score']:.4f}"
                                )

                                st.metric(
                                    "Baseline F1-score",
                                    f"{metrics['Baseline F1-score']:.4f}"
                                )

                                st.caption(
                                    f"Evaluation test cutoff year: "
                                    f"{int(metrics['Test cutoff year'])}. "
                                    "Metrics are calculated on historical test data; "
                                    "they do not guarantee future performance."
                                )
                            else:
                                st.warning("The evaluation file contains no metrics.")
                        else:
                            st.warning("Sport model evaluation file was not found.")

        else:
            st.error(
                "Required sport model or history file is missing. "
                "Check the files inside the models folder."
            )

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption(
    "Olympic Intelligence System | Historical analytics and "
    "machine-learning demonstration"
)