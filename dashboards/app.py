import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import joblib 
import seaborn as sns 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Zukunft And Geist",
    page_icon="Z&G",
    layout="wide"
)
st.title("Z&G Zukunft & Geist")
st.subheader("Data-Driven Analysis")

st.write(
    "An interactive Dashboard to explore Digital Well-Being"
)

from pathlib import Path
PROJECT_ROOT= Path(__file__).resolve().parents[1]
DATA_PATH= PROJECT_ROOT/"data"/"raw"/"Mental_Health_and_Social_Media_Balance_Dataset.csv"
MODEL_PATH=Path(__file__).parent/"tuned_random_forest_pipeline.pkl"
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
st.success("Dataset and Trained Model Loaded Successfully! ")
st.write("Dataset shape:",df.shape)

st.header("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users", len(df))
col2.metric("Average Happiness",
            round(df["Happiness_Index(1-10)"].mean(),2)
            )
col3.metric("Average Screen Time",
            round(df["Daily_Screen_Time(hrs)"].mean(),2)
            )
col4.metric("Average Sleep Quality",
            round(df["Sleep_Quality(1-10)"].mean(),2)
            )
st.header("Screen time vs Hapiness ")
fig, ax =  plt.subplots(figsize=(8,5))
ax.scatter(
    df["Daily_Screen_Time(hrs)"],
    df["Happiness_Index(1-10)"],
    alpha=0.6
)
ax.set_xlabel("Daily_Screen_Time(hrs)")
ax.set_ylabel("hapiness Index")
ax.set_title("Screen Time Vs Hapiness")

st.pyplot(fig)

st.header("Social Media Platform Distribution ")
platform_counts = df["Social_Media_Platform"].value_counts()
fig,ax = plt.subplots(figsize=(8,5))
platform_counts.plot(kind="bar",ax=ax)

ax.set_xlabel("Social Media Platform")
ax.set_ylabel("Number of users")
ax.set_title("Distribution of Social Media Platforms ")
plt.xticks(rotation=45)

st.pyplot(fig)


st.header("Stress Level vs Hapiness ")
fig, ax =  plt.subplots(figsize=(8,5))
ax.scatter(
    df["Stress_Level(1-10)"],
    df["Happiness_Index(1-10)"],
    alpha=0.6
)
ax.set_xlabel("Stress Level")
ax.set_ylabel("Hapiness Index")
ax.set_title("Stress Level Vs Happiness")

st.pyplot(fig)

st.header("Correlational Heatmap")
numeric_df = df.select_dtypes(include=["int64", "float64"])
corr_matrix = numeric_df.corr()
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)
ax.set_title("Correlation Between Numerical variables")
st.pyplot(fig)

st.header("Interactive Happiness Prediction")
st.write("Enter user characteristics to predict the Happiness Index ")
with st.form("prediction_form"):
    age = st.slider("Age", 15, 60, 22)

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )
    screen_time = st.slider(
        "Daily_Screen_Time(Hours)",
        0.0, 15.0, 5.0, 0.5
    )

    sleep_quality = st.slider(
            "Sleep_Quality(1-10)",
            1, 10, 6
        )
    stress_level = st.slider(
            "Stress_Level(1-10)",
            1, 10, 5
        )
    days_without_social_media = st.slider(
            "Days Without Social Media",
            0, 30, 2
        )
    exercise_frequency = st.slider(
            "Exercise Frequency per Week ",
            0, 7, 3
        )
    platform = st.selectbox(
            "Social Media Platform",
            df["Social_Media_Platform"].unique()
        )
    submit = st.form_submit_button("Predict Happiness")

if submit:
    screen_category = (
        "Low" if screen_time <=4
        else "Moderate" if screen_time <=7
        else "High"
    )
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Daily_Screen_Time(hrs)": screen_time,
        "Sleep_Quality(1-10)": sleep_quality,
        "Stress_Level(1-10)": stress_level,
        "Days_Without_Social_Media": days_without_social_media,
        "Exercise_Frequency(week)": exercise_frequency,
        "Social_Media_Platform": platform,
        "Screen_Time_Category": screen_category
    }])
    prediction = model.predict(input_data)[0]

    st.success("Predicted Happiness Index: " + str(round(prediction, 2)) + " / 10")
    