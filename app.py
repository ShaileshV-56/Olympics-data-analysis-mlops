import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))
import preprocessor, helper

# Page config
st.set_page_config(page_title="Olympics Analysis", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(Path(__file__).parent / 'data' / 'athlete_events.csv')
    region_df = pd.read_csv(Path(__file__).parent / 'data' / 'noc_regions.csv')
    return preprocessor.preprocess(df, region_df)

df = load_data()

# ---------------- LOAD MODEL ----------------
try:
    model = pickle.load(open(Path(__file__).parent / 'models' / 'model.pkl', 'rb'))
except:
    model = None

# ---------------- TITLE ----------------
st.title("🏅 Olympic Athlete Performance Prediction Engine")
st.markdown("""
Analyze Olympic data, explore trends, and predict athlete performance.
""")
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Olympics Analysis")
st.sidebar.image('https://e7.pngegg.com/pngimages/1020/402/png-clipart-2024-summer-olympics-brand-circle-area-olympic-rings-olympics-logo-text-sport.png')

menu = st.sidebar.radio(
    'Select an Option',
    ('Medal Tally', 'Overall Analysis', 'Country-wise Analysis', 'Athlete wise Analysis', 'Prediction')
)

# ---------------- MEDAL TALLY ----------------
if menu == 'Medal Tally':
    st.header("Medal Tally")

    years, countries = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_country = st.sidebar.selectbox("Select Country", countries)

    result = helper.fetch_medal_tally(df, selected_year, selected_country)
    st.dataframe(result)

# ---------------- OVERALL ANALYSIS ----------------
elif menu == 'Overall Analysis':
    st.header("Top Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Editions", df['Year'].nunique())
    col2.metric("Hosts", df['City'].nunique())
    col3.metric("Sports", df['Sport'].nunique())

    col1, col2, col3 = st.columns(3)
    col1.metric("Events", df['Event'].nunique())
    col2.metric("Nations", df['region'].nunique())
    col3.metric("Athletes", df['Name'].nunique())

    st.subheader("Participation Over Time")

    st.plotly_chart(px.line(helper.data_over_time(df, 'region'), x="Edition", y="region"))
    st.plotly_chart(px.line(helper.data_over_time(df, 'Event'), x="Edition", y="Event"))
    st.plotly_chart(px.line(helper.data_over_time(df, 'Name'), x="Edition", y="Name"))

    st.subheader("Events Heatmap")
    temp = df.drop_duplicates(['Year', 'Sport', 'Event'])
    pt = temp.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(pt, ax=ax)
    st.pyplot(fig)

    st.subheader("Most Successful Athletes")
    sport = st.selectbox('Select Sport', ['Overall'] + sorted(df['Sport'].unique()))
    st.dataframe(helper.most_successful(df, sport))

# ---------------- COUNTRY ANALYSIS ----------------
elif menu == 'Country-wise Analysis':
    st.header("Country Analysis")

    country = st.sidebar.selectbox('Select Country', sorted(df['region'].dropna().unique()))

    country_df = helper.yearwise_medal_tally(df, country)
    st.plotly_chart(px.line(country_df, x="Year", y="Medal"))

    st.subheader("Sports Heatmap")
    pt = helper.country_event_heatmap(df, country)
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(pt, ax=ax)
    st.pyplot(fig)

    st.subheader("Top Athletes")
    st.dataframe(helper.most_successful_countrywise(df, country))

# ---------------- ATHLETE ANALYSIS ----------------
elif menu == 'Athlete wise Analysis':
    st.header("Athlete Analysis")

    athlete_df = df.drop_duplicates(subset=['Name', 'region']).copy()

    st.subheader("Age Distribution")
    st.plotly_chart(px.histogram(athlete_df, x="Age", color="Medal"))

    st.subheader("Height vs Weight")
    sport = st.selectbox('Select Sport', ['Overall'] + sorted(df['Sport'].unique()))
    temp_df = helper.weight_v_height(df, sport)

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=temp_df,
        x='Weight',
        y='Height',
        hue='Medal',
        style='Sex',
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Men vs Women Participation")
    final = helper.men_vs_women(df)
    st.plotly_chart(px.line(final, x="Year", y=["Male", "Female"]))

# ---------------- PREDICTION ----------------
elif menu == 'Prediction':
    st.header("Athlete Medal Prediction")

    if model is None:
        st.error("Model not found. Train and save model.pkl in /models folder.")
    else:
        age = st.slider("Age", 10, 60)
        height = st.slider("Height (cm)", 140, 220)
        weight = st.slider("Weight (kg)", 40, 120)
        sex = st.selectbox("Sex", ["M", "F"])

        sex_M = 1 if sex == "M" else 0

        if st.button("Predict"):
            pred = model.predict([[age, height, weight, sex_M]])

            if pred[0] == 1:
                st.success("🏅 Likely to win a medal")
            else:
                st.warning("No medal predicted")