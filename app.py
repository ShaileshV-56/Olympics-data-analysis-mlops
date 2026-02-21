import streamlit as st
import pandas as pd
import preprocessor, helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Olympics Analysis", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv('athlete_events.csv')
    region_df = pd.read_csv('noc_regions.csv')
    return preprocessor.preprocess(df, region_df)

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Olympics Analysis")
st.sidebar.image('https://e7.pngegg.com/pngimages/1020/402/png-clipart-2024-summer-olympics-brand-circle-area-olympic-rings-olympics-logo-text-sport.png')

menu = st.sidebar.radio(
    'Select an Option',
    ('Medal Tally', 'Overall Analysis', 'Country-wise Analysis', 'Athlete wise Analysis')
)

# ---------------- MEDAL TALLY ----------------
if menu == 'Medal Tally':
    st.title("Medal Tally")

    years, countries = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_country = st.sidebar.selectbox("Select Country", countries)

    result = helper.fetch_medal_tally(df, selected_year, selected_country)
    st.dataframe(result)

# ---------------- OVERALL ANALYSIS ----------------
elif menu == 'Overall Analysis':
    st.title("Top Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Editions", df['Year'].nunique())
    col2.metric("Hosts", df['City'].nunique())
    col3.metric("Sports", df['Sport'].nunique())

    col1, col2, col3 = st.columns(3)
    col1.metric("Events", df['Event'].nunique())
    col2.metric("Nations", df['region'].nunique())
    col3.metric("Athletes", df['Name'].nunique())

    st.title("Participation Over Time")

    st.plotly_chart(px.line(helper.data_over_time(df, 'region'), x="Edition", y="region"))
    st.plotly_chart(px.line(helper.data_over_time(df, 'Event'), x="Edition", y="Event"))
    st.plotly_chart(px.line(helper.data_over_time(df, 'Name'), x="Edition", y="Name"))

    st.title("Events Heatmap")
    temp = df.drop_duplicates(['Year', 'Sport', 'Event'])
    pt = temp.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(pt, ax=ax)
    st.pyplot(fig)

    st.title("Most Successful Athletes")
    sport = st.selectbox('Select Sport', ['Overall'] + sorted(df['Sport'].unique()))
    st.dataframe(helper.most_successful(df, sport))

# ---------------- COUNTRY ANALYSIS ----------------
elif menu == 'Country-wise Analysis':
    st.title("Country Analysis")

    country = st.sidebar.selectbox('Select Country', sorted(df['region'].dropna().unique()))

    country_df = helper.yearwise_medal_tally(df, country)
    st.plotly_chart(px.line(country_df, x="Year", y="Medal"))

    st.title("Sports Heatmap")
    pt = helper.country_event_heatmap(df, country)
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(pt, ax=ax)
    st.pyplot(fig)

    st.title("Top Athletes")
    st.dataframe(helper.most_successful_countrywise(df, country))

# ---------------- ATHLETE ANALYSIS ----------------
elif menu == 'Athlete wise Analysis':
    st.title("Athlete Analysis")

    athlete_df = df.drop_duplicates(subset=['Name', 'region']).copy()

    st.title("Age Distribution")
    st.plotly_chart(px.histogram(athlete_df, x="Age", color="Medal"))

    st.title("Height vs Weight")
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

    st.title("Men vs Women Participation")
    final = helper.men_vs_women(df)
    st.plotly_chart(px.line(final, x="Year", y=["Male", "Female"]))