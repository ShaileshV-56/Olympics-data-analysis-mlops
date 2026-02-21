import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import pickle
import numpy as np

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from predict import predict, load_model

st.set_page_config(page_title="Medal Prediction", layout="wide")

st.title("🥇 Medal Prediction")
st.write("Predict whether an athlete will win a medal based on their characteristics and event details")

# Load model
@st.cache_resource
def get_model():
    model_path = Path(__file__).parent.parent / 'models' / 'model.pkl'
    try:
        model, features, top_sports, top_events, top_teams = load_model(str(model_path))
        return model, features, top_sports, top_events, top_teams
    except FileNotFoundError:
        st.error("❌ Model file not found. Please train the model first using `python src/train.py`")
        return None, None, None, None, None

model, features, top_sports, top_events, top_teams = get_model()

if model is None:
    st.stop()

# Load data for reference
@st.cache_data
def load_reference_data():
    df = pd.read_csv(Path(__file__).parent.parent / 'data' / 'athlete_events.csv')
    return df

df_ref = load_reference_data()

# Input options
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

if input_method == "Manual Entry":
    st.subheader("Enter Athlete Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=10, max_value=80, value=25)
        height = st.slider("Height (cm)", min_value=140, max_value=230, value=175)
        weight = st.slider("Weight (kg)", min_value=40, max_value=200, value=70)
        sex = st.selectbox("Sex", options=["M", "F"])
    
    with col2:
        sport_options = list(top_sports) if top_sports else sorted(df_ref['Sport'].unique())
        sport = st.selectbox("Sport", options=sport_options)
        
        event_options = sorted(df_ref[df_ref['Sport'] == sport]['Event'].unique())
        if top_events:
            event_options = [e for e in event_options if e in top_events]
        event = st.selectbox("Event", options=event_options if event_options else ['N/A'])
        
        team_options = list(top_teams) if top_teams else sorted(df_ref['Team'].unique())
        team = st.selectbox("Team/Country", options=team_options)
        year = st.slider("Year", min_value=1896, max_value=2024, value=2024, step=4)
    
    if st.button("Predict Medal", type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'Sex': [sex],
            'Sport': [sport],
            'Event': [event],
            'Team': [team],
            'Year': [year]
        })
        
        # One-hot encode
        input_encoded = pd.get_dummies(input_data, columns=['Sex', 'Sport', 'Event', 'Team'], drop_first=True)
        
        # Ensure all features match
        for col in features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        input_encoded = input_encoded[features]
        
        # Predict
        try:
            prediction = model.predict(input_encoded)[0]
            probability = model.predict_proba(input_encoded)[0, 1]
            
            st.divider()
            st.subheader("🎯 Prediction Result")
            
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.success(f"✅ **Medal Predicted** ({probability*100:.1f}% confidence)")
                else:
                    st.info(f"❌ **No Medal Predicted** ({(1-probability)*100:.1f}% confidence)")
            
            with col2:
                st.metric("Medal Probability", f"{probability*100:.1f}%")
            
            # Show input summary
            st.write("**Input Summary:**")
            st.dataframe(input_data, width='stretch')
        except Exception as e:
            st.error(f"⚠️ The selected combination is not in the model's training data. Please choose different values.\nError: {str(e)}")

else:  # Upload CSV
    st.subheader("Upload CSV File")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        
        st.write("**Preview of uploaded data:**")
        st.dataframe(input_df.head(10), width='stretch')
        
        required_cols = ['Age', 'Height', 'Weight', 'Sex', 'Sport', 'Event', 'Team', 'Year']
        missing_cols = [col for col in required_cols if col not in input_df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            if st.button("Predict on All Records", type="primary"):
                try:
                    # Process predictions
                    results = []
                    for idx, row in input_df.iterrows():
                        data_point = pd.DataFrame([row[required_cols]])
                        data_encoded = pd.get_dummies(data_point, columns=['Sex', 'Sport', 'Event', 'Team'], drop_first=True)
                        
                        for col in features:
                            if col not in data_encoded.columns:
                                data_encoded[col] = 0
                        
                        data_encoded = data_encoded[features]
                        
                        pred = model.predict(data_encoded)[0]
                        prob = model.predict_proba(data_encoded)[0, 1]
                        
                        results.append({
                            **row[required_cols].to_dict(),
                            'Medal_Predicted': 'Yes' if pred == 1 else 'No',
                            'Medal_Probability': f"{prob*100:.1f}%"
                        })
                    
                    results_df = pd.DataFrame(results)
                    st.success("✅ Predictions complete!")
                    st.dataframe(results_df, width='stretch')
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="medal_predictions.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

# Model info
st.divider()
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.write(f"- **Total Features**: {len(features)}")
st.sidebar.write(f"- **Model Type**: Random Forest Classifier")
st.sidebar.write(f"- **Input Features**: Age, Height, Weight, Sex, Sport, Event, Team, Year")
