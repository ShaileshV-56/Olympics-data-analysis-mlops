"""Prediction module for Olympics data analysis model."""

import pickle
import pandas as pd
from pathlib import Path


def load_model(model_path: str):
    """Load trained model and feature names."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return (data['model'], 
            data['features'], 
            data.get('top_sports', []),
            data.get('top_events', []),
            data.get('top_teams', []))


def preprocess_input(df: pd.DataFrame, features: list, top_sports=None, top_events=None, top_teams=None):
    """Preprocess input data to match training format."""
    
    # Select features and drop rows with missing data
    df = df[['Age', 'Height', 'Weight', 'Sex', 'Sport', 'Event', 'Team', 'Year']].dropna()
    
    if df.empty:
        raise ValueError("No valid data after preprocessing")
    
    # Filter by top categories if provided
    if top_sports and len(top_sports) > 0:
        df = df[df['Sport'].isin(top_sports)]
    if top_events and len(top_events) > 0:
        df = df[df['Event'].isin(top_events)]
    if top_teams and len(top_teams) > 0:
        df = df[df['Team'].isin(top_teams)]
    
    if df.empty:
        raise ValueError("No data matches the model's training categories")

    # One-hot encoding for categorical variables (match training)
    df = pd.get_dummies(df, columns=['Sex', 'Sport', 'Event', 'Team'], drop_first=True)

    # Ensure all training features exist in prediction data
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct column order and only select training features
    df = df[features]

    return df


def predict(model_path: str, data_path: str):
    """
    Make predictions using trained model.
    """
    model, features, top_sports, top_events, top_teams = load_model(model_path)

    df = pd.read_csv(data_path)
    X = preprocess_input(df, features, top_sports, top_events, top_teams)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]  # Get probability of winning medal

    df_result = df[['Age', 'Height', 'Weight', 'Sex', 'Sport', 'Event', 'Team', 'Year']].copy()
    df_result = df_result.iloc[:len(X)]
    df_result['Medal_Prediction'] = predictions
    df_result['Medal_Probability'] = probabilities

    return df_result


if __name__ == "__main__":
    model_path = str(Path("models") / "model.pkl")
    data_path = str(Path("data") / "athlete_events.csv")

    predictions = predict(model_path, data_path)
    print(predictions.head())