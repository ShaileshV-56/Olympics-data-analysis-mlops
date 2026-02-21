import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import pickle

# ---------------- LOAD DATA ----------------
df = pd.read_csv('data/athlete_events.csv')

# ---------------- PREPROCESS ----------------
df = df[['Age', 'Height', 'Weight', 'Sex', 'Sport', 'Event', 'Team', 'Year', 'Medal']].copy()

# Convert medal to binary (1 if medal was won, 0 if no medal)
df['Medal'] = (~df['Medal'].isna()).astype(int)

# Drop rows with missing required features
df = df.dropna(subset=['Age', 'Height', 'Weight', 'Sex', 'Sport', 'Event', 'Team', 'Year'])

# Reduce dimensionality by keeping only top categories
top_sports = df['Sport'].value_counts().head(15).index
top_events = df['Event'].value_counts().head(40).index
top_teams = df['Team'].value_counts().head(50).index

df = df[df['Sport'].isin(top_sports)]
df = df[df['Event'].isin(top_events)]
df = df[df['Team'].isin(top_teams)]

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Sport', 'Event', 'Team'], drop_first=True)

# Features & target
X = df.drop('Medal', axis=1)
y = df['Medal']

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print(f"Total samples: {len(df)}")
print(f"Features used: {len(X.columns)}")

# ---------------- SAVE MODEL ----------------
# Ensure models directory exists
model_dir = Path('models')
model_dir.mkdir(exist_ok=True)

model_path = model_dir / 'model.pkl'

# Save model + feature names + categorical encodings
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': model,
        'features': X.columns.tolist(),
        'categorical_features': ['Sex', 'Sport', 'Event', 'Team'],
        'top_sports': top_sports.tolist(),
        'top_events': top_events.tolist(),
        'top_teams': top_teams.tolist()
    }, f)

print(f"Model saved at: {model_path}")