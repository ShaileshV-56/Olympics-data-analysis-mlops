import pandas as pd

df = pd.read_csv('data/athlete_events.csv')
df = df[['Age', 'Height', 'Weight', 'Sex', 'Sport', 'Event', 'Team', 'Year', 'Medal']].copy()
df['Medal'] = (~df['Medal'].isna()).astype(int)
df = df.dropna(subset=['Age', 'Height', 'Weight', 'Sex', 'Sport', 'Event', 'Team', 'Year'])

print(f'Data shape after cleaning: {df.shape}')
print(f'Unique Sports: {df["Sport"].nunique()}')
print(f'Unique Events: {df["Event"].nunique()}')
print(f'Unique Teams: {df["Team"].nunique()}')

# Estimate features after one-hot encoding
approx_features = 1 + 1 + 1 + 1 + (df["Sex"].nunique()-1) + (df["Sport"].nunique()-1) + (df["Event"].nunique()-1) + (df["Team"].nunique()-1)
print(f'Approximately {approx_features} features after one-hot encoding')
