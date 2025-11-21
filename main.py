import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error as mae
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
# Load dataset
df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\python\Boxoffice_revenue_prediction_using_linearregression_in_ML\boxoffice (1).csv", encoding='latin-1')
print(df.head())
print(df.shape)
print(df.info())
print(df.describe().T)
# Drop unwanted columns
to_remove = ['world_revenue', 'opening_revenue']
df.drop(to_remove, axis=1, inplace=True)
df.drop('budget', axis=1, inplace=True)
# Fill missing values
for col in ['MPAA', 'genres']:
    df[col] = df[col].fillna(df[col].mode()[0])
df.dropna(inplace=True)
# Clean numeric columns
df['domestic_revenue'] = df['domestic_revenue'].astype(str).str[1:]
for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
    df[col] = df[col].astype(str).str.replace(',', '')
    temp = ~df[col].isnull()
    df.loc[temp, col] = df.loc[temp, col].astype(float)
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Visualize
plt.figure(figsize=(10,5))
sb.countplot(x='MPAA', data=df)
plt.show()
print(df.groupby('MPAA')['domestic_revenue'].mean())
plt.figure(figsize=(15,5))
features = ['domestic_revenue', 'opening_theaters', 'release_days']
for i, col in enumerate(features):
    plt.subplot(1,3,i+1)
    sb.histplot(df[col], kde=True)
plt.tight_layout()
plt.show()
# Boxplots
plt.figure(figsize=(15,5))
for i, col in enumerate(features):
    plt.subplot(1,3,i+1)
    sb.boxplot(y=df[col])
plt.tight_layout()
plt.show()
# Log transform
for col in features:
    df[col] = df[col].apply(lambda x: np.log10(x+1))
plt.figure(figsize=(15,5))
for i, col in enumerate(features):
    plt.subplot(1,3,i+1)
    sb.histplot(df[col], kde=True)
plt.tight_layout()
plt.show()
# Encode genres
vectorizer = CountVectorizer()
vectorizer.fit(df['genres'])
genre_features = vectorizer.transform(df['genres']).toarray()
genre_names = vectorizer.get_feature_names_out()
for i, name in enumerate(genre_names):
    df[name] = genre_features[:, i]
df.drop('genres', axis=1, inplace=True)
# Remove sparse genre columns
removed = 0
if 'action' in df.columns and 'western' in df.columns:
    for col in df.loc[:, 'action':'western'].columns:
        if (df[col] == 0).mean() > 0.95:
            df.drop(col, axis=1, inplace=True)
            removed += 1
print("Removed columns:", removed)
print(df.shape)
# Encode categorical features
for col in ['distributor', 'MPAA']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
# Correlation heatmap
plt.figure(figsize=(8,8))
sb.heatmap(df.select_dtypes(include=np.number).corr() > 0.8, annot=True, cbar=False)
plt.show()
# Split data
X = df.drop(['title', 'domestic_revenue'], axis=1)
y = df['domestic_revenue'].values
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.1, random_state=22)
print(X_train.shape, X_val.shape)
# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
# Train model
model = XGBRegressor()
model.fit(X_train, Y_train)
train_preds = model.predict(X_train)
val_preds = model.predict(X_val)
print('Training Error:', mae(Y_train, train_preds))
print('Validation Error:', mae(Y_val, val_preds))
