import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score


df = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/Kaggle datasets/15 Years Stock Data of NVDA AAPL MSFT GOOGL and AMZN.csv')
print('Begin the operation...')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# cleaning data
warnings.filterwarnings("ignore")
print(df.head())
print(df.isna().sum())
df.dropna(inplace=True)

# EDA, data visualization
print('Summary statistics:')
print(df.describe())

# correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.shape[1] > 4:
    plt.figure(figsize=(10, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Heatmap')
    plt.show()
print('EDA and data analsis complete')

print('Engineering features...')
# daily return
df['Daily_return'] = df['Close_MSFT'].pct_change()
df.dropna(subset=['Daily_return'], inplace=True)

#rolling volatility
win_size = 20
df['Rolling_volatility'] = df['Daily_return'].rolling(window=win_size).std()

# threshold for volatility
threshold = df['Rolling_volatility'].quantile(0.55)
df['Volatility'] = (df['Rolling_volatility'] >threshold).astype(int)

# 5-day moving average
MA_win_size = [5, 10, 20]
for size in MA_win_size:
    df[f'MA_{size}'] = df['Close_MSFT'].rolling(window=size).mean()

# lagged features
df[f'Prev_volatility'] = df[f'Volatility'].shift(1)
df[f'Prev_close'] = df[f'Close_MSFT'].shift(1)
df.dropna(subset=['Prev_volatility', 'Prev_close'], inplace=True)
print('Feature engineering complete')

print('Begin train & test split...')
# scaling and splitting data with time-based split
features = ['MA_5', 'MA_10', 'MA_20', 'Prev_volatility', 'Prev_close']
X = df[features]
y = df['Volatility']

ratio = 0.8
split_index = int(len(df) * ratio)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model training and fitting
rfc = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced',
)
rfc.fit(X_train_scaled, y_train)
y_pred = rfc.predict(X_test_scaled)
print('Model training complete')

print('Begin model evaluation...')
# model evaluation
print(f'Classification report:', classification_report(y_test, y_pred))
print(f'Confusion matrix:', confusion_matrix(y_test, y_pred))

# time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
cross_val_scores = cross_val_score(rfc, X_train_scaled, y_train, cv=tscv)
print(f'CV scores:', cross_val_scores)

#hyperparameters
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 3],
}

gs = GridSearchCV(rfc, param_grid, cv=tscv, scoring='f1_macro', n_jobs=-1)
gs.fit(X_train_scaled, y_train)
best_model = gs.best_estimator_
print(f'Best parameters:', gs.best_params_)
print(f'Best score:', gs.best_score_)
y_pred_best = best_model.predict(X_test)

print(f'Best classification report:', classification_report(y_test, y_pred_best))
print(f'Best confusion matrix:', confusion_matrix(y_test, y_pred_best))
print('Evaluation & tuning complete')