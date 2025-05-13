import sklearn
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/Kaggle datasets/15 Years Stock Data of NVDA AAPL MSFT GOOGL and AMZN.csv')
print(df.columns)

from sklearn.ensemble import GradientBoostingClassifier

def stock_iteration_gb(df, stock):
    df = df.copy()
    df[f'Volatility_{stock}'] = df[f'High_{stock}'] - df[f'Low_{stock}']
    median = df[f'Volatility_{stock}'].median()
    df[f'Volatility_{stock}'] = (df[f'Volatility_{stock}'] > median).astype(int)
    #feature engineering part
    # lagged features
    df[f'Prev_volatility_{stock}'] = df[f'Volatility_{stock}'].shift(1)
    df[f'Prev_closing_{stock}'] = df[f'Close_{stock}'].shift(1)

     # moving averages
    df[f'MA_5_{stock}'] = df[f'Close_{stock}'].rolling(window=5).mean()
    df[f'MA_10_{stock}'] = df[f'Close_{stock}'].rolling(window=10).mean()

    # price change in return 
    df[f'Return_{stock}'] = df[f'Close_{stock}'].pct_change()

    #compute_rolling_volatility
    df[f'Rolling_volatility_{stock}'] = df[f'Return_{stock}'].rolling(window=5).std()

    df = df.dropna()

    y = df[f'Volatility_{stock}']
    X = df[[f'Prev_volatility_{stock}', f'Prev_closing_{stock}', f'MA_5_{stock}', f'Return_{stock}', f'Rolling_volatility_{stock}']]

    #train_test_split set 
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)

    #creating a Decision Tree model and fitting it
    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)

    #evaluation
    y_pred = gb.predict(X_test)
    print(accuracy_score(y_pred, y_test))
    print(confusion_matrix(y_pred, y_test))

for stock in ['AAPL', 'AMZN', 'MSFT', 'NVDA', 'GOOGL']:
    stock_iteration_gb(df, stock)
