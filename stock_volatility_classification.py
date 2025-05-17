import sklearn
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/Kaggle datasets/15 Years Stock Data of NVDA AAPL MSFT GOOGL and AMZN.csv')
print(df.columns)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

def stock_iteration_gb(df, stock):
    print('Initialising the function...',)
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

    train_accuracy = gb.score(X_train, y_train)
    print(f'Accuracy score for training dataset: {train_accuracy}')

    test_accuracy = gb.score(X_test, y_test)
    print(f'Accuracy for test dataset: {test_accuracy}')

    #evaluation
    y_pred = gb.predict(X_test)
    print(accuracy_score(y_pred, y_test))
    print(confusion_matrix(y_pred, y_test))
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.05],
        'max_depth': [2, 3],
    }

    gs = GridSearchCV(gb, param_grid, cv=5, scoring='f1')
    gs.fit(X_train, y_train)

    print(f'Best parameters are: {gs.best_params_}')
    best_model = gs.best_estimator_
    print(best_model)

    y_pred = best_model.predict(X_test)
    print(accuracy_score(y_pred, y_test))
    print(confusion_matrix(y_pred, y_test))

    from sklearn.metrics import classification_report
    print(classification_report(y_pred, y_test))

    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(df):
        print(train_index, test_index)

    from sklearn.model_selection import learning_curve
    train_size, train_score, test_score = learning_curve(gb, X, y, cv=5)
    train_mean = train_score.mean(axis=1)
    test_mean = test_score.mean(axis=1)

    plt.plot(train_size, train_mean, label='Training acc')
    plt.plot(train_size, test_mean, label='Test acc')
    plt.xlabel('Training size')
    plt.ylabel('Acc')
    plt.title('learning curve')
    plt.legend()
    plt.show()

for stock in ['AAPL', 'AMZN', 'MSFT', 'NVDA', 'GOOGL']:
    stock_iteration_gb(df, stock)