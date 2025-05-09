import sklearn
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/Kaggle datasets/15 Years Stock Data of NVDA AAPL MSFT GOOGL and AMZN.csv')
print(df.head())