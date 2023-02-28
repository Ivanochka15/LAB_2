from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=2400000, n_features=29, noise=0.1, random_state=42)
df = pd.DataFrame(X)
df['target'] = y
df.to_csv('regression_data.csv', index=False)
print("finish")

df = pd.read_csv('regression_data.csv')
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df['target'], test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

plt.figure(figsize=(10,6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={"s": 20})
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()
