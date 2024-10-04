import pandas as pd
import numpy as np
import seaborn as sns
import os
import glob
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('openweathermap.csv', sep=',')

df1 = pd.read_csv('Photovoltaic Panels.csv', sep=';')

df1 = df1.drop(columns = ["Number", "Working State"], axis=1)

df2 = pd.concat([df, df1], axis=1)

df2 = df2.drop(columns=["Time", "cod", "message", "cnt", "list__dt", "list__main__feels_like", "list__main__temp_min", "list__main__temp_max",
                        "list__main__pressure", "list__main__sea_level", "list__main__grnd_level", "list__main__temp_kf", "list__weather__id",
                        "list__weather__description", "list__weather__icon", "list__clouds__all", "list__wind__deg", "list__wind__gust",
                        "list__visibility", "list__pop", "list__rain__3h", "list__sys__pod", "list__dt_txt", "city__id", "city__name",
                        "city__coord__lat", "city__coord__lon", "city__country", "city__population", "city__timezone", "city__sunrise", "city__sunset"], axis = 1)

df2.fillna(0, inplace=True)

indexNames = df2[df2["list__main__temp"] == 0].index
df2.drop(indexNames, inplace=True)
df2

df2.rename(columns={"list__main__temp": "Temperature", "list__main__humidity": "Humidity", "list__weather__main": "Weather_Conditions",
                    "list__wind__speed": "Wind_Speed"}, inplace = True)

df2['Temperature'] = df2['Temperature'] - 273.15

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df2['Weather_Conditions'] = le.fit_transform(df2.Weather_Conditions)

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))
sns.heatmap(df2.corr(), annot=True, cmap="coolwarm")
plt.title("Dataset Correlation Matrix")
plt.show()

X = df2.drop(columns = ["Grid(W)"], axis = 1)
y = df2["Grid(W)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)

print("------shapes------")
print(f"training  {X_train.shape}")
print(f"test {X_test.shape}" )

MinMax = MinMaxScaler(feature_range= (0,1))
X_train = MinMax.fit_transform(X_train)
X_test = MinMax.transform(X_test)

import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import math
import numpy as np

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Define the CatBoost Regressor model
model = CatBoostRegressor(task_type='GPU', loss_function='RMSE',
                         per_float_feature_quantization="0:border_count=128", bagging_temperature=1,
                            bootstrap_type="Bayesian", random_strength=0.5, grow_policy='Depthwise', border_count=128,
                           has_time=True, feature_border_type="Uniform", depth = 3, iterations = 1000, l2_leaf_reg = 0.1, learning_rate = 0.1)
                           

# Train the model on the training data

model.fit(X_train, y_train)


# Make predictions on the test set

predictions = model.predict(X_test)


# Calculate R-squared score

r2 = r2_score(y_test, predictions)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("R-squared Score:", r2)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)



# Create a scatter plot of actual vs predicted values

plt.scatter(y_test, predictions)


# Plot a line of perfect prediction

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')


# Set title and labels

plt.title('Actual vs Predicted Values (R2 = {:.4f})'.format(r2))

plt.xlabel('Actual Values')

plt.ylabel('Predicted Values')


# Show the plot

plt.show()


residuals = y_test - predictions


# Create a scatter plot of residuals

plt.scatter(predictions, residuals)


# Plot a horizontal line at y=0

plt.axhline(y=0, color='r', linestyle='-')


# Set title and labels

plt.title('Residuals vs Predicted Values (R2 = {:.4f})'.format(r2))

plt.xlabel('Predicted Values')

plt.ylabel('Residuals')


# Show the plot

plt.show()
