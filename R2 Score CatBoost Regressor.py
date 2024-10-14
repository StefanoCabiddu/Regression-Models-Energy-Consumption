import pandas as pd
import numpy as np
import seaborn as sns
import os
import glob
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('openweathermap.csv', sep=',')

df = df.drop(columns=["cod", "message", "cnt", "list__dt", "list__main__feels_like", "list__main__temp_min", "list__main__temp_max",
                        "list__main__pressure", "list__main__sea_level", "list__main__grnd_level", "list__main__temp_kf", "list__weather__id",
                        "list__weather__description", "list__weather__icon", "list__clouds__all", "list__wind__deg", "list__wind__gust",
                        "list__visibility", "list__pop", "list__rain__3h", "list__sys__pod", "list__dt_txt", "city__id", "city__name",
                        "city__coord__lat", "city__coord__lon", "city__country", "city__population", "city__timezone", "city__sunrise", "city__sunset"], axis = 1)

df.rename(columns={"list__main__temp": "Temperature(°C)", "list__main__humidity": "Humidity(%)", "list__weather__main": "Weather_Conditions",
                    "list__wind__speed": "Wind_speed(m/s)"}, inplace = True)

df['Temperature(°C)'] = df['Temperature(°C)'] - 273.15

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Weather_Conditions'] = le.fit_transform(df.Weather_Conditions)

new_df = pd.DataFrame(columns=df.columns)


for index, row in df.iterrows():

    new_df = pd.concat([new_df, pd.DataFrame([row])], ignore_index=True)

    if row.notna().any():

        new_df = pd.concat([new_df, pd.DataFrame(np.nan, index=range(14), columns=df.columns)], ignore_index=True)

df_interpolated = new_df.interpolate(axis=0)

df_interpolated = df_interpolated.round(2)

df1 = pd.read_csv('Dati professoressa Sanguinetti.csv', sep=';')

df1 = df1.drop(columns=["Datetime"], axis = 1)

df2 = pd.concat([df_interpolated, df1], axis=1)

df2.fillna(0, inplace=True)
indexNames = df2[df2["Consumption"] == 0].index
df2.drop(indexNames, inplace=True)

columns_to_convert = ['Consumption', 'FeedIn', 'Production', 'Purchased', 'SelfConsumption']
df2[columns_to_convert] = df2[columns_to_convert].applymap(lambda x: float(x.replace(',', '.')))

df2['Weather_Conditions'] = df2['Weather_Conditions'].astype('int')

df2['Weather_Conditions'] = le.inverse_transform(df2['Weather_Conditions'])

import matplotlib.pyplot as plt
import seaborn as sns

df2.groupby('Weather_Conditions').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

df2.hist(bins=20, figsize=(20,15))
plt.show()

le = LabelEncoder()
df2['Weather_Conditions'] = le.fit_transform(df2.Weather_Conditions)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

plt.figure(figsize=(20, 15))
sns.heatmap(df2.corr(), annot=True, cmap="coolwarm")
plt.title("Dataset Correlation Matrix")
plt.show()

X = df2.drop(columns=["Purchased"], axis=1)
y = df2["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)

print("------shapes------")
print(f"training  {X_train.shape}")
print(f"test {X_test.shape}")

MinMax = MinMaxScaler(feature_range=(0, 1))
X_train = MinMax.fit_transform(X_train)
X_test = MinMax.transform(X_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor

# Define the hyperparameter grid for CatBoost Regressor
param_grid = {
        'learning_rate': [0.01, 0.1, 1],
        'depth': [3, 5, 7],
        'l2_leaf_reg': [0.1, 1, 10],
        'iterations': [100, 500, 1000]
}

# Define the CatBoost Regressor model
model = CatBoostRegressor(task_type='GPU', loss_function='RMSE',
                              per_float_feature_quantization="0:border_count=1024", bagging_temperature=1,
                              bootstrap_type="Bayesian", random_strength=0.5, grow_policy='Depthwise', border_count=1024,
                              has_time=True, feature_border_type="Uniform")

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_model1 = grid_search.best_params_
print(best_model1)
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'Best R2 score: {r2:.4f}')

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Create a scatter plot of actual vs predicted values
plt.scatter(y_test, y_pred)

# Plot a line of perfect prediction
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')

# Set title and labels
plt.title('Actual vs Predicted Values (R2 = {:.4f})'.format(r2))
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Show the plot
plt.show()

residuals = y_test - y_pred

# Create a scatter plot of residuals
plt.scatter(y_pred, residuals)

# Plot a horizontal line at y=0
plt.axhline(y=0, color='r', linestyle='-')

# Set title and labels
plt.title('Residuals vs Predicted Values (R2 = {:.4f})'.format(r2))
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

# Show the plot
plt.show()
