import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

global_temp = pd.read_csv("GlobalTemperatures.csv")
print(global_temp.shape)
print(global_temp.columns)
print(global_temp.info())
print(global_temp.isnull().sum())

# Data Preparation
def wrangle(df):
    df = df.copy()
    df = df.drop(columns=[
        "LandAverageTemperatureUncertainty",
        "LandMaxTemperatureUncertainty",
        "LandMinTemperatureUncertainty",
        "LandAndOceanAverageTemperatureUncertainty"
    ], axis=1)

    def converttemp(x):
        x = (x * 1.8) + 32
        return float(x)

    df["LandAverageTemperature"] = df["LandAverageTemperature"].apply(converttemp)
    df["LandMaxTemperature"] = df["LandMaxTemperature"].apply(converttemp)
    df["LandMinTemperature"] = df["LandMinTemperature"].apply(converttemp)
    df["LandAndOceanAverageTemperature"] = df["LandAndOceanAverageTemperature"].apply(converttemp)

    df["dt"] = pd.to_datetime(df["dt"])
    df["Month"] = df["dt"].dt.month
    df["Year"] = df["dt"].dt.year
    df = df.drop("dt", axis=1)
    df = df.drop("Month", axis=1)
    df = df[df.Year >= 1850]
    df = df.set_index(["Year"])
    df = df.dropna()
    return df

global_temp = wrangle(global_temp)
print(global_temp.head())

corrMatrix = global_temp.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

target = "LandAndOceanAverageTemperature"
y = global_temp[target]
x = global_temp[["LandAverageTemperature", "LandMaxTemperature", "LandMinTemperature"]]

xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.25, random_state=42)
print(xtrain.shape)
print(xval.shape)
print(ytrain.shape)
print(yval.shape)
baseline_pred = [ytrain.mean()] * len(yval)
errors = abs(baseline_pred - yval)
mape = 100 * (errors / yval)
accuracy = 100 - np.mean(mape)
print("Baseline Accuracy: ", round(accuracy, 2), "%")


forest = make_pipeline(
    SelectKBest(k="all"),
    StandardScaler(),
    RandomForestRegressor(
        n_estimators=100,
        max_depth=50,
        random_state=77,
        n_jobs=-1
    )
)
forest.fit(xtrain, ytrain)

# Model prediction
forest_pred = forest.predict(xval)

# Evaluate
forest_errors = abs(forest_pred - yval)
forest_mape = 100 * (forest_errors / yval)
forest_accuracy = 100 - np.mean(forest_mape)
print("Random Forest Accuracy: ", round(forest_accuracy, 2), "%")

plt.figure(figsize=(10, 6))
plt.plot(yval.values, label="Actual", alpha=0.7)
plt.plot(forest_pred, label="Predicted", alpha=0.7)
plt.legend()
plt.title("Actual vs Predicted Temperatures")
plt.xlabel("Sample Index")
plt.ylabel("Temperature (°F)")
plt.show()


joblib.dump(forest, "random_forest_temp_model.pkl")