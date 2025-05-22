import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from pathlib import Path

# Filepaths
base_dir = Path(__file__).parent
weather_fp = str(base_dir / "NYCWeatherData/New York Weather Data.csv")
load_fp    = str(base_dir / "NYCEnergyLoad/OASIS_Real_Time_Dispatch_Actual_Load.csv")

# 1) Load raw weather CSV
#    We read all columns, but we’ll filter to rows where Hourly* are NaN
weather_raw = pd.read_csv(
    weather_fp,
    parse_dates=["DATE"],
    low_memory=False
)

# 2) Identify daily-summary rows
#    We assume hourly observations are NaN for the daily-summary row,
#    and that DailyAverageDryBulbTemperature is populated there.
mask = weather_raw["DailyAverageDryBulbTemperature"].notna()
weather_daily = weather_raw.loc[mask, [
    "DATE",
    "DailyAverageDryBulbTemperature",
    "DailyAverageRelativeHumidity"
]].rename(columns={
    "DailyAverageDryBulbTemperature": "avg_temp_f",
    "DailyAverageRelativeHumidity":  "avg_humidity_pct"
})

# 3) Index by date (strip time) and keep only one entry per day
weather_daily["date"] = weather_daily["DATE"].dt.date
weather_daily = weather_daily.groupby("date", as_index=True).first()
weather_daily.index = pd.to_datetime(weather_daily.index)

# 4) Load & process NYISO load: take daily max
load = pd.read_csv(
    load_fp,
    parse_dates=["RTD End Time Stamp"],
    usecols=["RTD End Time Stamp","RTD Actual Load"]
).rename(columns={"RTD End Time Stamp":"datetime","RTD Actual Load":"load_mw"})

load["date"] = load["datetime"].dt.date
load_daily = load.groupby("date", as_index=True)["load_mw"].max()
load_daily.index = pd.to_datetime(load_daily.index)

# 5) Merge weather_daily & load_daily
df_daily = pd.concat([
    weather_daily[["avg_temp_f","avg_humidity_pct"]],
    load_daily.rename("daily_max_load_mw")
], axis=1, join="inner")

# Reset index to turn the date into a column named 'date'
df_daily = df_daily.reset_index().rename(columns={"index":"date"})

# analyze if is weekend and holiday
df_daily["is_weekend"] = df_daily["date"].dt.weekday >= 5
us_holidays = holidays.US(years=df_daily["date"].dt.year.unique().tolist())
df_daily["is_holiday"] = df_daily["date"].isin(us_holidays)
df_daily = pd.get_dummies(df_daily, columns=["is_weekend", "is_holiday"], drop_first=True)

#analyze what is the max load, temp, humidity, avg temp, humidity and load
max_load = df_daily["daily_max_load_mw"].max()
max_temp = df_daily["avg_temp_f"].max()
max_humidity = df_daily["avg_humidity_pct"].max()
avg_temp = df_daily["avg_temp_f"].mean()
avg_humidity = df_daily["avg_humidity_pct"].mean()
avg_load = df_daily["daily_max_load_mw"].mean()

print(f"Max daily load: {max_load:.2f} MW")
print(f"Max daily temperature: {max_temp:.2f} F")
print(f"Max daily humidity: {max_humidity:.2f} %")
print(f"Average daily temperature: {avg_temp:.2f} F")
print(f"Average daily humidity: {avg_humidity:.2f} %")
print(f"Average daily load: {avg_load:.2f} MW")

#analyze if the max load is higher on weekend or holiday
weekend_load = df_daily[df_daily["is_weekend_True"]]["daily_max_load_mw"].mean()
holiday_load = df_daily[df_daily["is_holiday_True"]]["daily_max_load_mw"].mean()

print(f"Average daily max load on weekends: {weekend_load:.2f} MW")
print(f"Average daily max load on holidays: {holiday_load:.2f} MW")

# 6) Ready to analyze or export
print(df_daily.head())

#Save to CSV 
df_daily.to_csv("daily_weather_and_load_2023.csv", index=False)

# -----------------------------------------------------------------------------
# 7) Handle Missing Data & Outliers
# -----------------------------------------------------------------------------
# 7.1 Missing values
print("Missing values per column:\n", df_daily.isna().sum())

# Drop any days with missing weather or load
df_daily = df_daily.dropna()

# 7.2 Outlier detection via IQR and clipping
for col in ["avg_temp_f", "avg_humidity_pct", "daily_max_load_mw"]:
    Q1 = df_daily[col].quantile(0.25)
    Q3 = df_daily[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    print(f"{col}: clipping to [{lower:.1f}, {upper:.1f}]")
    df_daily[col] = df_daily[col].clip(lower, upper)

# -----------------------------------------------------------------------------
# 8) Exploratory Data Analysis (EDA)
# -----------------------------------------------------------------------------
print("\nSummary statistics:\n", df_daily.describe())

# 8.1 Time‐series plot: daily max load vs. avg temp
fig, ax1 = plt.subplots(figsize=(12,4))
ax1.plot(df_daily["date"], df_daily["daily_max_load_mw"], label="Daily Max Load (MW)")
ax2 = ax1.twinx()
ax2.plot(df_daily["date"], df_daily["avg_temp_f"], color="orange", label="Avg Temp (°F)")
ax1.set_xlabel("Date")
ax1.set_ylabel("Load (MW)")
ax2.set_ylabel("Temp (°F)")
fig.suptitle("Daily Peak Load vs. Average Temperature")
fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
plt.show()

# 8.2 Scatter & Correlation heatmap
sns.pairplot(df_daily, vars=["avg_temp_f","avg_humidity_pct","daily_max_load_mw"])
plt.suptitle("Pairplot of Weather vs. Load", y=1.02)
plt.show()

corr = df_daily[["avg_temp_f","avg_humidity_pct","daily_max_load_mw"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# -----------------------------------------------------------------------------
# 9) Identify Trends Through Decomposition of Daily Peak Load
# -----------------------------------------------------------------------------
ts = df_daily.set_index("date")["daily_max_load_mw"]

# Use weekly seasonality (period=7)
decomp = seasonal_decompose(ts, model="additive", period=7)

decomp.trend .plot(title="Trend Component of Daily Peak Load");     plt.show()
decomp.seasonal.plot(title="Seasonal Component (Weekly)");           plt.show()
decomp.resid .plot(title="Residual Component of Daily Peak Load");  plt.show()

# -----------------------------------------------------------------------------
# 10) Simple Regression: Load ~ Temp + Humidity
# -----------------------------------------------------------------------------
X = df_daily[["avg_temp_f","avg_humidity_pct"]]
y = df_daily["daily_max_load_mw"]

lr = LinearRegression().fit(X, y)
print("Regression coefficients:", dict(zip(X.columns, lr.coef_)))
print("Intercept:", lr.intercept_)
print("R² score:", lr.score(X,y))

# Plot regression line for Temperature (keeping humidity at its mean)
temp_range = np.linspace(X["avg_temp_f"].min(), X["avg_temp_f"].max(), 100)
humid_mean = X["avg_humidity_pct"].mean()
y_pred = lr.intercept_ + lr.coef_[0]*temp_range + lr.coef_[1]*humid_mean

plt.figure(figsize=(8,4))
plt.scatter(df_daily["avg_temp_f"], y, alpha=0.6, label="Actual")
plt.plot(temp_range, y_pred, color="red", label="Fit @ avg humidity")
plt.xlabel("Avg Temp (°F)")
plt.ylabel("Daily Max Load (MW)")
plt.title("Linear Fit: Load vs. Temp")
plt.legend()
plt.show()