# NYC Energy Load and Weather Analysis

This project analyzes the relationship between weather conditions and energy consumption patterns in New York City using data from 2023. The analysis combines daily weather measurements with real-time energy load data to uncover correlations and trends.

> This repository is a **forked version** of the original project created in collaboration with a classmate. It is shared here for personal reference.

## Data Sources

- **Weather Data**: NYC weather measurements including temperature and humidity
  - Source: `NYCWeatherData/New York Weather Data.csv`
  - Metrics: Daily average temperature (°F) and relative humidity (%)

- **Energy Load Data**: NYISO Real-Time Dispatch Actual Load
  - Source: `NYCEnergyLoad/OASIS_Real_Time_Dispatch_Actual_Load.csv`
  - Metrics: Daily maximum load (MW)

## Data Processing Steps

1. **Weather Data Processing**
   - Extracted daily summary rows from raw weather data
   - Selected relevant columns: temperature and humidity
   - Standardized date formatting

2. **Energy Load Processing**
   - Calculated daily maximum load from real-time data
   - Merged with weather data on matching dates

3. **Data Quality**
   - Handled missing values through removal
   - Applied IQR-based outlier detection and clipping
   - Generated clean dataset: `daily_weather_and_load_2023.csv`

## Analysis Methods

### 1. Exploratory Data Analysis (EDA)
- Time series visualization of daily peak load vs. average temperature
- Scatter plots examining relationships between variables
- Correlation analysis between weather metrics and energy load

### 2. Time Series Decomposition
- Applied additive decomposition with weekly seasonality
- Analyzed trend, seasonal, and residual components

### 3. Linear Regression Analysis
- Modeled energy load as a function of temperature and humidity
- Evaluated model performance through R² score

## Key Findings

1. **Seasonal Patterns**
   - Clear weekly seasonality in energy consumption
   - Strong correlation between temperature and energy load

2. **Temperature Impact**
   - Higher temperatures generally associated with increased energy consumption
   - Peak loads observed during summer months (July-September)

3. **Humidity Effects**
   - Secondary influence on energy consumption
   - Contributes to overall model accuracy

## Usage

1. Run the analysis script:
   ```bash
   python analyze_energy_temp.py
   ```

2. The script will:
   - Process raw data files
   - Generate visualizations
   - Output regression analysis results
   - Save processed data to CSV

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib: Data visualization
- seaborn: Statistical data visualization
- statsmodels: Time series analysis
- scikit-learn: Machine learning (regression analysis)

## Results

The analysis reveals significant correlations between weather conditions and energy consumption in NYC:

- Temperature shows a strong positive correlation with energy load
- Weekly patterns indicate consistent usage cycles
- The linear regression model demonstrates good predictive power
- Summer months exhibit the highest energy demand

Detailed visualizations and statistical results are generated during script execution.
