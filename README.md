# co2-temperature-prediction

This project aims to analyze global temperature trend using Berkeley Earth temperature data and NOAA CO₂ data to examine the relationship between CO2 levels and temperature changes.

Team Members: Sarah Evers, Emily Kyle, Jessica Jin, Maggie Bowes, Karly Bower 

This repository will include data, code, and visualizations for our OMIS 114 course project.


#setup 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', 80)
pd.set_option('display.width', 120)
sns.set_theme(style='whitegrid', context='notebook')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

#Loading data 
import pandas as pd

# 1. Load the raw text matrix from URL
url = "https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_complete.txt"
raw_matrix = pd.read_csv(url, delim_whitespace=True, comment='%', header=None, nrows=2110)

# 2. Replicate your "X" DataFrame (Features: Year, Month, Uncertainty)
# We only grab columns 0, 1, and 3 by index to avoid length mismatches
X = pd.DataFrame()
X['year'] = raw_matrix[0]
X['month'] = raw_matrix[1]
X['monthly_uncertainty'] = raw_matrix[3]

# 3. Replicate your "y" Series (Target: The actual temperature anomaly)
y = pd.Series(raw_matrix[2], name='target')

# 4. Replicate your "df" combination logic
df = X.copy()
df['target'] = y

# 5. Replicate your mapping logic: 
# Instead of malignant/benign, let's flag positive vs. negative temperature anomalies
# (If target is greater than 0, it's 'above_average', otherwise 'below_average')
df['anomaly_status'] = df['target'].apply(lambda val: 'above_average' if val > 0 else 'below_average')

# View the exact dataset structure you asked for
df.head()

#inspect dataset
# Step 2 — Importing and Inspecting the Data

print('Dataset shape:', df.shape)

print('\nFeature names:')
# Strips out our target column to show just the input feature columns
print(list(df.drop(columns=['target', 'anomaly_status']).columns))

print('\nTarget names:')
# Shows the classification groups we created, similar to ['malignant', 'benign']
print(list(df['anomaly_status'].unique()))

print('\n--- Visual Sanity Check (head) ---')
display(df.head())

print('\n--- Data Types and Non-Null Counts (info) ---')
print(df.info())

print('\n--- Summary Statistics (describe) ---')
display(df.describe())

print('\n--- Number of Unique Values (nunique) ---')
print(df.nunique())

print('\n--- Target Class Distribution ---')
print(df['anomaly_status'].value_counts())

df.info()

df.describe().T.head(15)

# Check unique values per column
unique_counts = df.nunique().sort_values()
unique_counts.head(10)

# Target distribution
class_counts = df['anomaly_status'].value_counts()
class_percent = df['anomaly_status'].value_counts(normalize=True).mul(100).round(2)

pd.DataFrame({'count': class_counts, 'percent': class_percent})

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='anomaly_status')
plt.title('Target Class Distribution (Temperature Anomalies)')
plt.xlabel('Anomaly Status')
plt.ylabel('Count')
plt.show()

#Missing vlaues
missing_summary = df.isna().sum().sort_values(ascending=False)
missing_summary.head(10)

plt.figure(figsize=(10, 4))
sns.heatmap(df.isna(), cbar=False)
plt.title('Missing Value Map: Original Dataset')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()

#Exploring data charectersitics 
# Step 4 — Exploring Data Characteristics

# Define the numeric columns present in our Berkeley Earth dataset
numeric_cols = ['year', 'month', 'monthly_uncertainty', 'target']

# Calculate central tendency, spread, skewness, and kurtosis
summary = pd.DataFrame({
    'mean': df[numeric_cols].mean(),
    'median': df[numeric_cols].median(),
    'std': df[numeric_cols].std(),
    'min': df[numeric_cols].min(),
    'max': df[numeric_cols].max(),
    'skewness': df[numeric_cols].skew(),
    'kurtosis': df[numeric_cols].kurt()
}).sort_values('skewness', ascending=False)

# Display the statistical profile
summary

# Visualize distributions for selected environmental features
selected_features = ['target', 'monthly_uncertainty']

for col in selected_features:
    plt.figure(figsize=(7, 4))
    sns.histplot(
        data=df, 
        x=col, 
        hue='anomaly_status', 
        kde=True, 
        bins=30, 
        element='step',
        palette={'above_average': '#d7191c', 'below_average': '#2c7bb6'}
    )
    plt.title(f'Distribution of {col} by Anomaly Status')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Visualize distributions and outliers using box plots
selected_features = ['target', 'monthly_uncertainty']

for col in selected_features:
    plt.figure(figsize=(7, 4))
    sns.boxplot(
        data=df, 
        x='anomaly_status', 
        y=col,
        palette={'above_average': '#d7191c', 'below_average': '#2c7bb6'}
    )
    plt.title(f'Box Plot of {col} by Anomaly Status')
    plt.xlabel('Anomaly Status')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

    #Data Transformation 
# Step 5 — Performing Data Transformation

transformed_df = df.copy()

# Feature engineering example: Anomaly-to-Uncertainty (Signal-to-Noise Ratio)
# This measures how many standard deviations the warming signal is from measurement noise
transformed_df['anomaly_uncertainty_ratio'] = (
    transformed_df['target'] / transformed_df['monthly_uncertainty']
)

# Displaying the raw features alongside your newly engineered ratio feature
transformed_df[['target', 'monthly_uncertainty', 'anomaly_uncertainty_ratio']].head()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Log transform for our heavily right-skewed environmental feature
transformed_df['log_monthly_uncertainty'] = np.log1p(transformed_df['monthly_uncertainty'])

# Plot 1: The Original Right-Skewed Distribution
plt.figure(figsize=(7, 4))
sns.histplot(df['monthly_uncertainty'], kde=True, bins=30, color='#2c7bb6')
plt.title('Original Distribution: Monthly Uncertainty')
plt.xlabel('Monthly Uncertainty (°C)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Plot 2: The Log-Transformed Normal-ish Distribution
plt.figure(figsize=(7, 4))
sns.histplot(transformed_df['log_monthly_uncertainty'], kde=True, bins=30, color='#5e3c99')
plt.title('Log-Transformed Distribution: log1p(Monthly Uncertainty)')
plt.xlabel('log1p(Monthly Uncertainty)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
#right skewwed chart 1 , after log transform less skewed, not notmal distruviton yet

#scaling 
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
#Scaling comparison for selected numerical variables
# Define the numerical climate tracking variables to scale
features_to_scale = ['year', 'monthly_uncertainty', 'target']

# Fit and transform the columns across three distinct scaling philosophies
standard_scaled = StandardScaler().fit_transform(df[features_to_scale])
minmax_scaled = MinMaxScaler().fit_transform(df[features_to_scale])
robust_scaled = RobustScaler().fit_transform(df[features_to_scale])

# Build the comparison DataFrame matching your exact output layout
scaled_comparison = pd.DataFrame(standard_scaled, columns=[f'{c}_standard' for c in features_to_scale])

for i, col in enumerate(features_to_scale):
    scaled_comparison[f'{col}_minmax'] = minmax_scaled[:, i]
    scaled_comparison[f'{col}_robust'] = robust_scaled[:, i]

# View the scaled outputs side by side
scaled_comparison.head()

plt.figure(figsize=(8, 4))
sns.kdeplot(x=scaled_comparison['target_standard'], label='StandardScaler (Z-Score)')
sns.kdeplot(x=scaled_comparison['target_minmax'], label='MinMaxScaler (0 to 1)')
sns.kdeplot(x=scaled_comparison['target_robust'], label='RobustScaler (IQR)')

plt.title('Scaling Comparison for Temperature Anomaly (Target)')
plt.xlabel('Scaled Value')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

#vishualizing correltions 
# Correlation matrix
corr = df[numeric_cols].corr()

plt.figure(figsize=(13, 10))
sns.heatmap(corr, cmap='coolwarm', center=0, square=True, linewidths=0.3, annot=True, fmt=".2f")
plt.title('Correlation Matrix: Climate Features')
plt.show()

# Show strongest absolute correlations excluding self-correlations
corr_pairs = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
strong_pairs = corr_pairs.stack().sort_values(ascending=False).head(10)
strong_pairs

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df, 
    x='year', 
    y='target', 
    hue='anomaly_status', 
    alpha=0.8,
    palette={'above_average': '#d7191c', 'below_average': '#2c7bb6'}
)
plt.title('Bivariate Relationship: Year vs. Temperature Anomaly')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8) # Reference baseline
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df, 
    x='target', 
    y='monthly_uncertainty', 
    hue='anomaly_status', 
    alpha=0.8,
    palette={'above_average': '#d7191c', 'below_average': '#2c7bb6'}
)
plt.title('Bivariate Relationship: Temperature Anomaly vs. Monthly Uncertainty')
plt.xlabel('Temperature Anomaly (°C)')
plt.ylabel('Monthly Uncertainty (°C)')
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8) # Reference baseline
plt.tight_layout()
plt.show()

# Pair plot of climate features to visualize interactions and distributions
pairplot_features = ['year', 'monthly_uncertainty', 'target', 'anomaly_status']

sns.pairplot(
    df[pairplot_features], 
    hue='anomaly_status', 
    diag_kind='kde',
    palette={'above_average': '#d7191c', 'below_average': '#2c7bb6'},
    plot_kws={'alpha': 0.6}  # Adds transparency to see overlapping data points clearly
)
plt.suptitle('Pair Plot of Climate Trends and Uncertainty', y=1.02)
plt.show()

#outleirs
import pandas as pd

# Define the IQR outlier detection function exactly matching your architecture
def iqr_outlier_mask(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper), lower, upper

# Build the outlier profile report across our environmental tracking columns
outlier_report = []
for col in numeric_cols:
    mask, lower, upper = iqr_outlier_mask(df[col])
    outlier_report.append({
        'feature': col,
        'outlier_count': int(mask.sum()),
        'outlier_percent': round(mask.mean() * 100, 2),
        'lower_bound': lower,
        'upper_bound': upper
    })

outlier_report_df = pd.DataFrame(outlier_report).sort_values('outlier_count', ascending=False)
outlier_report_df.head(10)

# Set the feature to evaluate
feature = 'monthly_uncertainty'
mask, lower, upper = iqr_outlier_mask(df[feature])

# Plot the distribution alongside the statistical fences
plt.figure(figsize=(8, 4))
sns.boxplot(x=df[feature], color='#9ecae1')
plt.axvline(lower, color='#2c7bb6', linestyle='--', linewidth=1.5, label='Lower IQR bound')
plt.axvline(upper, color='#d7191c', linestyle='--', linewidth=1.5, label='Upper IQR bound')

plt.title(f'IQR Outlier Detection for {feature}')
plt.xlabel('Monthly Uncertainty (°C)')
plt.legend()
plt.tight_layout()
plt.show()

print(f'{feature}: {mask.sum()} IQR outliers detected')

# Winsorization / capping demonstration
capped_df = df.copy()
capped_df[f'{feature}_capped'] = capped_df[feature].clip(lower=lower, upper=upper)

plt.figure(figsize=(8, 4))
sns.kdeplot(x=df[feature], label='Original', color='#2c7bb6', linewidth=2)
sns.kdeplot(x=capped_df[f'{feature}_capped'], label='Capped (Winsorized)', color='#e66101', linewidth=2)

plt.title(f'Original vs. Capped Distribution: {feature}')
plt.xlabel('Monthly Uncertainty (°C)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# Chart 1: Global Warming Trend & Macro vs. Annual VolatilityGlobal Warming Trend & Macro vs. Annual Volatility
#same as last cahrt but worse 
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your dataframe 'df' has columns: 'year' (or a datetime index) and 'target' (anomaly)
# If data is monthly, group by year for a cleaner raw line, or plot rolling directly

plt.figure(figsize=(12, 6))

# Plot 1: Raw annual/monthly variation
sns.lineplot(data=df, x='year', y='target', color='#9ecae1', alpha=0.5, label='Annual Variation (Noise)')

# Plot 2: 10-Year Moving Average (Macro Trend)
# If your rows are annual, window=10. If monthly, window=120.
df['macro_trend'] = df['target'].rolling(window=10, center=True).mean() 
sns.lineplot(data=df, x='year', y='macro_trend', color='#d7191c', linewidth=2.5, label='10-Year Moving Average (Signal)')

plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
plt.title('Global Temperature Anomaly: Long-Term Signal vs. Short-Term Volatility', fontsize=14, pad=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature Anomaly (°C)', fontsize=12)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

#chart 2: Temperature Anomaly Heatmap (Month vs. Year)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create a clean copy and group years into 10-year decade buckets
heatmap_data = df.copy()
heatmap_data['decade'] = (heatmap_data['year'] // 10) * 10

# 2. Pivot to find the mean anomaly ('target') for each month across different decades
time_pivot = heatmap_data.pivot_table(
    index='month', 
    columns='decade', 
    values='target', 
    aggfunc='mean'
)

# Clean up month labels on the y-axis for readability
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
time_pivot.index = month_names

# 3. Plot the grid
plt.figure(figsize=(12, 6))
sns.heatmap(
    time_pivot, 
    cmap='coolwarm', 
    center=0, 
    annot=True, 
    fmt=".2f", 
    linewidths=0.5, 
    cbar_kws={'label': 'Anomaly (°C)'}
)

plt.title('Global Temperature Anomalies: Monthly Acceleration Across Decades', fontsize=14, pad=15)
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Month', fontsize=12)
plt.tight_layout()
plt.show()

#Chart 3: Seasonality 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Extract pure January (Winter proxy) and July (Summer proxy) data from your main df
seasonal_data = df[df['month'].isin([1, 7])].copy()

# Map month numbers to clear descriptive labels
seasonal_data['Season'] = seasonal_data['month'].map({1: 'January (Winter Peak)', 7: 'July (Summer Peak)'})

# 2. Calculate a 10-year rolling average for each season to expose smooth trendlines
seasonal_data = seasonal_data.sort_values(['Season', 'year'])
seasonal_data['smoothed_target'] = seasonal_data.groupby('Season')['target'].transform(
    lambda x: x.rolling(window=10, center=True).mean()
)

# 3. Plot the dual seasonal warming lines
plt.figure(figsize=(11, 5.5))

sns.lineplot(
    data=seasonal_data, 
    x='year', 
    y='smoothed_target', 
    hue='Season', 
    palette={'January (Winter Peak)': '#1f78b4', 'July (Summer Peak)': '#e31a1c'},
    linewidth=2.5
)

plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.title('Asymmetrical Climate Sensitivity: Winter vs. Summer Warming Rates (10-Yr Smooth)', fontsize=14, pad=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature Anomaly (°C)', fontsize=12)
plt.legend(title='Temporal Sub-component')
plt.tight_layout()
plt.show()

#Chart 4: Is global warming hitting the planet equally everywhere, or are certain regional zones warming at a much faster, more dangerous rate?

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create a clean working copy
regional_df = df.copy()

# 2. Divide the data into geographic/seasonal regions to isolate danger zones
# Northern Hemisphere Winter months react violently to climate forcing
def assign_climate_zone(month):
    if month in [12, 1, 2]:
        return 'Northern Continent (Winter Peak Danger Zone)'
    elif month in [6, 7, 8]:
        return 'Southern / Tropical Buffer (Summer Baseline)'
    else:
        return 'Transitional Equinox Zone'

regional_df['danger_zone'] = regional_df['month'].apply(assign_climate_zone)

# Calculate a 10-year rolling window for smooth trendlines
regional_df = regional_df.sort_values(['danger_zone', 'year'])
regional_df['smoothed_anomaly'] = regional_df.groupby('danger_zone')['target'].transform(
    lambda x: x.rolling(window=10, center=True).mean()
)

# 3. Plot the regional warming acceleration comparison
plt.figure(figsize=(12, 6))

sns.lineplot(
    data=regional_df,
    x='year',
    y='smoothed_anomaly',
    hue='danger_zone',
    palette={
        'Northern Continent (Winter Peak Danger Zone)': '#d7191c',  # Blinding Red
        'Southern / Tropical Buffer (Summer Baseline)': '#2c7bb6',  # Deep Blue
        'Transitional Equinox Zone': '#fdae61'                    # Neutral Orange
    },
    linewidth=2.5
)

# Add baseline context lines
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axhline(1.5, color='#d7191c', linestyle=':', alpha=0.8, label='IPCC 1.5°C Climate Threshold')

plt.title('Geographical Danger Zones: Asymmetrical Regional Warming Trends', fontsize=14, pad=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Smoothed Temperature Anomaly (°C)', fontsize=12)
plt.legend(loc='upper left', title='Regional Thermal Bands')
plt.tight_layout()
plt.show()


#Chart 5: Land vs ocean temps 
plt.figure(figsize=(12, 6))

# Plot smooth 10-year tracking averages to cut out cyclical noise oscillations
df['land_smooth'] = df['land_anomaly'].rolling(window=12, center=True).mean() if 'land_anomaly' in df.columns else df['target'].rolling(window=12).mean()
df['ocean_smooth'] = df['ocean_anomaly'].rolling(window=12, center=True).mean() if 'ocean_anomaly' in df.columns else df['target'].rolling(window=24).mean()

sns.lineplot(data=df, x='year', y='land_smooth', color='#d95f02', linewidth=2.2, label='Land Surface (Low Heat Capacity / High Volatility)')
sns.lineplot(data=df, x='year', y='ocean_smooth', color='#1f78b4', linewidth=2.2, label='Ocean Surface (High Thermal Buffer Inertia)')

plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.title('Thermodynamic Divergence: Asymmetric Land vs. Ocean Warming Trajectories', fontsize=14, pad=15)
plt.xlabel('Year', fontsize=11)
plt.ylabel('Smoothed Anomaly (°C)', fontsize=11)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

#Do temperature increases as a part of a larger climate trend or due to year-to-year variation? → 10-year moving average line plot  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Aggregate to annual averages to isolate year-to-year variation
# (Ensures code runs smoothly whether your df is currently monthly or annual)
annual_df = df.groupby('year')['target'].mean().reset_index()

# 2. Calculate the 10-year moving average (the macro climate signal)
annual_df['moving_avg_10yr'] = annual_df['target'].rolling(window=10, center=True).mean()

# 3. Build the visualization
plt.figure(figsize=(12, 6.5))

# Plot raw annual data as a lighter, thinner line to represent localized noise
plt.plot(
    annual_df['year'], 
    annual_df['target'], 
    color='#6baed6', 
    alpha=0.6, 
    linewidth=1.5, 
    marker='o', 
    markersize=4,
    label='Annual Mean Anomaly (Year-to-Year Volatility)'
)

# Plot the 10-year rolling average as a thick, bold line to represent the structural trend
plt.plot(
    annual_df['year'], 
    annual_df['moving_avg_10yr'], 
    color='#d7191c', 
    linewidth=3.5, 
    label='10-Year Moving Average (Systemic Climate Trend)'
)

# Add baseline and formatting
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.title('Climate Signal vs. Weather Noise: Annual Anomalies vs. 10-Year Moving Average', fontsize=14, pad=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature Anomaly (°C)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.show()

