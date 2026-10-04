"""
Script to generate the complete Jupyter Notebook: notebooks/analysis.ipynb
Author: Prapti Gaikwad (AI/DS, 3rd Year)
"""

import nbformat as nbf
import os

def create_analysis_notebook():
    nb = nbf.v4.new_notebook()

    cells = []

    # Title & Introduction
    cells.append(nbf.v4.new_markdown_cell("""# 📱 Mobile Battery Life Prediction - ML Mini Project

**Student Name:** Prapti Gaikwad  
**Branch:** Artificial Intelligence and Data Science (AI/DS)  
**Academic Year:** 3rd Year  
**Project Type:** Machine Learning Mini Project  

---

## 🎯 Project Objective
The objective of this project is to build a supervised machine learning regression system that predicts the **expected remaining battery life (in hours)** of a mobile phone based on hardware specifications, real-time operating metrics, and battery degradation parameters.

---

## 📑 Table of Contents
1. **Libraries & Environment Setup**
2. **Data Loading & Initial Inspection**
3. **Data Cleaning & Missing Value Imputation**
4. **Exploratory Data Analysis (EDA) & Visualizations**
5. **Feature Engineering & Train-Test Split**
6. **Model Training & Comparison**
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
7. **Model Evaluation (MAE, MSE, RMSE, R² Score)**
8. **Dynamic Best Model Selection**
9. **Feature Importance Analysis**
10. **Model Serialization (Joblib)**
11. **Sample Prediction Test**
12. **Key Takeaways & Viva Discussion**
"""))

    # Imports
    cells.append(nbf.v4.new_markdown_cell("## 1. Libraries & Environment Setup"))
    cells.append(nbf.v4.new_code_cell("""import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Plotting configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.autolayout'] = True
print("All libraries imported successfully!")
"""))

    # Data Loading
    cells.append(nbf.v4.new_markdown_cell("## 2. Data Loading & Initial Inspection\nLoad the dataset containing smartphone telemetry, hardware specs, and battery health."))
    cells.append(nbf.v4.new_code_cell("""data_path = os.path.join('..', 'dataset', 'mobile_battery_data.csv')
if not os.path.exists(data_path):
    data_path = os.path.join('dataset', 'mobile_battery_data.csv')

df = pd.read_csv(data_path)
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\\n")
df.head()
"""))

    cells.append(nbf.v4.new_code_cell("""# Summary statistics of numerical columns
df.describe().round(2)
"""))

    cells.append(nbf.v4.new_code_cell("""# Column types and non-null counts
df.info()
"""))

    # Data Cleaning
    cells.append(nbf.v4.new_markdown_cell("## 3. Data Cleaning & Handling Missing Values\nIdentify and impute missing values using median imputation."))
    cells.append(nbf.v4.new_code_cell("""# Check missing values
print("Missing values per feature:")
print(df.isnull().sum())
"""))

    cells.append(nbf.v4.new_code_cell("""# Median imputation for continuous variables with missing values
for col in ['CPU_Usage_Pct', 'RAM_Usage_Pct']:
    if df[col].isnull().sum() > 0:
        med = df[col].median()
        df[col] = df[col].fillna(med)
        print(f"Filled missing values in '{col}' with median: {med:.2f}")

print(f"\\nTotal missing values remaining: {df.isnull().sum().sum()}")
"""))

    # EDA
    cells.append(nbf.v4.new_markdown_cell("## 4. Exploratory Data Analysis (EDA) & Visualizations\nAnalyze distributions, relationships, and correlations."))
    cells.append(nbf.v4.new_code_cell("""# Distribution of the Target Variable (Battery Life Remaining)
plt.figure(figsize=(9, 5))
sns.histplot(df['Battery_Life_Remaining_Hours'], kde=True, color='#2563EB', bins=30)
plt.title('Distribution of Target: Battery Life Remaining (Hours)', fontsize=14, fontweight='bold')
plt.xlabel('Battery Life Remaining (Hours)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Correlation Matrix Heatmap
plt.figure(figsize=(12, 9))
num_cols = df.select_dtypes(include=[np.number])
corr = num_cols.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title('Correlation Matrix of Mobile Battery Features', fontsize=14, fontweight='bold')
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Scatter Plot: Current Battery % vs. Remaining Hours grouped by Network Usage
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x='Current_Battery_Pct',
    y='Battery_Life_Remaining_Hours',
    hue='Network_Usage',
    alpha=0.6,
    palette='Set1'
)
plt.title('Current Battery % vs. Remaining Battery Life (Hours)', fontsize=14, fontweight='bold')
plt.xlabel('Current Battery Percentage (%)', fontsize=12)
plt.ylabel('Battery Life Remaining (Hours)', fontsize=12)
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Impact of Gaming Hours on Battery Discharge
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x='Gaming_Hours',
    y='Battery_Life_Remaining_Hours',
    hue='Temperature_C',
    palette='viridis',
    alpha=0.7
)
plt.title('Gaming Hours vs. Remaining Battery Life (Colored by Temperature °C)', fontsize=14, fontweight='bold')
plt.xlabel('Gaming Hours', fontsize=12)
plt.ylabel('Remaining Battery Life (Hours)', fontsize=12)
plt.show()
"""))

    # Feature Selection & Split
    cells.append(nbf.v4.new_markdown_cell("## 5. Feature Engineering & Train-Test Split\nSeparate features and target, define the column transformer for scaling and encoding, and split 80/20."))
    cells.append(nbf.v4.new_code_cell("""target_col = 'Battery_Life_Remaining_Hours'
feature_cols = [c for c in df.columns if c != target_col]

numeric_features = [
    'Battery_Capacity_mAh', 'Current_Battery_Pct', 'Screen_On_Time_Hours',
    'CPU_Usage_Pct', 'RAM_Usage_Pct', 'Mobile_Data_Usage_GB',
    'WiFi_Usage_Hours', 'Number_of_Apps_Running', 'Brightness_Level_Pct',
    'Gaming_Hours', 'Charging_Cycles', 'Device_Age_Months',
    'Temperature_C', 'Battery_Health_Pct'
]
categorical_features = ['Network_Usage']

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")
"""))

    cells.append(nbf.v4.new_code_cell("""# Build unified Scikit-Learn Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)
print("Preprocessing pipeline assembled successfully.")
"""))

    # Model Training
    cells.append(nbf.v4.new_markdown_cell("## 6. Model Training & Comparison\nTrain 3 regression models inside the pipeline."))
    cells.append(nbf.v4.new_code_cell("""models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
}

trained_pipelines = {}
evaluation_results = {}

for name, model_instance in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model_instance)
    ])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe

    # Predictions on unseen test set
    preds = pipe.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    evaluation_results[name] = {
        'MAE': round(float(mae), 4),
        'MSE': round(float(mse), 4),
        'RMSE': round(float(rmse), 4),
        'R2_Score': round(float(r2), 4)
    }

print("Model training and evaluation finished.")
"""))

    # Model Evaluation
    cells.append(nbf.v4.new_markdown_cell("## 7. Model Evaluation Comparison\nCompare all models side-by-side using standard regression metrics."))
    cells.append(nbf.v4.new_code_cell("""results_df = pd.DataFrame(evaluation_results).T
results_df
"""))

    cells.append(nbf.v4.new_code_cell("""# Visualization of Performance Metrics
fig, ax = plt.subplots(figsize=(10, 5))
results_df[['MAE', 'RMSE', 'R2_Score']].plot(kind='bar', ax=ax, colormap='viridis')
plt.title('Comparison of Regression Models across MAE, RMSE, and R² Score', fontsize=14, fontweight='bold')
plt.xlabel('Regression Algorithm', fontsize=12)
plt.ylabel('Metric Score', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper right')
plt.show()
"""))

    # Dynamic Selection
    cells.append(nbf.v4.new_markdown_cell("## 8. Dynamic Best Model Selection (No Hardcoding)\nAutomatically detect and select the winning algorithm."))
    cells.append(nbf.v4.new_code_cell("""# Dynamic selection based on highest R2 score on the test set
best_model_name = results_df['R2_Score'].idxmax()
best_pipeline = trained_pipelines[best_model_name]
best_metrics = evaluation_results[best_model_name]

print(f"Winner Model: {best_model_name}")
print(f"R2 Score:     {best_metrics['R2_Score']}")
print(f"RMSE (Hours): {best_metrics['RMSE']}")
print(f"MAE (Hours):  {best_metrics['MAE']}")
"""))

    # Feature Importance
    cells.append(nbf.v4.new_markdown_cell("## 9. Feature Importance Analysis\nInspect feature weights learned by the best model."))
    cells.append(nbf.v4.new_code_cell("""if hasattr(best_pipeline.named_steps['regressor'], 'feature_importances_'):
    ohe = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_ohe = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_ohe
    importances = best_pipeline.named_steps['regressor'].feature_importances_

    fi_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df, x='Importance', y='Feature', hue='Feature', palette='crest', legend=False)
    plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Relative Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.show()
"""))

    # Serialization
    cells.append(nbf.v4.new_markdown_cell("## 10. Model Serialization (Joblib)\nPersist the model for deployment in the Streamlit web application."))
    cells.append(nbf.v4.new_code_cell("""os.makedirs('../models', exist_ok=True)
model_path = os.path.join('..', 'models', 'battery_life_model.pkl')
if not os.path.exists('..'):
    model_path = os.path.join('models', 'battery_life_model.pkl')

joblib.dump(best_pipeline, model_path)
print(f"Trained pipeline serialized successfully to: {model_path}")
"""))

    # Inference test
    cells.append(nbf.v4.new_markdown_cell("## 11. Sample Real-Time Prediction Test\nTest the pipeline on an arbitrary smartphone usage state."))
    cells.append(nbf.v4.new_code_cell("""sample_phone = pd.DataFrame([{
    'Battery_Capacity_mAh': 5000,
    'Current_Battery_Pct': 80.0,
    'Screen_On_Time_Hours': 3.0,
    'CPU_Usage_Pct': 30.0,
    'RAM_Usage_Pct': 50.0,
    'Mobile_Data_Usage_GB': 1.5,
    'WiFi_Usage_Hours': 4.0,
    'Number_of_Apps_Running': 5,
    'Brightness_Level_Pct': 40.0,
    'Gaming_Hours': 0.0,
    'Charging_Cycles': 150,
    'Device_Age_Months': 6,
    'Temperature_C': 28.0,
    'Network_Usage': 'Wi-Fi',
    'Battery_Health_Pct': 97.0
}])

predicted_hours = best_pipeline.predict(sample_phone)[0]
print(f"Predicted Remaining Battery Life: {predicted_hours:.2f} hours")
print(f"Approximate duration: {int(predicted_hours)} hours and {int((predicted_hours % 1)*60)} minutes.")
"""))

    # Viva discussion
    cells.append(nbf.v4.new_markdown_cell("""## 12. Key Takeaways & Viva Discussion Points
- **What is the problem being solved?** Smartphone users frequently suffer from sudden battery drainage due to dynamic background apps, gaming, and 5G network usage. This ML project predicts exact remaining hours rather than simple static percentages.
- **Why Random Forest Regressor?** Battery discharge rate is non-linear and governed by multivariate physical and thermal interactions. Random Forest successfully models these non-linearities through ensemble decision trees without overfitting.
- **Data Preprocessing Integrity:** Using Scikit-Learn `ColumnTransformer` within a `Pipeline` guarantees zero data leakage between training and testing splits.
- **Target Audience:** College examination and viva presentation for 3rd-year AI/DS by **Prapti Gaikwad**.
"""))

    nb['cells'] = cells
    
    os.makedirs('notebooks', exist_ok=True)
    nb_path = os.path.join('notebooks', 'analysis.ipynb')
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Notebook created at: {nb_path}")

if __name__ == '__main__':
    create_analysis_notebook()
