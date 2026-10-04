"""
Model Training and Evaluation Pipeline
Topic: Mobile Battery Life Prediction
Student Name: Prapti Gaikwad
Branch: Artificial Intelligence and Data Science (AI/DS)
Year: 3rd Year

This script performs:
1. Dataset loading and inspection
2. Data cleaning & missing value imputation
3. Exploratory Data Analysis (EDA) figure generation
4. Feature selection and preprocessing pipeline
5. Train/Test splitting (80/20)
6. Training 3 Regression Models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
7. Model evaluation: MAE, MSE, RMSE, R2 Score
8. Dynamic automated selection of the best model (without hardcoding)
9. Serialization of the best model using joblib into 'models/battery_life_model.pkl'
10. Export of performance metrics to 'models/model_metadata.json'
"""

import os
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

# Set aesthetic visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def run_pipeline():
    print("=" * 70)
    print("      MOBILE BATTERY LIFE PREDICTION - MACHINE LEARNING PIPELINE")
    print("      Student: Prapti Gaikwad | Branch: AI & DS | Year: 3rd Year")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Load and Inspect the Dataset
    # -------------------------------------------------------------
    data_path = os.path.join('dataset', 'mobile_battery_data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_dataset.py first.")

    df = pd.read_csv(data_path)
    print(f"\n[INFO] Dataset loaded successfully.")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n[INFO] First 5 Rows of the Dataset:")
    print(df.head())

    # -------------------------------------------------------------
    # 2. Data Cleaning & Handling Missing Values
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[STEP 1] Data Cleaning & Missing Value Check")
    print("-" * 50)
    missing_counts = df.isnull().sum()
    print("Missing values per column before cleaning:")
    print(missing_counts[missing_counts > 0])

    # We impute numeric features with median (robust to outliers)
    numeric_imputer_cols = ['CPU_Usage_Pct', 'RAM_Usage_Pct']
    for col in numeric_imputer_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing values in '{col}' with median: {median_val:.2f}")

    print("\nMissing values after cleaning:")
    print(df.isnull().sum().sum(), "total missing values remain.")

    # -------------------------------------------------------------
    # 3. Exploratory Data Analysis (EDA) & Visualization
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[STEP 2] Exploratory Data Analysis & Graph Generation")
    print("-" * 50)
    os.makedirs('screenshots', exist_ok=True)

    # Plot 1: Target Variable Distribution (Battery Life Remaining)
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Battery_Life_Remaining_Hours'], kde=True, color='#1f77b4', bins=30)
    plt.title('Distribution of Target: Battery Life Remaining (Hours)', fontsize=14, fontweight='bold')
    plt.xlabel('Battery Life Remaining (Hours)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    dist_path = os.path.join('screenshots', 'battery_life_distribution.png')
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"Saved: {dist_path}")

    # Plot 2: Correlation Heatmap for Numerical Features
    plt.figure(figsize=(12, 9))
    num_df = df.select_dtypes(include=[np.number])
    corr_matrix = num_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Heatmap of Mobile Usage Features', fontsize=14, fontweight='bold')
    corr_path = os.path.join('screenshots', 'correlation_heatmap.png')
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved: {corr_path}")

    # Plot 3: Scatter Plot - Battery Percentage vs Battery Life Remaining
    plt.figure(figsize=(8, 5))
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
    plt.ylabel('Remaining Battery Life (Hours)', fontsize=12)
    scatter_path = os.path.join('screenshots', 'battery_pct_vs_remaining_life.png')
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Saved: {scatter_path}")

    # -------------------------------------------------------------
    # 4. Feature Selection & Train/Test Split
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[STEP 3] Feature Selection & Train-Test Split")
    print("-" * 50)

    target_column = 'Battery_Life_Remaining_Hours'
    feature_columns = [col for col in df.columns if col != target_column]

    numeric_features = [
        'Battery_Capacity_mAh',
        'Current_Battery_Pct',
        'Screen_On_Time_Hours',
        'CPU_Usage_Pct',
        'RAM_Usage_Pct',
        'Mobile_Data_Usage_GB',
        'WiFi_Usage_Hours',
        'Number_of_Apps_Running',
        'Brightness_Level_Pct',
        'Gaming_Hours',
        'Charging_Cycles',
        'Device_Age_Months',
        'Temperature_C',
        'Battery_Health_Pct'
    ]
    categorical_features = ['Network_Usage']

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Features count: {len(feature_columns)} ({len(numeric_features)} numeric, {len(categorical_features)} categorical)")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")

    # -------------------------------------------------------------
    # 5. Preprocessing Pipelines
    # -------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )

    # -------------------------------------------------------------
    # 6. Model Training & Comparison
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[STEP 4] Model Training & Comparison (At least 3 models)")
    print("-" * 50)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    }

    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predictions
        y_pred = pipeline.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'MAE': round(float(mae), 4),
            'MSE': round(float(mse), 4),
            'RMSE': round(float(rmse), 4),
            'R2_Score': round(float(r2), 4)
        }

    # Display comparison table
    results_df = pd.DataFrame(results).T
    print("\n" + "=" * 50)
    print("MODEL EVALUATION COMPARISON TABLE:")
    print("=" * 50)
    print(results_df.to_string())

    # Plot 4: Model Performance Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    metrics_to_plot = ['MAE', 'RMSE', 'R2_Score']
    results_df[metrics_to_plot].plot(kind='bar', figsize=(10, 5), colormap='viridis')
    plt.title('Model Performance Comparison (MAE, RMSE, R² Score)', fontsize=14, fontweight='bold')
    plt.ylabel('Score Value', fontsize=12)
    plt.xlabel('Regression Model', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    comp_path = os.path.join('screenshots', 'model_comparison.png')
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_path}")

    # -------------------------------------------------------------
    # 7. Dynamic Best Model Selection (NO HARDCODING)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[STEP 5] Dynamic Automated Model Selection")
    print("-" * 50)
    # We select dynamically by highest R2 Score (or lowest RMSE)
    best_model_name = results_df['R2_Score'].idxmax()
    best_pipeline = fitted_pipelines[best_model_name]
    best_metrics = results[best_model_name]

    print(f"\n[BEST MODEL] Best Model dynamically selected: '{best_model_name}'")
    print(f"   Reason: Highest R2 Score = {best_metrics['R2_Score']} | Lowest RMSE = {best_metrics['RMSE']}")

    # Plot 5: Feature Importance (if tree-based)
    if hasattr(best_pipeline.named_steps['regressor'], 'feature_importances_'):
        # Extract feature names after one-hot encoding
        ohe = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_features_ohe = list(ohe.get_feature_names_out(categorical_features))
        all_features = numeric_features + cat_features_ohe
        importances = best_pipeline.named_steps['regressor'].feature_importances_

        fi_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=fi_df, x='Importance', y='Feature', hue='Feature', palette='crest', legend=False)
        plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Relative Importance Score', fontsize=12)
        plt.ylabel('Feature Name', fontsize=12)
        fi_path = os.path.join('screenshots', 'feature_importance.png')
        plt.savefig(fi_path, dpi=300)
        plt.close()
        print(f"Saved: {fi_path}")

    # -------------------------------------------------------------
    # 8. Save the Best Trained Model and Metadata
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[STEP 6] Model Serialization (Joblib)")
    print("-" * 50)
    os.makedirs('models', exist_ok=True)
    model_save_path = os.path.join('models', 'battery_life_model.pkl')
    joblib.dump(best_pipeline, model_save_path)
    print(f"Trained pipeline model successfully saved to: {model_save_path}")

    # Metadata for Streamlit UI
    metadata = {
        'student_info': {
            'student_name': 'Prapti Gaikwad',
            'branch': 'Artificial Intelligence and Data Science (AI/DS)',
            'year': '3rd Year',
            'project_title': 'Mobile Battery Life Prediction',
            'project_type': 'Machine Learning Mini Project'
        },
        'best_model': best_model_name,
        'best_metrics': best_metrics,
        'all_model_results': results,
        'feature_columns': feature_columns,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'total_samples': int(len(df))
    }

    metadata_path = os.path.join('models', 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata and metrics saved to: {metadata_path}")
    print("\n" + "=" * 70)
    print("      TRAINING AND EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    run_pipeline()
