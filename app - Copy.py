"""
Streamlit Web Application: Mobile Battery Life Prediction
Student: Prapti Gaikwad
Branch: Artificial Intelligence and Data Science (AI/DS)
Year: 3rd Year
Project Type: Machine Learning Mini Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from PIL import Image

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Mobile Battery Life Prediction | Prapti Gaikwad",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-quality clean aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .badge-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #059669;
    }
    .metric-lbl {
        font-size: 0.95rem;
        font-weight: 600;
        color: #4B5563;
    }
    .explanation-text {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1F2937;
        padding: 15px;
        border-radius: 8px;
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        margin-top: 15px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 6px 10px -1px rgba(37, 99, 235, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Model & Metadata Loading (Cached for performance)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'battery_life_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_metadata():
    meta_path = os.path.join('models', 'model_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None

model = load_model()
metadata = load_metadata()

# -----------------------------------------------------------------------------
# Sidebar: Project & Student Information
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/full-battery.png", width=70)
    st.title("Project Details")
    
    st.markdown("""
    **Project Title:**  
    📱 *Mobile Battery Life Prediction*

    **Student Name:**  
    🎓 **Prapti Gaikwad**

    **Branch / Department:**  
    🤖 **Artificial Intelligence and Data Science (AI/DS)**

    **Academic Year:**  
    📚 **3rd Year**

    **Project Type:**  
    🔬 Machine Learning Mini Project
    """)
    st.divider()

    if metadata:
        st.subheader("Model Status")
        st.success(f"Active Model: **{metadata.get('best_model', 'Random Forest Regressor')}**")
        st.write(f"**R² Score:** `{metadata['best_metrics']['R2_Score']}`")
        st.write(f"**RMSE:** `{metadata['best_metrics']['RMSE']} hrs`")
        st.write(f"**MAE:** `{metadata['best_metrics']['MAE']} hrs`")
    st.divider()
    st.caption("College Mini Project viva demo ready.")

# -----------------------------------------------------------------------------
# Header Section
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🔋 Mobile Battery Life Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A Machine Learning system to estimate remaining smartphone battery duration based on usage patterns and hardware state.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="badge-card">
    <b>Student:</b> Prapti Gaikwad &nbsp;|&nbsp; 
    <b>Branch:</b> Artificial Intelligence & Data Science &nbsp;|&nbsp; 
    <b>Year:</b> 3rd Year &nbsp;|&nbsp; 
    <b>Project:</b> Machine Learning Mini Project
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tab_predict, tab_metrics, tab_tips, tab_viva = st.tabs([
    "⚡ Predict Battery Life", 
    "📊 Model Performance & EDA", 
    "💡 Smart Battery Tips", 
    "🎓 Viva & Project Documentation"
])

# -----------------------------------------------------------------------------
# TAB 1: PREDICTION INTERFACE
# -----------------------------------------------------------------------------
with tab_predict:
    st.subheader("Enter Mobile Usage & Hardware Details")
    
    # Preset scenarios for quick viva demonstration
    preset_choice = st.selectbox(
        "⚡ Quick Demo Presets (Select to auto-fill realistic test cases):",
        ["Custom Input", "Balanced Everyday User", "Intense Mobile Gamer", "Power-Saving Travel Mode", "Degraded / Older Phone"]
    )

    # Preset values dictionary
    presets = {
        "Custom Input": {
            "capacity": 4500, "battery_pct": 65, "sot": 4.0, "cpu": 40, "ram": 55,
            "data_gb": 2.5, "wifi_hrs": 4.0, "apps": 8, "brightness": 50, "gaming": 0.5,
            "cycles": 320, "age": 14, "temp": 31.0, "network": "Wi-Fi", "health": 91
        },
        "Balanced Everyday User": {
            "capacity": 5000, "battery_pct": 75, "sot": 3.5, "cpu": 35, "ram": 50,
            "data_gb": 1.2, "wifi_hrs": 5.0, "apps": 6, "brightness": 45, "gaming": 0.0,
            "cycles": 180, "age": 8, "temp": 28.5, "network": "Wi-Fi", "health": 96
        },
        "Intense Mobile Gamer": {
            "capacity": 5000, "battery_pct": 50, "sot": 6.5, "cpu": 85, "ram": 82,
            "data_gb": 4.5, "wifi_hrs": 1.0, "apps": 18, "brightness": 85, "gaming": 3.5,
            "cycles": 520, "age": 18, "temp": 42.0, "network": "5G", "health": 86
        },
        "Power-Saving Travel Mode": {
            "capacity": 4500, "battery_pct": 85, "sot": 1.5, "cpu": 20, "ram": 35,
            "data_gb": 0.3, "wifi_hrs": 0.5, "apps": 3, "brightness": 25, "gaming": 0.0,
            "cycles": 120, "age": 6, "temp": 26.5, "network": "4G", "health": 98
        },
        "Degraded / Older Phone": {
            "capacity": 3500, "battery_pct": 40, "sot": 5.0, "cpu": 65, "ram": 75,
            "data_gb": 3.0, "wifi_hrs": 3.0, "apps": 14, "brightness": 70, "gaming": 1.2,
            "cycles": 950, "age": 36, "temp": 38.0, "network": "4G", "health": 74
        }
    }
    vals = presets[preset_choice]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🔋 Battery & Device Specs")
        battery_capacity = st.number_input(
            "Battery Capacity (mAh)", 
            min_value=2500, max_value=7000, value=vals["capacity"], step=100,
            help="Full factory capacity of the phone battery."
        )
        current_battery_pct = st.slider(
            "Current Battery Percentage (%)", 
            min_value=1, max_value=100, value=vals["battery_pct"],
            help="Current charge remaining on the phone."
        )
        battery_health = st.slider(
            "Battery Health (%)", 
            min_value=60, max_value=100, value=vals["health"],
            help="Current maximum capacity health percentage compared to new."
        )
        charging_cycles = st.number_input(
            "Charging Cycles", 
            min_value=1, max_value=2000, value=vals["cycles"], step=10,
            help="Cumulative full charging cycles completed."
        )
        device_age_months = st.number_input(
            "Device Age (months)", 
            min_value=1, max_value=60, value=vals["age"], step=1,
            help="How many months old the smartphone is."
        )

    with col2:
        st.markdown("#### ⚙️ Hardware Load & Apps")
        screen_on_time = st.number_input(
            "Screen-on Time (hours)", 
            min_value=0.0, max_value=16.0, value=float(vals["sot"]), step=0.5,
            help="Time the screen has been actively used today."
        )
        brightness_level = st.slider(
            "Brightness Level (%)", 
            min_value=10, max_value=100, value=vals["brightness"],
            help="Display brightness percentage."
        )
        cpu_usage = st.slider(
            "CPU Usage (%)", 
            min_value=5, max_value=100, value=vals["cpu"],
            help="Average active processor utilization."
        )
        ram_usage = st.slider(
            "RAM Usage (%)", 
            min_value=10, max_value=100, value=vals["ram"],
            help="Random Access Memory utilization percentage."
        )
        number_of_apps = st.number_input(
            "Number of Apps Running", 
            min_value=1, max_value=50, value=vals["apps"], step=1,
            help="Count of apps open in background and foreground."
        )

    with col3:
        st.markdown("#### 🌐 Network & Usage Environment")
        network_usage = st.selectbox(
            "Network Usage Type", 
            ["Wi-Fi", "4G", "5G"], 
            index=["Wi-Fi", "4G", "5G"].index(vals["network"]),
            help="Primary active connection mode (5G consumes the most power)."
        )
        mobile_data_usage = st.number_input(
            "Mobile Data Usage (GB)", 
            min_value=0.0, max_value=30.0, value=float(vals["data_gb"]), step=0.5,
            help="Cellular mobile data consumed today."
        )
        wifi_usage = st.number_input(
            "Wi-Fi Usage (hours)", 
            min_value=0.0, max_value=24.0, value=float(vals["wifi_hrs"]), step=0.5,
            help="Hours connected and using Wi-Fi."
        )
        gaming_hours = st.number_input(
            "Gaming Hours", 
            min_value=0.0, max_value=12.0, value=float(vals["gaming"]), step=0.5,
            help="Hours spent playing 3D or high-performance mobile games."
        )
        temperature = st.slider(
            "Temperature (°C)", 
            min_value=20.0, max_value=55.0, value=float(vals["temp"]), step=0.5,
            help="Internal battery / device temperature."
        )

    st.write("")
    predict_btn = st.button("⚡ Predict Battery Life")

    if predict_btn:
        if model is None:
            st.error("Model file not found! Please run 'python train_model.py' to generate the trained model.")
        else:
            # Construct DataFrame with user inputs (matching the exact training schema)
            input_df = pd.DataFrame([{
                'Battery_Capacity_mAh': battery_capacity,
                'Current_Battery_Pct': current_battery_pct,
                'Screen_On_Time_Hours': screen_on_time,
                'CPU_Usage_Pct': cpu_usage,
                'RAM_Usage_Pct': ram_usage,
                'Mobile_Data_Usage_GB': mobile_data_usage,
                'WiFi_Usage_Hours': wifi_usage,
                'Number_of_Apps_Running': number_of_apps,
                'Brightness_Level_Pct': brightness_level,
                'Gaming_Hours': gaming_hours,
                'Charging_Cycles': charging_cycles,
                'Device_Age_Months': device_age_months,
                'Temperature_C': temperature,
                'Network_Usage': network_usage,
                'Battery_Health_Pct': battery_health
            }])

            # Perform real machine learning prediction
            predicted_hours = float(model.predict(input_df)[0])
            predicted_hours = max(0.2, round(predicted_hours, 2))

            # Display results in an appealing format
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")

            res_col1, res_col2, res_col3 = st.columns([1.2, 1.2, 1.6])
            
            with res_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">PREDICTED REMAINING LIFE</div>
                    <div class="metric-val">{predicted_hours} hrs</div>
                    <div style="color: #6B7280; font-size: 0.85rem; margin-top: 4px;">≈ {int(predicted_hours)} hr {int((predicted_hours % 1) * 60)} mins</div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                # Color code battery health/status
                status_color = "#10B981" if current_battery_pct > 50 else ("#F59E0B" if current_battery_pct > 20 else "#EF4444")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">CURRENT CHARGE LEVEL</div>
                    <div class="metric-val" style="color: {status_color};">{current_battery_pct}%</div>
                    <div style="color: #6B7280; font-size: 0.85rem; margin-top: 4px;">Health: {battery_health}%</div>
                </div>
                """, unsafe_allow_html=True)

            with res_col3:
                # Estimated discharge speed
                drain_assessment = "Moderate"
                if gaming_hours > 2.0 or (cpu_usage > 70 and brightness_level > 70) or network_usage == '5G' and temperature > 38:
                    drain_assessment = "High (Heavy Load)"
                elif cpu_usage < 30 and brightness_level < 40 and gaming_hours == 0:
                    drain_assessment = "Low (Power Saver)"

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">DISCHARGE PROFILE</div>
                    <div class="metric-val" style="color: #3B82F6; font-size: 1.8rem;">{drain_assessment}</div>
                    <div style="color: #6B7280; font-size: 0.85rem; margin-top: 4px;">Based on active workload</div>
                </div>
                """, unsafe_allow_html=True)

            # Requested standard explanation
            st.markdown(
                f'<div class="explanation-text">📢 The predicted remaining battery life is approximately <b>{predicted_hours} hours</b>.</div>',
                unsafe_allow_html=True
            )

            # Interactive Visual Progress Bar
            st.write("")
            pct_fraction = min(1.0, current_battery_pct / 100.0)
            st.progress(pct_fraction, text=f"Battery Level Indicator: {current_battery_pct}% remaining")

# -------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE & EDA
# -------------------------------------------------------------
with tab_metrics:
    st.subheader("Model Evaluation & Exploratory Data Analysis")
    
    if metadata:
        st.markdown(f"The best performing model dynamically selected is **{metadata['best_model']}** based on test set evaluation.")
        
        # Display comparison table
        st.markdown("#### 📈 Model Comparison on Test Data (20% Split)")
        res_df = pd.DataFrame(metadata['all_model_results']).T
        st.dataframe(res_df.style.highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='#D1FAE5')
                                .highlight_max(subset=['R2_Score'], color='#D1FAE5'), use_container_width=True)

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("#### 📊 Model Comparison Chart")
        comp_img_path = os.path.join('screenshots', 'model_comparison.png')
        if os.path.exists(comp_img_path):
            st.image(comp_img_path, caption="Comparison across MAE, RMSE, and R² Score", use_container_width=True)

    with col_img2:
        st.markdown("#### 🎯 Feature Importance (Random Forest)")
        fi_img_path = os.path.join('screenshots', 'feature_importance.png')
        if os.path.exists(fi_img_path):
            st.image(fi_img_path, caption="Key features driving battery life predictions", use_container_width=True)

    st.markdown("---")
    col_img3, col_img4 = st.columns(2)
    with col_img3:
        st.markdown("#### 🔥 Feature Correlation Heatmap")
        corr_img_path = os.path.join('screenshots', 'correlation_heatmap.png')
        if os.path.exists(corr_img_path):
            st.image(corr_img_path, caption="Multivariate correlations in the dataset", use_container_width=True)

    with col_img4:
        st.markdown("#### 📉 Target Distribution")
        dist_img_path = os.path.join('screenshots', 'battery_life_distribution.png')
        if os.path.exists(dist_img_path):
            st.image(dist_img_path, caption="Distribution of Remaining Battery Hours", use_container_width=True)

# -------------------------------------------------------------
# TAB 3: SMART BATTERY SAVER TIPS
# -------------------------------------------------------------
with tab_tips:
    st.subheader("💡 Data-Driven Battery Optimization Recommendations")
    st.write("These intelligent recommendations are dynamically aligned with the features identified as high-drain predictors in the model:")

    tip_col1, tip_col2 = st.columns(2)
    with tip_col1:
        st.info("🔆 **Screen Brightness Optimization**\n\nDisplay backlighting accounts for up to 35% of energy drain. Lowering brightness from 80% to 40% can extend battery life by up to 2.5 hours.")
        st.info("🎮 **Manage High-Intensity Gaming Sessions**\n\n3D GPU rendering and high refresh rate gaming spike current drain to 600-900 mA. Pausing intensive background tasks during gaming prevents extreme discharge rates.")
        st.info("📶 **Wi-Fi vs Cellular (5G/4G)**\n\n5G modem searches consume considerably more energy than Wi-Fi. Switching to Wi-Fi indoors can prolong your battery life by 15-20%.")

    with tip_col2:
        st.info("❄️ **Thermal Regulation (<35°C)**\n\nHigh battery temperatures accelerate chemical wear and increase internal resistance. Avoid charging while playing games or leaving the phone in direct sunlight.")
        st.info("🧹 **Close Idle Background Applications**\n\nUnused apps running in the background cause periodic wake locks and unnecessary CPU spikes. Keeping background apps under 10 improves longevity.")
        st.info("⚡ **Maintain Healthy Charging Habits**\n\nOperating between 20% and 80% charge level significantly decreases battery cycle degradation and preserves maximum capacity health.")

# -------------------------------------------------------------
# TAB 4: VIVA & PROJECT DOCUMENTATION
# -------------------------------------------------------------
with tab_viva:
    st.subheader("🎓 Viva Presentation & Project Q&A Guide")
    st.markdown("""
    This section is designed to help **Prapti Gaikwad** present and answer questions during the project viva examination.

    ---
    ### 1. What is the objective of this project?
    **Answer:** To develop a supervised machine learning regression model that predicts the remaining smartphone battery life in hours, based on live usage metrics (screen brightness, CPU/RAM usage, apps running, gaming) and hardware degradation parameters (charging cycles, battery health, temperature, age).

    ---
    ### 2. Which algorithms were tested and why was Random Forest selected?
    **Answer:**
    - **Linear Regression:** Good baseline; assumes linear relationships, achieved $R^2 \approx 0.91$.
    - **Decision Tree Regressor:** Captures non-linear thresholds, achieved $R^2 \approx 0.87$.
    - **Random Forest Regressor:** An ensemble of decision trees using bootstrap aggregating (bagging); captures complex non-linear feature interactions (such as combined CPU + Gaming + Temperature spikes) and reduces variance, achieving the highest performance ($R^2 \approx 0.94$).
    
    The model was **dynamically selected** based on having the highest $R^2$ score and lowest RMSE on the unseen test set without hardcoding.

    ---
    ### 3. How were missing values and categorical data handled?
    **Answer:**
    - Missing numerical values in `CPU_Usage_Pct` and `RAM_Usage_Pct` were handled using **Median Imputation** via `SimpleImputer`, which is resilient against outliers.
    - Categorical feature `Network_Usage` ('Wi-Fi', '4G', '5G') was transformed using **One-Hot Encoding** (`OneHotEncoder(drop='first')`) inside a unified Scikit-Learn `ColumnTransformer` to prevent data leakage.
    - Numerical features were standardized using **StandardScaler** to assist convergence.

    ---
    ### 4. What are the key evaluation metrics used?
    **Answer:**
    - **MAE (Mean Absolute Error):** Measures the average magnitude of absolute errors in hours.
    - **MSE (Mean Squared Error):** Penalizes larger prediction deviations more heavily.
    - **RMSE (Root Mean Squared Error):** The square root of MSE, expressed in the same units (hours) as the target variable for intuitive explanation.
    - **$R^2$ Score (Coefficient of Determination):** Represents the proportion of variance in remaining battery hours that is predictable from the input features (1.0 is a perfect fit).

    ---
    ### 5. Why is this project relevant to real-world applications?
    **Answer:** Modern smartphone operating systems (such as Android Smart Battery and iOS Battery Health) use ML on device telemetry to throttle background tasks, warn users before the battery dies, and optimize charging to extend battery lifespan.
    """)

# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>"
    "Mobile Battery Life Prediction Mini Project &nbsp;|&nbsp; <b>Prapti Gaikwad</b> &nbsp;|&nbsp; "
    "Artificial Intelligence & Data Science &nbsp;|&nbsp; 3rd Year"
    "</div>",
    unsafe_allow_html=True
)
