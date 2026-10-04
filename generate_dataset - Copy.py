"""
Dataset Generator for Mobile Battery Life Prediction
Author: Prapti Gaikwad (Artificial Intelligence & Data Science, 3rd Year)
Description:
    Generates a realistic dataset of 3,000 smartphone usage records with physical and
    empirical electrical discharge dynamics. Features include battery capacity, current percentage,
    screen-on time, hardware load, network consumption, and battery health.
"""

import numpy as np
import pandas as pd
import os

def generate_mobile_battery_dataset(n_samples=3000, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Device Hardware Specs
    # Typical smartphone capacities: 3000mAh to 6000mAh
    battery_capacity_choices = [3000, 3500, 4000, 4500, 5000, 5500, 6000]
    battery_capacity = np.random.choice(battery_capacity_choices, size=n_samples, p=[0.05, 0.10, 0.25, 0.30, 0.20, 0.07, 0.03])
    # Add slight manufacturing variance (+- 50 mAh)
    battery_capacity = battery_capacity + np.random.randint(-50, 50, size=n_samples)

    # Device Age in months (1 to 48 months)
    device_age_months = np.random.randint(1, 49, size=n_samples)

    # Charging Cycles: roughly 20-30 cycles per month of device age, with random variation
    charging_cycles = np.clip(
        device_age_months * np.random.normal(22, 5, size=n_samples),
        10, 1500
    ).astype(int)

    # Battery Health (%): degrades with charging cycles and age (typical 70% to 100%)
    health_degradation = (charging_cycles * 0.02) + (device_age_months * 0.12) + np.random.normal(0, 1.5, size=n_samples)
    battery_health_pct = np.clip(100.0 - health_degradation, 70.0, 100.0).round(1)

    # 2. Current State
    # Current battery percentage at time of observation (5% to 100%)
    current_battery_pct = np.random.uniform(5.0, 100.0, size=n_samples).round(1)

    # 3. Usage & System Workload
    # Screen-on time recorded today (hours): 0.5 to 10.0 hours
    screen_on_time_hours = np.clip(np.random.normal(4.5, 2.0, size=n_samples), 0.5, 11.5).round(2)

    # CPU Usage (%): 10% to 95%
    cpu_usage_pct = np.clip(np.random.normal(42.0, 18.0, size=n_samples), 10.0, 95.0).round(1)

    # RAM Usage (%): 25% to 95%
    ram_usage_pct = np.clip(np.random.normal(55.0, 16.0, size=n_samples), 20.0, 95.0).round(1)

    # Number of Apps Running: 2 to 32
    number_of_apps = np.random.poisson(lam=9, size=n_samples)
    number_of_apps = np.clip(number_of_apps, 2, 35)

    # Screen Brightness (%): 10% to 100%
    brightness_level_pct = np.clip(np.random.normal(55.0, 22.0, size=n_samples), 10.0, 100.0).round(1)

    # Gaming Hours: Zero-inflated (approx 45% non-gamers, 55% gamers between 0.5 to 5.5 hours)
    has_gaming = np.random.binomial(1, 0.55, size=n_samples)
    gaming_hours = has_gaming * np.clip(np.random.exponential(1.5, size=n_samples), 0.2, 6.0)
    gaming_hours = gaming_hours.round(2)

    # 4. Network Features
    network_usage_types = ['Wi-Fi', '4G', '5G']
    network_usage = np.random.choice(network_usage_types, size=n_samples, p=[0.45, 0.35, 0.20])

    # Mobile Data Usage (GB) & Wi-Fi Usage (hours)
    mobile_data_usage_gb = np.where(
        network_usage != 'Wi-Fi',
        np.clip(np.random.exponential(2.0, size=n_samples), 0.1, 12.0),
        np.clip(np.random.exponential(0.3, size=n_samples), 0.0, 2.0)
    ).round(2)

    wifi_usage_hours = np.where(
        network_usage == 'Wi-Fi',
        np.clip(np.random.normal(5.0, 2.2, size=n_samples), 0.5, 12.0),
        np.clip(np.random.exponential(0.8, size=n_samples), 0.0, 4.0)
    ).round(2)

    # 5. Device Temperature (°C)
    # Rises with CPU load, gaming, and 5G network usage
    base_temp = 26.0
    temp_rise_cpu = cpu_usage_pct * 0.11
    temp_rise_game = gaming_hours * 1.8
    temp_rise_network = np.where(network_usage == '5G', 2.0, np.where(network_usage == '4G', 1.0, 0.2))
    temperature_c = np.clip(
        base_temp + temp_rise_cpu + temp_rise_game + temp_rise_network + np.random.normal(0, 1.2, size=n_samples),
        25.0, 48.5
    ).round(1)

    # 6. Realistic Physical Target Variable: Battery Life Remaining (hours)
    # Remaining Effective Energy in mAh:
    effective_capacity_mah = battery_capacity * (battery_health_pct / 100.0)
    remaining_energy_mah = effective_capacity_mah * (current_battery_pct / 100.0)

    # Current Discharge Drain Rate in mA:
    base_drain_ma = 85.0
    screen_drain_ma = brightness_level_pct * 2.5
    cpu_drain_ma = cpu_usage_pct * 2.2
    ram_drain_ma = (ram_usage_pct * 0.5) + (number_of_apps * 3.2)
    gaming_drain_ma = gaming_hours * 52.0
    network_drain_ma = np.where(
        network_usage == '5G',
        (mobile_data_usage_gb * 18.0) + 90.0,
        np.where(
            network_usage == '4G',
            (mobile_data_usage_gb * 12.0) + 50.0,
            (wifi_usage_hours * 7.0) + 20.0
        )
    )
    temp_penalty_ma = np.maximum(0, temperature_c - 36.0) * 11.0
    cycle_aging_penalty_ma = (charging_cycles / 500.0) * 15.0

    total_drain_rate_ma = (
        base_drain_ma
        + screen_drain_ma
        + cpu_drain_ma
        + ram_drain_ma
        + gaming_drain_ma
        + network_drain_ma
        + temp_penalty_ma
        + cycle_aging_penalty_ma
    )

    # Theoretical remaining hours = Remaining Energy (mAh) / Average Drain (mA)
    calculated_hours = remaining_energy_mah / total_drain_rate_ma

    # Add realistic environmental variance/noise
    noise = np.random.normal(0, 0.30, size=n_samples)
    battery_life_remaining_hours = np.clip(calculated_hours + noise, 0.4, 30.0).round(2)

    df = pd.DataFrame({
        'Battery_Capacity_mAh': battery_capacity,
        'Current_Battery_Pct': current_battery_pct,
        'Screen_On_Time_Hours': screen_on_time_hours,
        'CPU_Usage_Pct': cpu_usage_pct,
        'RAM_Usage_Pct': ram_usage_pct,
        'Mobile_Data_Usage_GB': mobile_data_usage_gb,
        'WiFi_Usage_Hours': wifi_usage_hours,
        'Number_of_Apps_Running': number_of_apps,
        'Brightness_Level_Pct': brightness_level_pct,
        'Gaming_Hours': gaming_hours,
        'Charging_Cycles': charging_cycles,
        'Device_Age_Months': device_age_months,
        'Temperature_C': temperature_c,
        'Network_Usage': network_usage,
        'Battery_Health_Pct': battery_health_pct,
        'Battery_Life_Remaining_Hours': battery_life_remaining_hours
    })

    # Introduce a small number of realistic missing values (~0.8% in 2 columns)
    # This demonstrates the student's data cleaning and imputation methodology during viva
    missing_mask_cpu = np.random.rand(n_samples) < 0.008
    missing_mask_ram = np.random.rand(n_samples) < 0.008
    df.loc[missing_mask_cpu, 'CPU_Usage_Pct'] = np.nan
    df.loc[missing_mask_ram, 'RAM_Usage_Pct'] = np.nan

    return df

if __name__ == '__main__':
    output_dir = 'dataset'
    os.makedirs(output_dir, exist_ok=True)
    df = generate_mobile_battery_dataset(n_samples=3000, random_seed=42)
    output_path = os.path.join(output_dir, 'mobile_battery_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at: {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Features: {list(df.columns)}")
    print("\nMissing values introduced for data cleaning demonstration:")
    print(df.isnull().sum())
    print("\nSample summary statistics:")
    print(df.describe().round(2))
