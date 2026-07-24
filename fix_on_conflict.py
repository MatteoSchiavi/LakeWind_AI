#!/usr/bin/env python3
"""Apply ON CONFLICT clauses to access.py INSERT statements."""
with open('lakewind/db/access.py') as f:
    content = f.read()

# forecast_runs VALUES
old_fc = 'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
new_fc = '''VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (model_name, point_id, run_time, valid_time) DO UPDATE SET
                wind_speed_kn = EXCLUDED.wind_speed_kn,
                wind_dir_deg = EXCLUDED.wind_dir_deg,
                wind_gust_kn = EXCLUDED.wind_gust_kn,
                pressure_msl = EXCLUDED.pressure_msl,
                temperature_2m = EXCLUDED.temperature_2m,
                dew_point_2m = EXCLUDED.dew_point_2m,
                cloud_cover = EXCLUDED.cloud_cover,
                shortwave_radiation = EXCLUDED.shortwave_radiation,
                cape = EXCLUDED.cape,
                boundary_layer_height = EXCLUDED.boundary_layer_height,
                precipitation = EXCLUDED.precipitation,
                weather_code = EXCLUDED.weather_code,
                visibility = EXCLUDED.visibility,
                raw_json = EXCLUDED.raw_json'''

n = content.count(old_fc)
content = content.replace(old_fc, new_fc)
print(f"forecast_runs: {n} replacements")

# observations VALUES
old_obs = 'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
new_obs = '''VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source, timestamp, lat, lon) DO UPDATE SET
                wind_speed_kn = EXCLUDED.wind_speed_kn,
                wind_dir_deg = EXCLUDED.wind_dir_deg,
                wind_gust_kn = EXCLUDED.wind_gust_kn,
                pressure = EXCLUDED.pressure,
                temperature = EXCLUDED.temperature,
                humidity = EXCLUDED.humidity,
                quality_flag = EXCLUDED.quality_flag,
                confidence = EXCLUDED.confidence'''
n = content.count(old_obs)
content = content.replace(old_obs, new_obs)
print(f"observations: {n} replacements")

# model_registry VALUES
old_mr = 'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
new_mr = '''VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (model_version) DO UPDATE SET
                trained_at = EXCLUDED.trained_at,
                feature_set_version = EXCLUDED.feature_set_version,
                training_period_start = EXCLUDED.training_period_start,
                training_period_end = EXCLUDED.training_period_end,
                backtest_mae_kn = EXCLUDED.backtest_mae_kn,
                backtest_dir_error_deg = EXCLUDED.backtest_dir_error_deg,
                promoted_to_production = EXCLUDED.promoted_to_production,
                git_commit = EXCLUDED.git_commit,
                notes = EXCLUDED.notes'''
n = content.count(old_mr)
content = content.replace(old_mr, new_mr)
print(f"model_registry: {n} replacements")

with open('lakewind/db/access.py', 'w') as f:
    f.write(content)
print("Done")
