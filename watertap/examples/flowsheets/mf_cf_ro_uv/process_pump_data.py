
import pandas as pd

# Example: Load your data from a CSV file
# Replace 'your_file.csv' with your actual file path
# Ensure the file has 4 columns of corresponding data
mycsv = r"C:\Users\Adam\Box\WaterTAP (protected by NDA)\nawi 1.0 (NDA protected)\WaterTAP-5.09 MF-RO-UV\data\UCI_Pilot\pump_surrogate_models\pump_data\multispeed_ro_pump_head_power_concatenated_global_normalized.csv"

df = pd.read_csv(mycsv)

# Display original data
print("Original Data:")
print(df)

# Interpolate missing values (NaN) using linear interpolation
# This will handle gaps in all columns while preserving alignment
df_interpolated = df.interpolate(method='linear', limit_direction='forward', axis=0)

# Display interpolated data
print("\nInterpolated Data:")
print(df_interpolated)

# Optionally, save the result to a new CSV file
