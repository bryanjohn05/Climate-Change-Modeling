import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
  try:
      from climate_model import load_model_and_predict
      
      print("Initializing climate model and generating predictions...")
      
      # Look for CSV files in the current directory
      csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

      # Prioritize the new dataset name
      target_csv_file = 'climate_change_dataset.csv'
      if target_csv_file in csv_files:
          csv_file = target_csv_file
          print(f"Found target CSV file: {csv_file}")
      elif csv_files:
          csv_file = csv_files[0]  # Use the first CSV file found
          print(f"Using: {csv_file}")
      else:
          print("No CSV files found. Creating sample data...")
          csv_file = target_csv_file

          # Create a sample CSV file for testing with new columns
          import pandas as pd
          import numpy as np
          from datetime import datetime, timedelta

          # Generate sample data for the new columns
          start_year = 1990
          years = list(range(start_year, 2024))
          num_records = len(years)

          sample_data = {
              'Year': years,
              'Country': ['Global'] * num_records,
              'Avg Temperature (°C)': np.random.normal(15, 2, num_records),
              'CO2 Emissions (Tons/Capita)': np.random.normal(4, 1, num_records),
              'Sea Level Rise (mm)': np.cumsum(np.random.normal(2, 0.5, num_records)),
              'Rainfall (mm)': np.random.normal(1000, 100, num_records),
              'Population': np.linspace(5_000_000_000, 8_000_000_000, num_records),
              'Renewable Energy (%)': np.random.normal(15, 5, num_records),
              'Extreme Weather Events': np.random.randint(1, 10, num_records),
              'Forest Area (%)': np.random.normal(30, 2, num_records)
          }

          df = pd.DataFrame(sample_data)
          df.to_csv(csv_file, index=False)
          print(f"Created sample data file: {csv_file}")
      
      # Run the model
      load_model_and_predict(csv_file)
      print("✓ Climate dashboard data ready!")
      
  except Exception as e:
      print(f"✗ Error: {str(e)}")
      import traceback
      traceback.print_exc()
      return False
  
  return True

if __name__ == "__main__":
  success = main()
  if not success:
      input("Press Enter to exit...")
