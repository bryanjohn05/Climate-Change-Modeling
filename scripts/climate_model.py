import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os
from datetime import datetime

def load_and_prepare_data(csv_path):
  """Load and prepare the real climate dataset"""

  print(f"Loading dataset from {csv_path}...")

  # Load the CSV file
  df = pd.read_csv(csv_path)

  print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
  print(f"Columns: {list(df.columns)}")

  # Check if required columns exist
  required_columns = ['Year', 'Avg Temperature (°C)']
  missing_columns = [col for col in required_columns if col not in df.columns]

  if missing_columns:
      print(f"Missing required columns: {missing_columns}")
      print("Available columns:", list(df.columns))
      raise ValueError(f"Required columns {missing_columns} not found in dataset")

  # Handle missing values in target variable
  print(f"Rows before removing NaN temperatures: {len(df)}")
  df = df.dropna(subset=['Avg Temperature (°C)'])
  print(f"Rows after removing NaN temperatures: {len(df)}")

  # Rename target column for consistency if needed internally, or just use the new name
  df = df.rename(columns={'Avg Temperature (°C)': 'AverageTemperature'})

  # Handle new atmospheric features and fill NaNs with median
  new_features = [
      'CO2 Emissions (Tons/Capita)', 'Sea Level Rise (mm)', 'Rainfall (mm)',
      'Population', 'Renewable Energy (%)', 'Extreme Weather Events', 'Forest Area (%)'
  ]
  for feature in new_features:
      if feature in df.columns:
          df[feature] = pd.to_numeric(df[feature], errors='coerce')
          df[feature] = df[feature].fillna(df[feature].median())
      else:
          print(f"Warning: Feature '{feature}' not found in dataset. Adding with default 0.")
          df[feature] = 0 # Add as 0 if not present

  # Handle 'Country' column if it exists
  if 'Country' in df.columns:
      le_country = LabelEncoder()
      df['Country_encoded'] = le_country.fit_transform(df['Country'].astype(str))
  else:
      le_country = None
      df['Country_encoded'] = 0

  print(f"Data prepared with {len(df)} rows after cleaning")
  print(f"Temperature range: {df['AverageTemperature'].min():.2f} to {df['AverageTemperature'].max():.2f}")

  return df, le_country

def train_and_save_model(csv_path='climate_change_dataset.csv'):
  """Train the model using real climate data and save it as pickle file"""

  try:
      # Load and prepare data
      df, le_country = load_and_prepare_data(csv_path)

      # Define features and target
      feature_columns = [
          'Year',
          'CO2 Emissions (Tons/Capita)',
          'Sea Level Rise (mm)',
          'Rainfall (mm)',
          'Population',
          'Renewable Energy (%)',
          'Extreme Weather Events',
          'Forest Area (%)',
          'Country_encoded' # Keep country if it exists
      ]

      # Filter feature_columns to only include those present in the DataFrame
      available_features = [col for col in feature_columns if col in df.columns]
      print(f"Using features: {available_features}")

      X = df[available_features]
      y = df['AverageTemperature']

      # Remove any remaining NaN values
      mask = ~(X.isna().any(axis=1) | y.isna())
      X = X[mask]
      y = y[mask]

      print(f"Training data shape: {X.shape}")
      print(f"Target data shape: {y.shape}")

      if len(X) == 0:
          raise ValueError("No valid training data after cleaning")

      # Split the data
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

      # Scale features
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train)
      X_test_scaled = scaler.transform(X_test)

      # Train Random Forest model
      model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
      model.fit(X_train_scaled, y_train)

      # Calculate metrics
      train_score = model.score(X_train_scaled, y_train)
      test_score = model.score(X_test_scaled, y_test)

      # Make predictions for additional metrics
      y_pred = model.predict(X_test_scaled)
      mae = mean_absolute_error(y_test, y_pred)
      mse = mean_squared_error(y_test, y_pred)
      rmse = np.sqrt(mse)

      # Create public directory if it doesn't exist
      public_dir = os.path.join('..', 'public')
      if not os.path.exists(public_dir):
          os.makedirs(public_dir)

      # --- NEW: Group by Year and calculate mean for sample_data ---
      numerical_cols_for_avg = [
          'AverageTemperature',
          'CO2 Emissions (Tons/Capita)',
          'Sea Level Rise (mm)',
          'Rainfall (mm)',
          'Population',
          'Renewable Energy (%)',
          'Extreme Weather Events',
          'Forest Area (%)'
      ]
      # Filter to only include columns actually present in df
      cols_to_average = [col for col in numerical_cols_for_avg if col in df.columns]

      # Create a temporary DataFrame for grouping to avoid modifying the original df used for training
      df_for_sample = df[['Year'] + cols_to_average].copy()

      # Group by 'Year' and calculate the mean for the relevant columns
      # Reset index to make 'Year' a regular column again
      yearly_avg_df = df_for_sample.groupby('Year')[cols_to_average].mean().reset_index()

      # Convert to dictionary records for JSON serialization, ensuring it's sorted by year
      sample_data_for_json = yearly_avg_df.sort_values(by='Year').to_dict('records')
      # --- END NEW ---

      # Store means/medians of new features for prediction
      feature_means = {col: float(df[col].mean()) for col in available_features if col != 'Year'}
      
      # Save model and related data
      model_data = {
          'model': model,
          'scaler': scaler,
          'feature_columns': available_features,
          'label_encoders': {'country': le_country},
          'train_score': train_score,
          'test_score': test_score,
          'mae': mae,
          'mse': mse,
          'rmse': rmse,
          'sample_data': sample_data_for_json, # Now contains yearly averages
          'data_stats': {
              'mean_temp': float(df['AverageTemperature'].mean()),
              'std_temp': float(df['AverageTemperature'].std()),
              'min_temp': float(df['AverageTemperature'].min()),
              'max_temp': float(df['AverageTemperature'].max()),
              **feature_means # Add means of new features
          },
          'dataset_source': csv_path,  # Track which dataset was used
          'model_version': '2.5'  # Increment version for this fix
      }

      model_path = os.path.join(public_dir, 'climate_model.pkl')
      with open(model_path, 'wb') as f:
          pickle.dump(model_data, f)

      print(f"Model trained and saved to {model_path}!")
      print(f"Train R² Score: {train_score:.4f}")
      print(f"Test R² Score: {test_score:.4f}")
      print(f"MAE: {mae:.4f}")
      print(f"RMSE: {rmse:.4f}")

      return model_data

  except Exception as e:
      print(f"Error in train_and_save_model: {str(e)}")
      raise

def check_model_compatibility(model_data, csv_path):
  """Check if the existing model is compatible with the current dataset"""

  # Check if model was trained on the same dataset
  if model_data.get('dataset_source') != csv_path:
      print(f"Model was trained on different dataset: {model_data.get('dataset_source')} vs {csv_path}")
      return False

  # Check if it's an older model version that needs retraining
  if model_data.get('model_version', '1.0') < '2.5': # Force retrain for versions before this fix
      print("Found old model version, need to retrain with current data processing.")
      return False

  return True

def load_model_and_predict(csv_path='climate_change_dataset.csv'):
  """Load the model and generate predictions for dashboard"""

  try:
      # Define paths relative to script location
      public_dir = os.path.join('..', 'public')
      model_path = os.path.join(public_dir, 'climate_model.pkl')

      model_data = None
      need_retrain = False

      try:
          with open(model_path, 'rb') as f:
              model_data = pickle.load(f)
          print("Model loaded successfully!")

          # Check compatibility
          if not check_model_compatibility(model_data, csv_path):
              need_retrain = True

      except FileNotFoundError:
          print("Model not found, training new model...")
          need_retrain = True

      # Retrain if needed
      if need_retrain:
          print("Retraining model with current dataset...")
          model_data = train_and_save_model(csv_path)

      model = model_data['model']
      scaler = model_data['scaler']
      feature_columns = model_data['feature_columns']
      data_stats = model_data.get('data_stats', {})

      print(f"Model features: {feature_columns}")

      # Generate future predictions (2024-2030)
      future_years = list(range(2024, 2031))
      predictions = []

      # Get average values for new features from data_stats
      # Provide sensible defaults if not found in data_stats (e.g., for sample data generation)
      avg_co2 = data_stats.get('CO2 Emissions (Tons/Capita)', 5.0)
      avg_sea_level = data_stats.get('Sea Level Rise (mm)', 0.0)
      avg_rainfall = data_stats.get('Rainfall (mm)', 1000.0)
      avg_population = data_stats.get('Population', 7_000_000_000)
      avg_renewable_energy = data_stats.get('Renewable Energy (%)', 20.0)
      avg_extreme_events = data_stats.get('Extreme Weather Events', 5)
      avg_forest_area = data_stats.get('Forest Area (%)', 30.0)
      avg_country_encoded = data_stats.get('Country_encoded', 0) # Default to 0 if not present

      baseline_temp = data_stats.get('mean_temp', 15.0) # Default to 15 if not found

      for year in future_years:
          future_data = {}

          # Only add features that exist in the trained model
          if 'Year' in feature_columns:
              future_data['Year'] = year
          if 'CO2 Emissions (Tons/Capita)' in feature_columns:
              future_data['CO2 Emissions (Tons/Capita)'] = avg_co2 * (1 + (year - 2024) * 0.01) # Slight increase
          if 'Sea Level Rise (mm)' in feature_columns:
              future_data['Sea Level Rise (mm)'] = avg_sea_level + (year - 2024) * 3 # Increase by 3mm/year
          if 'Rainfall (mm)' in feature_columns:
              future_data['Rainfall (mm)'] = avg_rainfall
          if 'Population' in feature_columns:
              future_data['Population'] = avg_population * (1 + (year - 2024) * 0.005) # Slight increase
          if 'Renewable Energy (%)' in feature_columns:
              future_data['Renewable Energy (%)'] = avg_renewable_energy * (1 + (year - 2024) * 0.002) # Slight increase
          if 'Extreme Weather Events' in feature_columns:
              future_data['Extreme Weather Events'] = avg_extreme_events + (year - 2024) * 0.5 # Slight increase
          if 'Forest Area (%)' in feature_columns:
              future_data['Forest Area (%)'] = avg_forest_area * (1 - (year - 2024) * 0.001) # Slight decrease
          if 'Country_encoded' in feature_columns:
              future_data['Country_encoded'] = avg_country_encoded

          # Convert to DataFrame and scale
          future_df = pd.DataFrame([future_data])

          # Ensure columns are in the same order as training
          future_df = future_df.reindex(columns=feature_columns, fill_value=0)

          future_scaled = scaler.transform(future_df)

          # Make prediction
          prediction = model.predict(future_scaled)[0]

          predictions.append({
              'year': year,
              'predicted_temperature': round(prediction, 2),
              'temperature_anomaly': round(prediction - baseline_temp, 2),
              'uncertainty': round(data_stats.get('std_temp', 1.0), 3) # Use std dev as proxy for uncertainty
          })

      # Generate monthly data for current year (synthetic for yearly dataset)
      monthly_data = []
      current_year_prediction = predictions[0]['predicted_temperature'] if predictions else baseline_temp
      month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

      for month_idx, month_name in enumerate(month_names):
          # For a yearly dataset, monthly data will be an approximation or average
          # Here, we'll just use the current year's predicted temperature for all months
          monthly_data.append({
              'month': month_name,
              'temperature': round(current_year_prediction + np.random.normal(0, 0.5), 2), # Add slight variation
              'month_num': month_idx + 1
          })

      # Save predictions to JSON for frontend
      dashboard_data = {
          'future_predictions': predictions,
          'monthly_data': monthly_data,
          'model_info': {
              'train_score': model_data['train_score'],
              'test_score': model_data['test_score'],
              'mae': model_data.get('mae', 0),
              'rmse': model_data.get('rmse', 0),
              'feature_count': len(feature_columns),
              'model_type': 'Random Forest Regressor',
              'data_source': 'Historical Climate Dataset (Updated)',
              'model_version': model_data.get('model_version', '1.0')
          },
          'sample_data': model_data['sample_data'], # Now contains yearly averages
          'dataset_info': {
              'total_records': len(model_data['sample_data']),
              'date_range': f"{min([d.get('Year', 1900) for d in model_data['sample_data']])} - {max([d.get('Year', 2023) for d in model_data['sample_data']])}",
              'features_used': feature_columns,
              'temperature_stats': data_stats
          }
      }

      # Create public directory if it doesn't exist
      if not os.path.exists(public_dir):
          os.makedirs(public_dir)

      json_path = os.path.join(public_dir, 'climate-predictions.json')
      with open(json_path, 'w') as f:
          json.dump(dashboard_data, f, indent=2)

      print(f"Predictions generated and saved to {json_path}")
      return dashboard_data

  except Exception as e:
      print(f"Error in load_model_and_predict: {str(e)}")
      print(f"Error type: {type(e).__name__}")
      import traceback
      traceback.print_exc()
      raise

if __name__ == "__main__":
  # You can specify your CSV file path here
  csv_file = input("Enter the path to your climate CSV file (or press Enter for 'climate_change_dataset.csv'): ").strip()
  if not csv_file:
      csv_file = 'climate_change_dataset.csv'

  if os.path.exists(csv_file):
      try:
          load_model_and_predict(csv_file)
      except Exception as e:
          print(f"Failed to process data: {str(e)}")
  else:
      print(f"CSV file '{csv_file}' not found. Please make sure the file exists in the scripts directory.")
      print("Available files in current directory:")
      for file in os.listdir('.'):
          if file.endswith('.csv'):
              print(f"  - {file}")
