import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def force_retrain():
  """Force retrain the model by deleting the old one"""
  
  # Delete the old model file
  public_dir = os.path.join('..', 'public')
  model_path = os.path.join(public_dir, 'climate_model.pkl')
  
  if os.path.exists(model_path):
      os.remove(model_path)
      print("✓ Old model deleted")
  
  # Now retrain with real data
  from climate_model import load_model_and_predict
  
  # Look for CSV files
  csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
  
  # Prioritize the new dataset name
  target_csv_file = 'climate_change_dataset.csv'
  if target_csv_file in csv_files:
      csv_file = target_csv_file
      print(f"Using target CSV file: {csv_file}")
  elif csv_files:
      csv_file = csv_files[0]
      print(f"Using CSV file: {csv_file}")
  else:
      print("No CSV files found. Please add your climate dataset CSV file to the scripts directory.")
      return

  load_model_and_predict(csv_file)
  print("✓ Model retrained successfully!")

if __name__ == "__main__":
  force_retrain()
