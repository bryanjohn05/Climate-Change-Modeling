import subprocess
import sys
import os

def install_requirements():
    """Install required Python packages"""
    requirements = [
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.1.0'
    ]
    
    for requirement in requirements:
        try:
            print(f"Installing {requirement}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', requirement])
            print(f"✓ {requirement} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {requirement}: {e}")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['../public']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")

if __name__ == "__main__":
    print("Setting up Climate Change Modeling Project...")
    print("=" * 50)
    
    # Install requirements
    print("\n1. Installing Python dependencies...")
    if install_requirements():
        print("✓ All dependencies installed successfully")
    else:
        print("✗ Some dependencies failed to install")
        sys.exit(1)
    
    # Create directories
    print("\n2. Creating project directories...")
    create_directories()
    
    print("\n3. Initializing model...")
    try:
        from climate_model import load_model_and_predict
        load_model_and_predict()
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✓ Setup complete! You can now run the dashboard.")
    print("Run 'npm run dev' to start the Next.js development server.")
