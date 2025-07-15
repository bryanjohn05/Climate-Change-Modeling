# Climate Change Dashboard

This is an interactive climate change dashboard that visualizes historical atmospheric data and provides future temperature predictions using a machine learning model. The dashboard is built with Next.js (React) for the frontend and Python for the machine learning backend.

## Features

*   **Historical Data Visualization:** View trends for various atmospheric features like CO2 emissions, sea level rise, rainfall, population, renewable energy, extreme weather events, and forest area.
*   **Temperature Prediction:** Get future temperature projections based on a trained machine learning model.
*   **Scenario Analysis:** Explore optimistic, baseline, and pessimistic temperature scenarios.
*   **Model Details:** Access information about the trained model's performance and download the model and data.

## Setup and Installation

Follow these steps to set up and run the project locally.

### Prerequisites

*   Node.js (LTS version recommended)
*   Python 3.8+
*   npm or yarn

### 1. Clone the Repository

If you haven't already, clone this project to your local machine:

\`\`\`bash
git clone <your-repository-url>
cd <your-project-directory>
\`\`\`

### 2. Install Python Dependencies and Initialize Model

Navigate into the `scripts` directory and run the setup script. This will install the necessary Python packages and train the machine learning model, generating the `climate_model.pkl` and `climate-predictions.json` files in the `public` directory.

**On Windows:**

\`\`\`bash
cd scripts
run_model.bat
\`\`\`

**On macOS/Linux:**

\`\`\`bash
cd scripts
python setup.py
\`\`\`

If you have your own `climate_change_dataset.csv` file, place it in the `scripts` directory before running the setup script. Otherwise, a sample dataset will be generated.

### 3. Install Frontend Dependencies

Navigate back to the root of your project and install the Next.js dependencies:

\`\`\`bash
cd .. # If you are still in the scripts directory
npm install
# or
yarn install
\`\`\`

### 4. Run the Development Server

Start the Next.js development server:

\`\`\`bash
npm run dev
# or
yarn dev
\`\`\`

The dashboard should now be accessible at `http://localhost:3000`.

## Project Structure

*   `app/`: Next.js App Router pages and API routes.
    *   `app/page.tsx`: The main dashboard page.
    *   `app/api/update-model/route.ts`: API route to trigger model retraining.
*   `components/`: React components for the dashboard UI.
    *   `components/climate-dashboard.tsx`: The main dashboard component.
*   `scripts/`: Python scripts for data processing and machine learning.
    *   `scripts/climate_model.py`: Handles data loading, model training, and prediction generation.
    *   `scripts/initialize_data.py`: Initializes the model and generates sample data if no dataset is found.
    *   `scripts/retrain_model.py`: Script to force model retraining.
    *   `scripts/setup.py`: Installs Python requirements and runs initial model setup.
    *   `scripts/run_model.bat`: Windows batch script to run the setup.
*   `public/`: Static assets, including the generated `climate_model.pkl` and `climate-predictions.json` files.
*   `requirements.txt`: Python package dependencies.



---
