# Medical Device Maintenance Predictor

This Streamlit application predicts which medical devices are likely to need maintenance within the next 30 days using machine learning. The application includes data simulation, preprocessing, model training, and interactive visualizations.

## Features

- Simulated medical device data generation
- Data preprocessing and feature engineering
- Random Forest Classifier for maintenance prediction
- Interactive visualizations using Plotly
- Filtering capabilities by facility and department
- Performance metrics display
- Feature importance analysis

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run medical_device_maintenance_app.py
   ```
2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Data Structure

The application works with three main datasets:

1. Device Master Data:
   - device_id
   - device_type
   - purchase_date
   - warranty_expiry
   - facility
   - department
   - status

2. Usage Logs:
   - device_id
   - usage_hours
   - usage_date

3. Maintenance Logs:
   - device_id
   - last_maintenance_date
   - issue_reported
   - cost

## Features Used for Prediction

- Device age (in days)
- Average daily usage
- Days since last maintenance
- Warranty status

## Model Performance

The application displays:
- Accuracy
- Precision
- Recall
- F1 Score

## Visualizations

1. Feature Importance Chart
2. Risk Distribution Pie Chart
3. Usage Over Time Line Chart
4. Usage vs Days Since Last Maintenance Scatter Plot 