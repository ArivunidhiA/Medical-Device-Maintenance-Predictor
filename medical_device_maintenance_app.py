import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Medical Device Maintenance Predictor",
    page_icon="🏥",
    layout="wide"
)

# Cache data generation functions
@st.cache_data
def generate_device_master():
    np.random.seed(42)
    n_devices = 100
    
    device_types = ['MRI', 'CT Scanner', 'X-Ray', 'Ultrasound', 'ECG', 'Ventilator']
    facilities = ['Main Hospital', 'North Wing', 'South Wing', 'Emergency']
    departments = ['Radiology', 'Cardiology', 'Emergency', 'ICU', 'General']
    statuses = ['Active', 'Inactive', 'Maintenance']
    
    data = {
        'device_id': range(1, n_devices + 1),
        'device_type': np.random.choice(device_types, n_devices),
        'purchase_date': pd.date_range(start='2020-01-01', periods=n_devices, freq='D'),
        'warranty_expiry': pd.date_range(start='2023-01-01', periods=n_devices, freq='D'),
        'facility': np.random.choice(facilities, n_devices),
        'department': np.random.choice(departments, n_devices),
        'status': np.random.choice(statuses, n_devices)
    }
    return pd.DataFrame(data)

@st.cache_data
def generate_usage_logs():
    np.random.seed(42)
    n_devices = 100
    n_days = 90
    
    device_ids = range(1, n_devices + 1)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    usage_data = []
    for device_id in device_ids:
        base_usage = np.random.normal(8, 3)  # Different base usage for each device
        for date in dates:
            # Add daily variation
            usage_data.append({
                'device_id': device_id,
                'usage_hours': max(0, np.random.normal(base_usage, 2)),  # Ensure non-negative
                'usage_date': date
            })
    
    return pd.DataFrame(usage_data)

@st.cache_data
def generate_maintenance_logs():
    np.random.seed(42)
    n_devices = 100
    
    issues = ['Routine', 'Malfunction', 'Calibration', 'Repair', 'Inspection']
    
    # Generate more varied maintenance dates
    maintenance_dates = []
    for _ in range(n_devices):
        # Random date between 2023-01-01 and now, with some very old dates
        if np.random.random() < 0.2:  # 20% chance of old maintenance
            days_ago = np.random.randint(300, 500)
        else:
            days_ago = np.random.randint(1, 300)
        date = datetime.now() - timedelta(days=days_ago)
        maintenance_dates.append(date)
    
    data = {
        'device_id': range(1, n_devices + 1),
        'last_maintenance_date': maintenance_dates,
        'issue_reported': np.random.choice(issues, n_devices),
        'cost': np.random.uniform(100, 5000, n_devices)
    }
    return pd.DataFrame(data)

# Cache data preprocessing
@st.cache_data
def preprocess_data(device_master, usage_logs, maintenance_logs):
    # Calculate device age
    device_master['device_age'] = (datetime.now() - device_master['purchase_date']).dt.days
    
    # Calculate average daily usage
    avg_usage = usage_logs.groupby('device_id')['usage_hours'].mean().reset_index()
    device_master = device_master.merge(avg_usage, on='device_id', how='left')
    device_master = device_master.rename(columns={'usage_hours': 'avg_daily_usage'})
    
    # Calculate days since last maintenance
    device_master = device_master.merge(
        maintenance_logs[['device_id', 'last_maintenance_date']],
        on='device_id',
        how='left'
    )
    device_master['days_since_last_maintenance'] = (
        datetime.now() - device_master['last_maintenance_date']
    ).dt.days
    
    # Calculate warranty status
    device_master['warranty_expired'] = (datetime.now() > device_master['warranty_expiry']).astype(int)
    
    # Normalize numerical features
    device_master['device_age_norm'] = device_master['device_age'] / 365.0  # Convert to years
    device_master['days_since_maintenance_norm'] = device_master['days_since_last_maintenance'] / 180.0  # Normalize by 6 months
    
    # Create target variable with more randomness
    maintenance_score = (
        0.3 * device_master['days_since_maintenance_norm'] +
        0.3 * (device_master['avg_daily_usage'] / 12.0) +  # Normalize by 12 hours
        0.2 * device_master['device_age_norm'] +
        0.2 * device_master['warranty_expired'].astype(float)
    )
    
    # Add significant random noise
    np.random.seed(42)  # For reproducibility
    random_factor = np.random.normal(0, 0.3, len(device_master))  # Increased noise
    maintenance_prob = maintenance_score + random_factor
    
    # Normalize probabilities to [0, 1] range
    maintenance_prob = (maintenance_prob - maintenance_prob.min()) / (maintenance_prob.max() - maintenance_prob.min())
    
    # Create binary target with threshold
    device_master['needs_maintenance'] = (maintenance_prob > 0.5).astype(int)
    
    return device_master

# Cache model training
@st.cache_resource
def train_model(X_train, X_test, y_train, y_test):
    # Use a more balanced model configuration
    model = RandomForestClassifier(
        n_estimators=50,  # Reduced number of trees
        max_depth=5,      # Limit tree depth
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return model, metrics

def main():
    st.title("🏥 Medical Device Maintenance Predictor")
    st.write("""
    This application predicts which medical devices are likely to need maintenance 
    within the next 30 days using machine learning.
    """)
    
    # Sidebar for data upload
    st.sidebar.header("Data Upload")
    uploaded_device = st.sidebar.file_uploader("Upload Device Master Data", type=['csv'])
    uploaded_usage = st.sidebar.file_uploader("Upload Usage Logs", type=['csv'])
    uploaded_maintenance = st.sidebar.file_uploader("Upload Maintenance Logs", type=['csv'])
    
    # Load or generate data
    if uploaded_device and uploaded_usage and uploaded_maintenance:
        device_master = pd.read_csv(uploaded_device)
        usage_logs = pd.read_csv(uploaded_usage)
        maintenance_logs = pd.read_csv(uploaded_maintenance)
    else:
        device_master = generate_device_master()
        usage_logs = generate_usage_logs()
        maintenance_logs = generate_maintenance_logs()
    
    # Preprocess data
    processed_data = preprocess_data(device_master, usage_logs, maintenance_logs)
    
    # Show raw data
    with st.expander("View Raw Data"):
        tab1, tab2, tab3 = st.tabs(["Device Master", "Usage Logs", "Maintenance Logs"])
        with tab1:
            st.dataframe(device_master)
        with tab2:
            st.dataframe(usage_logs)
        with tab3:
            st.dataframe(maintenance_logs)
    
    # Feature selection and model training
    features = ['device_age', 'avg_daily_usage', 'days_since_last_maintenance', 'warranty_expired']
    X = processed_data[features]
    y = processed_data['needs_maintenance']
    
    # Use stratified split with larger test size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3,  # Increased test size
        random_state=42,
        stratify=y  # Ensure balanced split
    )
    
    # Train model
    model, metrics = train_model(X_train, X_test, y_train, y_test)
    
    # Display metrics
    st.header("Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.2%}")
    with col4:
        st.metric("F1 Score", f"{metrics['f1']:.2%}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig_importance = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        title='Feature Importance'
    )
    st.plotly_chart(fig_importance)
    
    # Predictions and visualizations
    st.header("Device Maintenance Predictions")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        selected_facility = st.multiselect(
            "Filter by Facility",
            options=processed_data['facility'].unique(),
            default=processed_data['facility'].unique()
        )
    with col2:
        selected_department = st.multiselect(
            "Filter by Department",
            options=processed_data['department'].unique(),
            default=processed_data['department'].unique()
        )
    
    # Filter data
    filtered_data = processed_data.loc[
        (processed_data['facility'].isin(selected_facility)) &
        (processed_data['department'].isin(selected_department))
    ].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Make predictions only if there is filtered data
    if not filtered_data.empty:
        predictions = model.predict_proba(filtered_data[features])
        # Handle case where all predictions are of one class
        if predictions.shape[1] == 1:
            filtered_data.loc[:, 'maintenance_probability'] = predictions[:, 0]
        else:
            filtered_data.loc[:, 'maintenance_probability'] = predictions[:, 1]
        
        # Display predictions table
        st.dataframe(
            filtered_data[['device_id', 'device_type', 'facility', 'department', 'maintenance_probability']]
            .style.background_gradient(subset=['maintenance_probability'], cmap='RdYlGn_r')
        )
    else:
        st.warning("No data available for the selected filters. Please adjust your selection.")
    
    # Visualizations
    st.header("Data Visualizations")
    
    if not filtered_data.empty:
        # Risk distribution
        fig_risk = px.pie(
            filtered_data,
            names='needs_maintenance',
            title='Distribution of High-Risk vs Low-Risk Devices',
            labels={0: 'Low Risk', 1: 'High Risk'}
        )
        st.plotly_chart(fig_risk)
        
        # Usage over time
        usage_over_time = usage_logs.groupby('usage_date')['usage_hours'].mean().reset_index()
        fig_usage = px.line(
            usage_over_time,
            x='usage_date',
            y='usage_hours',
            title='Average Daily Usage Over Time'
        )
        st.plotly_chart(fig_usage)
        
        # Scatter plot
        fig_scatter = px.scatter(
            filtered_data,
            x='avg_daily_usage',
            y='days_since_last_maintenance',
            color='needs_maintenance',
            title='Usage vs Days Since Last Maintenance',
            labels={'needs_maintenance': 'Needs Maintenance'}
        )
        st.plotly_chart(fig_scatter)
    else:
        st.warning("No data available for visualizations. Please adjust your filters.")

if __name__ == "__main__":
    main() 