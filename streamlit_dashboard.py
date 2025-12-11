import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>📊 Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)

# ======================== Load Data and Model ========================
@st.cache_resource
def load_model():
    try:
        with open(r'churn_Random_model.pkl', 'rb') as f:
            model = pickle.load(f)
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Predicted_churn.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load resources
model = load_model()
data = load_data()

# Check if resources loaded correctly
if model is None or data is None:
    st.error("Failed to load model or data. Please ensure both files are in the correct location.")
    st.stop()

# ======================== Sidebar Navigation ========================
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Select a page:", 
    ["📈 Dashboard", "🔮 Single Prediction", "📊 Data Explorer", "📉 Model Analytics", "⚙️ Model Settings"])

# ======================== PAGE 1: DASHBOARD ========================
if page == "📈 Dashboard":
    st.header("Overview Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(data)
    churn_count = (data['Churn_prediction'] == 1).sum()
    churn_rate = (churn_count / total_customers) * 100
    avg_tenure = data['Tenure'].mean()
    
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churn_count:,}", delta=f"{churn_rate:.1f}%")
    col3.metric("Churn Rate", f"{churn_rate:.2f}%")
    col4.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")
    
    st.divider()
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    # Churn Distribution
    with col1:
        st.subheader("Churn Distribution")
        churn_counts = data['Churn_prediction'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['Not Churned', 'Churned'],
            values=[churn_counts[0], churn_counts[1]],
            marker=dict(colors=['#2ecc71', '#e74c3c']),
            textposition='inside',
            textinfo='label+percent'
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Age vs Churn
    with col2:
        st.subheader("Age vs Churn Status")
        fig = px.box(data, x='Churn_prediction', y='Age', 
                     labels={'Churn_prediction': 'Churn Status'},
                     color='Churn_prediction',
                     color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
        fig.update_xaxes(ticktext=['Not Churned', 'Churned'], tickvals=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # Monthly Charges vs Churn
    with col1:
        st.subheader("Monthly Charges vs Churn")
        fig = px.scatter(data, x='MonthlyCharges', y='TotalCharges', 
                        color='Churn_prediction',
                        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                        size='Age', hover_data=['CustomerID', 'Tenure'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Tenure Distribution
    with col2:
        st.subheader("Tenure Distribution by Churn")
        fig = px.histogram(data, x='Tenure', color='Churn_prediction',
                          nbins=30,
                          color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                          labels={'Churn_prediction': 'Churn Status'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Internet Usage Stats
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Internet Usage vs Churn")
        fig = px.box(data, x='Churn_prediction', y='InternetUsageGB',
                    color='Churn_prediction',
                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
        fig.update_xaxes(ticktext=['Not Churned', 'Churned'], tickvals=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Calls Per Month by Churn")
        fig = px.box(data, x='Churn_prediction', y='CallsPerMonth',
                    color='Churn_prediction',
                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
        fig.update_xaxes(ticktext=['Not Churned', 'Churned'], tickvals=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

# ======================== PAGE 2: SINGLE PREDICTION ========================
elif page == "🔮 Single Prediction":
    st.header("Make a Single Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=24)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0, step=5.0)
    
    with col2:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1500.0, step=100.0)
        internet_usage = st.number_input("Internet Usage (GB)", min_value=0.0, value=200.0, step=10.0)
        calls_per_month = st.number_input("Calls Per Month", min_value=0, max_value=200, value=50)
    
    with col3:
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        subscription_type = st.selectbox("Subscription Type", [0, 1, 2])
        payment_method = st.selectbox("Payment Method", [0, 1, 2, 3])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        contract_type = st.selectbox("Contract Type", [0, 1, 2])
    with col2:
        has_dependents = st.selectbox("Has Dependents", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col3:
        complaints = st.number_input("Number of Complaints", min_value=0, max_value=20, value=0)
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'SubscriptionType': [subscription_type],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'InternetUsageGB': [internet_usage],
        'CallsPerMonth': [calls_per_month],
        'Complaints': [complaints],
        'PaymentMethod': [payment_method],
        'ContractType': [contract_type],
        'HasDependents': [has_dependents]
    })
    
    st.divider()
    
    if st.button("🔮 Predict Churn Status", use_container_width=True, type="primary"):
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_prob = model.predict_proba(input_data)[0]
        
        st.success("✅ Prediction Complete!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.metric("Churn Status", "Not Churned ✅", delta="0%", delta_color="inverse")
                st.info(f"**Churn Probability: {prediction_prob[1]*100:.2f}%**")
            else:
                st.metric("Churn Status", "Will Churn ⚠️", delta="High Risk", delta_color="off")
                st.warning(f"**Churn Probability: {prediction_prob[1]*100:.2f}%**")
        
        with col2:
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction_prob[1]*100,
                title="Churn Risk %",
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': '#e74c3c'},
                    'steps': [
                        {'range': [0, 30], 'color': "#2ecc71"},
                        {'range': [30, 70], 'color': "#f39c12"},
                        {'range': [70, 100], 'color': "#e74c3c"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction details
        st.subheader("Input Summary")
        input_display = input_data.copy()
        st.dataframe(input_display, use_container_width=True)

# ======================== PAGE 3: DATA EXPLORER ========================
elif page == "📊 Data Explorer":
    st.header("Data Explorer with Filters")
    
    # Add filters
    st.subheader("Filter Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_range = st.slider("Age Range", int(data['Age'].min()), int(data['Age'].max()), 
                             (int(data['Age'].min()), int(data['Age'].max())))
    
    with col2:
        tenure_range = st.slider("Tenure Range (months)", int(data['Tenure'].min()), int(data['Tenure'].max()), 
                                (int(data['Tenure'].min()), int(data['Tenure'].max())))
    
    with col3:
        churn_filter = st.multiselect("Churn Status", [0, 1], default=[0, 1],
                                     format_func=lambda x: "Not Churned" if x == 0 else "Churned")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_charges = st.number_input("Min Monthly Charges", min_value=0.0, value=0.0, step=100.0)
    
    with col2:
        max_charges = st.number_input("Max Monthly Charges", min_value=0.0, value=data['MonthlyCharges'].max(), step=100.0)
    
    with col3:
        complaints_min = st.number_input("Min Complaints", min_value=0, value=0)
    
    # Apply filters
    filtered_data = data[
        (data['Age'] >= age_range[0]) &
        (data['Age'] <= age_range[1]) &
        (data['Tenure'] >= tenure_range[0]) &
        (data['Tenure'] <= tenure_range[1]) &
        (data['Churn_prediction'].isin(churn_filter)) &
        (data['MonthlyCharges'] >= min_charges) &
        (data['MonthlyCharges'] <= max_charges) &
        (data['Complaints'] >= complaints_min)
    ]
    
    st.subheader(f"Filtered Results: {len(filtered_data)} customers")
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", len(filtered_data))
    col2.metric("Avg Age", f"{filtered_data['Age'].mean():.1f}")
    col3.metric("Avg Tenure", f"{filtered_data['Tenure'].mean():.1f}")
    col4.metric("Churn Rate", f"{(filtered_data['Churn_prediction'].sum() / len(filtered_data) * 100):.2f}%")
    
    st.divider()
    
    # Column selector
    st.subheader("Select Columns to Display")
    all_columns = filtered_data.columns.tolist()
    selected_columns = st.multiselect("Display Columns", all_columns, default=all_columns[:8])
    
    # Display data
    st.subheader("Data Table")
    st.dataframe(filtered_data[selected_columns], use_container_width=True, height=400)
    
    # Download filtered data
    csv = filtered_data[selected_columns].to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_churn_data.csv",
        mime="text/csv",
        use_container_width=True
    )

# ======================== PAGE 4: MODEL ANALYTICS ========================
elif page == "📉 Model Analytics":
    st.header("Model Performance Analytics")
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    predictions = data['Churn_prediction'].values
    actual_churn = (data['Churn_prediction'] == 1).sum()
    pred_churn = (data['Churn_prediction'] == 1).sum()
    
    col1.metric("Predicted Churners", pred_churn)
    col2.metric("Total Customers", len(data))
    col3.metric("Prediction Rate", f"{(pred_churn/len(data)*100):.2f}%")
    
    st.divider()
    
    # Feature Importance-like analysis
    st.subheader("Feature Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn by Age Groups")
        data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 20, 30, 40, 50, 60, 100], 
                                  labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+'])
        age_churn = data.groupby('AgeGroup')['Churn_prediction'].agg(['sum', 'count'])
        age_churn['rate'] = (age_churn['sum'] / age_churn['count'] * 100)
        
        fig = px.bar(age_churn.reset_index(), x='AgeGroup', y='rate',
                    title='Churn Rate by Age Group',
                    labels={'rate': 'Churn Rate (%)', 'AgeGroup': 'Age Group'},
                    color='rate',
                    color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Churn by Tenure Groups")
        data['TenureGroup'] = pd.cut(data['Tenure'], bins=[0, 6, 12, 24, 48, 72],
                                     labels=['0-6mo', '6-12mo', '12-24mo', '24-48mo', '48+mo'])
        tenure_churn = data.groupby('TenureGroup')['Churn_prediction'].agg(['sum', 'count'])
        tenure_churn['rate'] = (tenure_churn['sum'] / tenure_churn['count'] * 100)
        
        fig = px.bar(tenure_churn.reset_index(), x='TenureGroup', y='rate',
                    title='Churn Rate by Tenure',
                    labels={'rate': 'Churn Rate (%)', 'TenureGroup': 'Tenure Group'},
                    color='rate',
                    color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("High Risk Customers")
        # Identify high-risk factors
        high_churn_data = data[data['Churn_prediction'] == 1]
        
        st.write("**Average Characteristics of Churned Customers:**")
        comparison = pd.DataFrame({
            'Metric': ['Age', 'Tenure', 'Monthly Charges', 'Total Charges', 'Internet Usage', 'Complaints'],
            'Churned': [
                f"{high_churn_data['Age'].mean():.1f}",
                f"{high_churn_data['Tenure'].mean():.1f}",
                f"${high_churn_data['MonthlyCharges'].mean():.2f}",
                f"${high_churn_data['TotalCharges'].mean():.2f}",
                f"{high_churn_data['InternetUsageGB'].mean():.1f} GB",
                f"{high_churn_data['Complaints'].mean():.2f}"
            ],
            'All': [
                f"{data['Age'].mean():.1f}",
                f"{data['Tenure'].mean():.1f}",
                f"${data['MonthlyCharges'].mean():.2f}",
                f"${data['TotalCharges'].mean():.2f}",
                f"{data['InternetUsageGB'].mean():.1f} GB",
                f"{data['Complaints'].mean():.2f}"
            ]
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Correlation with Churn")
        # Calculate correlations
        numeric_cols = data[['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 
                            'InternetUsageGB', 'CallsPerMonth', 'Complaints']].copy()
        numeric_cols['Churn'] = data['Churn_prediction']
        
        correlations = numeric_cols.corr()['Churn'].drop('Churn').sort_values(ascending=False)
        
        fig = px.bar(x=correlations.values, y=correlations.index,
                    orientation='h',
                    labels={'x': 'Correlation', 'y': 'Feature'},
                    color=correlations.values,
                    color_continuous_scale='RdBu',
                    color_continuous_midpoint=0)
        st.plotly_chart(fig, use_container_width=True)

# ======================== PAGE 5: MODEL SETTINGS ========================
elif page == "⚙️ Model Settings":
    st.header("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Information")
        st.info("""
        **Model Type:** Random Forest Classifier
        
        **Features:** 12
        - Gender
        - Age
        - Tenure
        - Subscription Type
        - Monthly Charges
        - Total Charges
        - Internet Usage (GB)
        - Calls Per Month
        - Complaints
        - Payment Method
        - Contract Type
        - Has Dependents
        
        **Output:** Binary Classification
        - 0 = Not Churned
        - 1 = Churned
        """)
    
    with col2:
        st.subheader("Data Statistics")
        stats_data = {
            'Metric': ['Total Records', 'Churned', 'Not Churned', 'Churn Rate', 'Features'],
            'Value': [
                f"{len(data):,}",
                f"{(data['Churn_prediction'] == 1).sum():,}",
                f"{(data['Churn_prediction'] == 0).sum():,}",
                f"{((data['Churn_prediction'] == 1).sum() / len(data) * 100):.2f}%",
                f"{len(data.columns) - 1}"
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.subheader("Data Sample")
    st.dataframe(data.head(10), use_container_width=True)
    
    st.divider()
    
    # Model feature encoding info
    st.subheader("Feature Encoding Reference")
    encoding_info = {
        'Feature': ['Gender', 'Subscription Type', 'Payment Method', 'Contract Type', 'Has Dependents'],
        'Encoding': [
            '0=Female, 1=Male',
            '0=Basic, 1=Standard, 2=Premium',
            '0=Credit Card, 1=Bank Transfer, 2=Debit Card, 3=Digital Wallet',
            '0=Month-to-Month, 1=One Year, 2=Two Years',
            '0=No, 1=Yes'
        ]
    }
    encoding_df = pd.DataFrame(encoding_info)
    st.dataframe(encoding_df, use_container_width=True, hide_index=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9rem;'>
📊 Churn Prediction Dashboard | Powered by Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
