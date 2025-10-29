import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import io
import base64

# ML and Analysis Libraries
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Fitness Data Anomaly Detection",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FitnessDataAnalyzer:
    def __init__(self):
        self.data = None
        self.cleaned_data = None
        self.features = None
        self.anomalies = None
        self.prophet_results = {}
        self.cluster_results = None
        
    # Module 1: Data Collection and Preprocessing
    def load_data(self, file, file_type):
        """Load data from CSV or JSON file"""
        try:
            if file_type == "CSV":
                self.data = pd.read_csv(file)
            else:  # JSON
                self.data = pd.read_json(file)
            return True, f"Data loaded successfully! Shape: {self.data.shape}"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def clean_and_preprocess(self, timestamp_col, value_cols, resample_freq='1H'):
        """Clean and preprocess the fitness data"""
        try:
            df = self.data.copy()
            
            # Convert timestamp
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values(timestamp_col)
            
            # Set timestamp as index
            df.set_index(timestamp_col, inplace=True)
            
            # Interpolate missing values for each metric
            for col in value_cols:
                if col in df.columns:
                    # Forward fill then backward fill for missing values
                    df[col] = df[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            
            # Resample to consistent intervals
            resampled_data = {}
            for col in value_cols:
                if col in df.columns:
                    if resample_freq == '1H':
                        resampled_data[col] = df[col].resample('1H').mean()
                    elif resample_freq == '30min':
                        resampled_data[col] = df[col].resample('30min').mean()
                    elif resample_freq == '1D':
                        resampled_data[col] = df[col].resample('1D').mean()
                    elif resample_freq == '1W':
                        resampled_data[col] = df[col].resample('1W').mean()
            
            self.cleaned_data = pd.DataFrame(resampled_data)
            # Fill any remaining NaN values
            self.cleaned_data = self.cleaned_data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            
            return True, "Data cleaned and preprocessed successfully!"
            
        except Exception as e:
            return False, f"Error in preprocessing: {str(e)}"
    
    # Module 2: Feature Extraction and Modeling
    def extract_features_basic(self, column_name):
        """Extract basic time-series features since TSFresh can be heavy"""
        try:
            if self.cleaned_data is None or column_name not in self.cleaned_data.columns:
                return False, "No cleaned data available or column not found"
            
            features = {}
            data_series = self.cleaned_data[column_name]
            
            # Basic statistical features
            features['mean'] = data_series.mean()
            features['std'] = data_series.std()
            features['min'] = data_series.min()
            features['max'] = data_series.max()
            features['median'] = data_series.median()
            features['variance'] = data_series.var()
            features['skewness'] = data_series.skew()
            features['kurtosis'] = data_series.kurtosis()
            
            # Rolling features
            features['rolling_mean_24h'] = data_series.rolling(window=24, min_periods=1).mean().iloc[-1]
            features['rolling_std_24h'] = data_series.rolling(window=24, min_periods=1).std().iloc[-1]
            
            # Trend features
            if len(data_series) > 1:
                x = np.arange(len(data_series))
                slope = np.polyfit(x, data_series.values, 1)[0]
                features['trend_slope'] = slope
            
            self.features = pd.DataFrame([features])
            return True, f"Extracted {len(features)} basic features"
            
        except Exception as e:
            return False, f"Error in feature extraction: {str(e)}"
    
    def prophet_analysis(self, column_name):
        """Apply Facebook Prophet for trend analysis and anomaly detection"""
        try:
            if self.cleaned_data is None or column_name not in self.cleaned_data.columns:
                return False, "No cleaned data available or column not found", None, None
            
            # Prepare data for Prophet
            prophet_df = self.cleaned_data[[column_name]].copy()
            prophet_df = prophet_df.reset_index()
            prophet_df.columns = ['ds', 'y']
            prophet_df = prophet_df.dropna()
            
            if len(prophet_df) < 10:
                return False, "Not enough data points for Prophet analysis", None, None
            
            # Fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_df)
            
            # Make predictions
            future = model.make_future_dataframe(periods=0, freq='H')
            forecast = model.predict(future)
            
            # Merge with original data
            results = prophet_df.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
            
            # Calculate residuals and anomalies
            results['residual'] = results['y'] - results['yhat']
            residual_std = results['residual'].std()
            results['anomaly_prophet'] = np.abs(results['residual']) > 2 * residual_std
            
            # Store results
            self.prophet_results[column_name] = {
                'results': results,
                'forecast': forecast,
                'model': model
            }
            
            return True, "Prophet analysis completed", results, forecast
            
        except Exception as e:
            return False, f"Error in Prophet analysis: {str(e)}", None, None
    
    def behavioral_clustering(self, n_clusters=3):
        """Apply clustering to identify behavioral patterns"""
        try:
            if self.cleaned_data is None:
                return False, "No cleaned data available", None, None, None, None
            
            # Prepare data for clustering
            cluster_data = self.cleaned_data.copy().dropna()
            
            if len(cluster_data) < n_clusters:
                return False, "Not enough data points for clustering", None, None, None, None
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Apply DBSCAN for outlier detection
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            outlier_labels = dbscan.fit_predict(scaled_data)
            
            # Apply Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_labels = iso_forest.fit_predict(scaled_data)
            
            results = cluster_data.copy()
            results['cluster_kmeans'] = cluster_labels
            results['cluster_dbscan'] = outlier_labels
            results['is_outlier_dbscan'] = outlier_labels == -1
            results['is_outlier_iso'] = iso_labels == -1
            
            # PCA for visualization
            if scaled_data.shape[1] >= 2:
                pca = PCA(n_components=2)
                pca_results = pca.fit_transform(scaled_data)
                results['pca1'] = pca_results[:, 0]
                results['pca2'] = pca_results[:, 1]
            else:
                results['pca1'] = scaled_data[:, 0]
                results['pca2'] = np.zeros_like(scaled_data[:, 0])
            
            self.cluster_results = results
            return True, "Clustering completed", results, kmeans, dbscan, pca
            
        except Exception as e:
            return False, f"Error in clustering: {str(e)}", None, None, None, None
    
    # Module 3: Anomaly Detection and Visualization
    def detect_anomalies(self, column_name, threshold_std=2):
        """Comprehensive anomaly detection using multiple methods"""
        try:
            if self.cleaned_data is None:
                return False, "No cleaned data available"
            
            anomalies_data = self.cleaned_data.copy()
            
            # Method 1: Rule-based (threshold) anomalies
            if column_name in anomalies_data.columns:
                mean_val = anomalies_data[column_name].mean()
                std_val = anomalies_data[column_name].std()
                upper_threshold = mean_val + threshold_std * std_val
                lower_threshold = mean_val - threshold_std * std_val
                
                anomalies_data['anomaly_threshold'] = (
                    (anomalies_data[column_name] > upper_threshold) | 
                    (anomalies_data[column_name] < lower_threshold)
                )
            else:
                anomalies_data['anomaly_threshold'] = False
            
            # Method 2: Prophet-based anomalies
            if column_name in self.prophet_results:
                prophet_results = self.prophet_results[column_name]['results']
                prophet_anomalies = prophet_results.set_index('ds')['anomaly_prophet']
                prophet_anomalies.name = 'anomaly_prophet'  # Give the series a name
                anomalies_data = anomalies_data.merge(
                    prophet_anomalies, 
                    left_index=True, 
                    right_index=True, 
                    how='left'
                )
                anomalies_data['anomaly_prophet'] = anomalies_data['anomaly_prophet'].fillna(False)
            else:
                anomalies_data['anomaly_prophet'] = False
            
            # Method 3: Clustering-based anomalies
            if self.cluster_results is not None:
                # Combine multiple clustering methods
                cluster_anomalies = self.cluster_results[['is_outlier_dbscan', 'is_outlier_iso']].any(axis=1)
                cluster_anomalies.name = 'is_outlier'  # Give the series a name
                anomalies_data = anomalies_data.merge(
                    cluster_anomalies, 
                    left_index=True, 
                    right_index=True, 
                    how='left'
                )
                anomalies_data['is_outlier'] = anomalies_data['is_outlier'].fillna(False)
            else:
                anomalies_data['is_outlier'] = False
            
            # Combined anomaly score
            anomalies_data['anomaly_score'] = (
                anomalies_data['anomaly_threshold'].astype(int) +
                anomalies_data['anomaly_prophet'].astype(int) +
                anomalies_data['is_outlier'].astype(int)
            )
            
            anomalies_data['final_anomaly'] = anomalies_data['anomaly_score'] >= 2
            
            self.anomalies = anomalies_data
            
            return True, "Anomaly detection completed successfully!"
            
        except Exception as e:
            return False, f"Error in anomaly detection: {str(e)}"
    
    def create_visualizations(self, column_name):
        """Create comprehensive visualizations"""
        try:
            if self.cleaned_data is None:
                return None, None, None
            
            # Time series with anomalies
            fig_time_series = go.Figure()
            
            # Main time series
            fig_time_series.add_trace(go.Scatter(
                x=self.cleaned_data.index,
                y=self.cleaned_data[column_name],
                mode='lines',
                name=column_name,
                line=dict(color='blue', width=2)
            ))
            
            # Anomaly points if available
            if self.anomalies is not None and 'final_anomaly' in self.anomalies.columns:
                anomaly_points = self.anomalies[self.anomalies['final_anomaly']]
                if not anomaly_points.empty:
                    fig_time_series.add_trace(go.Scatter(
                        x=anomaly_points.index,
                        y=anomaly_points[column_name],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=8, symbol='x', line=dict(width=2))
                    ))
            
            fig_time_series.update_layout(
                title=f'Time Series with Anomaly Detection - {column_name}',
                xaxis_title='Time',
                yaxis_title=column_name,
                height=400,
                showlegend=True
            )
            
            # Distribution plot
            fig_distribution = px.histogram(
                self.cleaned_data, 
                x=column_name,
                title=f'Distribution of {column_name}',
                nbins=50,
                color_discrete_sequence=['lightblue']
            )
            fig_distribution.update_layout(height=400)
            
            # Anomaly breakdown if available
            if self.anomalies is not None and 'final_anomaly' in self.anomalies.columns:
                anomaly_breakdown = pd.DataFrame({
                    'Method': ['Threshold', 'Prophet', 'Clustering', 'Final'],
                    'Count': [
                        self.anomalies['anomaly_threshold'].sum() if 'anomaly_threshold' in self.anomalies.columns else 0,
                        self.anomalies['anomaly_prophet'].sum() if 'anomaly_prophet' in self.anomalies.columns else 0,
                        self.anomalies['is_outlier'].sum() if 'is_outlier' in self.anomalies.columns else 0,
                        self.anomalies['final_anomaly'].sum() if 'final_anomaly' in self.anomalies.columns else 0
                    ]
                })
                
                fig_breakdown = px.bar(
                    anomaly_breakdown,
                    x='Method',
                    y='Count',
                    title='Anomaly Detection Breakdown',
                    color='Method',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_breakdown.update_layout(height=400)
            else:
                fig_breakdown = go.Figure()
                fig_breakdown.add_annotation(text="No anomaly data available",
                                           xref="paper", yref="paper",
                                           x=0.5, y=0.5, showarrow=False)
                fig_breakdown.update_layout(height=400, title="Anomaly Detection Breakdown")
            
            return fig_time_series, fig_distribution, fig_breakdown
            
        except Exception as e:
            st.error(f"Error creating visualizations: {str(e)}")
            return None, None, None

def create_sample_data():
    """Create sample fitness data for testing"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    np.random.seed(42)
    
    # Create realistic fitness data with some anomalies
    heart_rate = 65 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + np.random.normal(0, 3, len(dates))
    steps = np.maximum(0, np.random.poisson(50, len(dates)) + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24))
    sleep_hours = np.clip(np.random.normal(7, 1, len(dates)), 4, 10)
    
    # Add some anomalies
    heart_rate[100] = 120  # High heart rate anomaly
    heart_rate[200] = 45   # Low heart rate anomaly
    steps[150] = 300       # High steps anomaly
    sleep_hours[250] = 12  # Long sleep anomaly
    steps[50] = 0          # Zero steps anomaly
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'heart_rate': np.round(heart_rate, 1),
        'steps': steps.astype(int),
        'sleep_hours': np.round(sleep_hours, 1)
    })
    
    return sample_data

def main():
    st.title("🏃‍♂️ Fitness Data Anomaly Detection Dashboard")
    st.markdown("""
    This application analyzes fitness tracker data (heart rate, steps, sleep) to detect anomalies 
    and identify behavioral patterns using advanced time-series analysis and machine learning.
    """)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = FitnessDataAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    modules = [
        "Module 1: Data Upload & Preprocessing",
        "Module 2: Feature Extraction & Modeling", 
        "Module 3: Anomaly Detection & Visualization",
        "Module 4: Dashboard & Reports"
    ]
    selected_module = st.sidebar.selectbox("Select Module", modules)
    
    # Quick start with sample data
    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Start")
    if st.sidebar.button("Load Sample Data"):
        sample_data = create_sample_data()
        analyzer.data = sample_data
        st.sidebar.success("Sample data loaded! Go to Module 1 for preprocessing.")
    
    # Module 1: Data Upload & Preprocessing
    if selected_module == "Module 1: Data Upload & Preprocessing":
        st.header("📊 Data Upload & Preprocessing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Fitness Data")
            file_type = st.radio("Select file type:", ["CSV", "JSON"])
            uploaded_file = st.file_uploader(
                f"Upload {file_type} file", 
                type=['csv', 'json'],
                help="Upload your fitness data in CSV or JSON format"
            )
            
            if uploaded_file is not None:
                success, message = analyzer.load_data(uploaded_file, file_type)
                if success:
                    st.success(message)
                    
                    # Display data preview
                    st.subheader("Data Preview")
                    st.dataframe(analyzer.data.head(10))
                    
                    st.subheader("Data Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Shape:** {analyzer.data.shape}")
                        st.write("**Columns:**", list(analyzer.data.columns))
                    with col2:
                        st.write("**Data Types:**")
                        st.write(analyzer.data.dtypes.astype(str))
                    
                else:
                    st.error(message)
        
        with col2:
            st.subheader("Data Preprocessing")
            
            if analyzer.data is not None:
                # Auto-detect timestamp column
                timestamp_candidates = [col for col in analyzer.data.columns if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp'])]
                if timestamp_candidates:
                    default_timestamp = timestamp_candidates[0]
                else:
                    default_timestamp = analyzer.data.columns[0]
                
                timestamp_col = st.selectbox(
                    "Select timestamp column:", 
                    analyzer.data.columns,
                    index=list(analyzer.data.columns).index(default_timestamp) if default_timestamp in analyzer.data.columns else 0
                )
                
                numeric_cols = analyzer.data.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    st.warning("No numeric columns found in the data!")
                    value_cols = []
                else:
                    value_cols = st.multiselect(
                        "Select metrics to analyze:",
                        numeric_cols,
                        default=numeric_cols[:min(3, len(numeric_cols))]
                    )
                
                resample_freq = st.selectbox(
                    "Resampling frequency:",
                    ['1H', '30min', '1D', '1W'],
                    index=0
                )
                
                if st.button("Preprocess Data") and value_cols:
                    with st.spinner("Preprocessing data..."):
                        success, message = analyzer.clean_and_preprocess(
                            timestamp_col, value_cols, resample_freq
                        )
                        if success:
                            st.success(message)
                            
                            # Display cleaned data
                            st.subheader("Cleaned Data Preview")
                            st.dataframe(analyzer.cleaned_data.head(10))
                            
                            # Basic statistics
                            st.subheader("Basic Statistics")
                            st.dataframe(analyzer.cleaned_data.describe())
                            
                            # Data overview with visualizations
                            st.subheader("Data Overview")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Time Period", 
                                         f"{analyzer.cleaned_data.index.min().strftime('%Y-%m-%d')} to {analyzer.cleaned_data.index.max().strftime('%Y-%m-%d')}")
                            with col2:
                                st.metric("Total Records", len(analyzer.cleaned_data))
                            with col3:
                                st.metric("Metrics", len(analyzer.cleaned_data.columns))
                            
                            # Show initial visualizations
                            st.subheader("Initial Data Visualizations")
                            if len(value_cols) > 0:
                                selected_viz_col = st.selectbox("Select metric to visualize:", value_cols)
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=analyzer.cleaned_data.index,
                                    y=analyzer.cleaned_data[selected_viz_col],
                                    mode='lines',
                                    name=selected_viz_col,
                                    line=dict(color='blue', width=2)
                                ))
                                fig.update_layout(
                                    title=f'{selected_viz_col} Over Time',
                                    xaxis_title='Time',
                                    yaxis_title=selected_viz_col,
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Distribution plot
                                fig_hist = px.histogram(
                                    analyzer.cleaned_data,
                                    x=selected_viz_col,
                                    title=f'Distribution of {selected_viz_col}',
                                    nbins=30
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                        else:
                            st.error(message)
            else:
                st.info("Please upload data or use sample data to get started.")
    
    # Module 2: Feature Extraction & Modeling
    elif selected_module == "Module 2: Feature Extraction & Modeling":
        st.header("🔍 Feature Extraction & Modeling")
        
        if analyzer.cleaned_data is None:
            st.warning("Please complete data preprocessing in Module 1 first.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Feature Extraction")
                selected_column = st.selectbox(
                    "Select column for feature extraction:",
                    analyzer.cleaned_data.columns
                )
                
                if st.button("Extract Basic Features"):
                    with st.spinner("Extracting features..."):
                        success, message = analyzer.extract_features_basic(selected_column)
                        if success:
                            st.success(message)
                            if analyzer.features is not None:
                                st.write(f"**Feature Matrix Shape:** {analyzer.features.shape}")
                                # Display features in a nice format
                                features_df = analyzer.features.T.reset_index()
                                features_df.columns = ['Feature', 'Value']
                                st.dataframe(features_df.style.format({'Value': '{:.4f}'}))
            
            with col2:
                st.subheader("Prophet Trend Analysis")
                prophet_column = st.selectbox(
                    "Select column for Prophet analysis:",
                    analyzer.cleaned_data.columns,
                    key="prophet_col"
                )
                
                if st.button("Run Prophet Analysis"):
                    with st.spinner("Running Prophet analysis..."):
                        success, message, results, forecast = analyzer.prophet_analysis(prophet_column)
                        if success:
                            st.success(message)
                            
                            # Display Prophet forecast
                            if forecast is not None and results is not None:
                                fig_forecast = go.Figure()
                                
                                # Actual values
                                fig_forecast.add_trace(go.Scatter(
                                    x=results['ds'],
                                    y=results['y'],
                                    mode='lines',
                                    name='Actual',
                                    line=dict(color='blue')
                                ))
                                
                                # Predicted values
                                fig_forecast.add_trace(go.Scatter(
                                    x=forecast['ds'],
                                    y=forecast['yhat'],
                                    mode='lines',
                                    name='Predicted',
                                    line=dict(color='red', dash='dash')
                                ))
                                
                                # Confidence interval
                                fig_forecast.add_trace(go.Scatter(
                                    x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                                    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                                    fill='toself',
                                    fillcolor='rgba(255,0,0,0.2)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name='Confidence Interval',
                                    showlegend=True
                                ))
                                
                                fig_forecast.update_layout(
                                    title=f'Prophet Forecast - {prophet_column}',
                                    xaxis_title='Date',
                                    yaxis_title=prophet_column,
                                    height=400
                                )
                                st.plotly_chart(fig_forecast, use_container_width=True)
                            
                            # Show anomalies detected by Prophet
                            if results is not None and 'anomaly_prophet' in results.columns:
                                prophet_anomalies = results[results['anomaly_prophet']]
                                st.write(f"**Prophet detected {len(prophet_anomalies)} anomalies**")
                                if not prophet_anomalies.empty:
                                    st.dataframe(prophet_anomalies[['ds', 'y', 'residual']].head(10))
                        else:
                            st.error(message)
            
            st.subheader("Behavioral Pattern Clustering")
            n_clusters = st.slider("Number of clusters:", 2, 5, 3)
            
            if st.button("Perform Clustering"):
                with st.spinner("Performing clustering analysis..."):
                    success, message, results, kmeans, dbscan, pca = analyzer.behavioral_clustering(n_clusters)
                    if success:
                        st.success(message)
                        
                        # Clustering results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # KMeans clusters visualization
                            if 'pca1' in results.columns and 'pca2' in results.columns:
                                fig_clusters = px.scatter(
                                    results.reset_index(), 
                                    x='pca1', 
                                    y='pca2',
                                    color='cluster_kmeans',
                                    title='KMeans Clustering (PCA Visualization)',
                                    hover_data=analyzer.cleaned_data.columns.tolist()
                                )
                                st.plotly_chart(fig_clusters, use_container_width=True)
                        
                        with col2:
                            # Outlier detection results
                            outlier_counts = {
                                'DBSCAN Outliers': results['is_outlier_dbscan'].sum(),
                                'Isolation Forest Outliers': results['is_outlier_iso'].sum()
                            }
                            
                            fig_outliers = px.bar(
                                x=list(outlier_counts.keys()),
                                y=list(outlier_counts.values()),
                                title='Outlier Detection Results',
                                color=list(outlier_counts.keys()),
                                color_discrete_sequence=['red', 'orange']
                            )
                            st.plotly_chart(fig_outliers, use_container_width=True)
                        
                        # Cluster statistics
                        st.subheader("Cluster Statistics")
                        if 'cluster_kmeans' in results.columns:
                            cluster_stats = results.groupby('cluster_kmeans').agg({
                                col: ['mean', 'std'] for col in analyzer.cleaned_data.columns
                            })
                            st.dataframe(cluster_stats.round(3))
                    else:
                        st.error(message)
    
    # Module 3: Anomaly Detection & Visualization
    elif selected_module == "Module 3: Anomaly Detection & Visualization":
        st.header("🚨 Anomaly Detection & Visualization")
        
        if analyzer.cleaned_data is None:
            st.warning("Please complete data preprocessing in Module 1 first.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Anomaly Detection Settings")
                anomaly_column = st.selectbox(
                    "Select column for anomaly detection:",
                    analyzer.cleaned_data.columns
                )
                
                threshold_std = st.slider(
                    "Threshold standard deviations:",
                    min_value=1.0,
                    max_value=3.0,
                    value=2.0,
                    step=0.1
                )
                
                # Run Prophet and Clustering first if not done
                if st.button("Run Complete Analysis"):
                    with st.spinner("Running complete analysis (Prophet + Clustering + Anomaly Detection)..."):
                        # Run Prophet analysis
                        success_prophet, _, _, _ = analyzer.prophet_analysis(anomaly_column)
                        if success_prophet:
                            st.success("✓ Prophet analysis completed")
                        
                        # Run Clustering
                        success_cluster, _, _, _, _, _ = analyzer.behavioral_clustering()
                        if success_cluster:
                            st.success("✓ Clustering completed")
                        
                        # Run Anomaly Detection
                        success_anomaly, message = analyzer.detect_anomalies(anomaly_column, threshold_std)
                        if success_anomaly:
                            st.success("✓ " + message)
                        else:
                            st.error("✗ " + message)
                
                if st.button("Detect Anomalies Only"):
                    with st.spinner("Running anomaly detection..."):
                        success, message = analyzer.detect_anomalies(anomaly_column, threshold_std)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
            
            with col2:
                st.subheader("Anomaly Summary")
                if analyzer.anomalies is not None and 'final_anomaly' in analyzer.anomalies.columns:
                    total_points = len(analyzer.anomalies)
                    anomaly_count = analyzer.anomalies['final_anomaly'].sum()
                    anomaly_percentage = (anomaly_count / total_points) * 100
                    
                    st.metric("Total Data Points", total_points)
                    st.metric("Anomalies Detected", anomaly_count)
                    st.metric("Anomaly Percentage", f"{anomaly_percentage:.2f}%")
                    
                    # Anomaly details
                    st.subheader("Anomaly Details")
                    anomaly_details = analyzer.anomalies[analyzer.anomalies['final_anomaly']]
                    if not anomaly_details.empty:
                        display_cols = [anomaly_column, 'anomaly_score']
                        available_cols = [col for col in display_cols if col in anomaly_details.columns]
                        st.dataframe(anomaly_details[available_cols].head(10))
                    else:
                        st.info("No anomalies detected with current settings.")
                else:
                    st.info("Run anomaly detection to see results here.")
            
            # Visualizations
            st.subheader("Anomaly Visualizations")
            
            if analyzer.cleaned_data is not None:
                fig1, fig2, fig3 = analyzer.create_visualizations(anomaly_column)
                
                if fig1 is not None:
                    st.plotly_chart(fig1, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if fig2 is not None:
                        st.plotly_chart(fig2, use_container_width=True)
                with col2:
                    if fig3 is not None:
                        st.plotly_chart(fig3, use_container_width=True)
                
                # Additional anomaly timeline
                if analyzer.anomalies is not None and 'final_anomaly' in analyzer.anomalies.columns:
                    anomaly_timeline = analyzer.anomalies['final_anomaly'].astype(int).resample('1D').sum()
                    fig_timeline = px.line(
                        x=anomaly_timeline.index,
                        y=anomaly_timeline.values,
                        title='Daily Anomaly Count Timeline',
                        labels={'x': 'Date', 'y': 'Anomaly Count'}
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Module 4: Dashboard & Reports
    elif selected_module == "Module 4: Dashboard & Reports":
        st.header("📈 Comprehensive Dashboard & Reports")
        
        if analyzer.cleaned_data is None:
            st.warning("Please complete data preprocessing in Module 1 first.")
        else:
            # Summary statistics
            st.subheader("📊 Overall Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(analyzer.cleaned_data))
            with col2:
                st.metric("Metrics Tracked", len(analyzer.cleaned_data.columns))
            with col3:
                date_range = analyzer.cleaned_data.index.max() - analyzer.cleaned_data.index.min()
                st.metric("Data Duration", f"{date_range.days} days")
            with col4:
                if analyzer.anomalies is not None and 'final_anomaly' in analyzer.anomalies.columns:
                    anomaly_count = analyzer.anomalies['final_anomaly'].sum()
                    st.metric("Total Anomalies", anomaly_count)
                else:
                    st.metric("Total Anomalies", "N/A")
            
            # Interactive metric selection
            st.subheader("📈 Metric Analysis")
            selected_metric = st.selectbox(
                "Select metric to analyze:",
                analyzer.cleaned_data.columns
            )
            
            # Comprehensive charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Time series with rolling average
                fig_rolling = go.Figure()
                fig_rolling.add_trace(go.Scatter(
                    x=analyzer.cleaned_data.index,
                    y=analyzer.cleaned_data[selected_metric],
                    mode='lines',
                    name=selected_metric,
                    line=dict(color='lightblue')
                ))
                
                # Rolling average
                rolling_avg = analyzer.cleaned_data[selected_metric].rolling(window=24, min_periods=1).mean()
                fig_rolling.add_trace(go.Scatter(
                    x=analyzer.cleaned_data.index,
                    y=rolling_avg,
                    mode='lines',
                    name='24h Rolling Avg',
                    line=dict(color='blue')
                ))
                
                fig_rolling.update_layout(
                    title=f'{selected_metric} with Rolling Average',
                    height=400
                )
                st.plotly_chart(fig_rolling, use_container_width=True)
                
                # Distribution by time of day
                if hasattr(analyzer.cleaned_data.index, 'hour'):
                    hourly_avg = analyzer.cleaned_data.groupby(
                        analyzer.cleaned_data.index.hour
                    )[selected_metric].mean()
                    
                    fig_hourly = px.bar(
                        x=hourly_avg.index,
                        y=hourly_avg.values,
                        title=f'{selected_metric} - Average by Hour of Day',
                        labels={'x': 'Hour of Day', 'y': selected_metric}
                    )
                    st.plotly_chart(fig_hourly, use_container_width=True)
            
            with col2:
                # Box plot for distribution
                fig_box = px.box(
                    analyzer.cleaned_data, 
                    y=selected_metric,
                    title=f'{selected_metric} Distribution'
                )
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Correlation heatmap
                if len(analyzer.cleaned_data.columns) > 1:
                    corr_matrix = analyzer.cleaned_data.corr()
                    fig_heatmap = px.imshow(
                        corr_matrix,
                        title='Metrics Correlation Heatmap',
                        aspect='auto',
                        color_continuous_scale='RdBu_r'
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Downloadable report
            st.subheader("📥 Generate Report")
            
            if st.button("Generate Comprehensive Report"):
                with st.spinner("Generating report..."):
                    # Create a summary report
                    report_data = {
                        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "data_shape": analyzer.cleaned_data.shape,
                        "data_columns": list(analyzer.cleaned_data.columns),
                        "date_range": {
                            "start": analyzer.cleaned_data.index.min().strftime("%Y-%m-%d"),
                            "end": analyzer.cleaned_data.index.max().strftime("%Y-%m-%d")
                        },
                        "summary_statistics": analyzer.cleaned_data.describe().to_dict(),
                    }
                    
                    if analyzer.anomalies is not None and 'final_anomaly' in analyzer.anomalies.columns:
                        report_data["anomaly_summary"] = {
                            "total_anomalies": int(analyzer.anomalies['final_anomaly'].sum()),
                            "anomaly_percentage": f"{(analyzer.anomalies['final_anomaly'].sum() / len(analyzer.anomalies)) * 100:.2f}%",
                            "anomaly_breakdown": {
                                "threshold_based": int(analyzer.anomalies['anomaly_threshold'].sum()) if 'anomaly_threshold' in analyzer.anomalies.columns else 0,
                                "prophet_based": int(analyzer.anomalies['anomaly_prophet'].sum()) if 'anomaly_prophet' in analyzer.anomalies.columns else 0,
                                "clustering_based": int(analyzer.anomalies['is_outlier'].sum()) if 'is_outlier' in analyzer.anomalies.columns else 0
                            }
                        }
                    
                    # Convert to JSON for download
                    report_json = json.dumps(report_data, indent=2)
                    
                    st.success("Report generated successfully!")
                    
                    # Download button
                    st.download_button(
                        label="Download Report (JSON)",
                        data=report_json,
                        file_name=f"fitness_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                    # Display report preview
                    with st.expander("Report Preview"):
                        st.json(report_data)

if __name__ == "__main__":
    main()