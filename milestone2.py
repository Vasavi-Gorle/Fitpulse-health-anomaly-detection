import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Feature Extraction
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters

# Time Series Modeling
from prophet import Prophet

# Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

class DataUploader:
    """Handle raw data upload and preview"""
    
    def __init__(self):
        self.uploaded_files = {}
        self.raw_data = {}
        self.preprocessed_data = {}
    
    def upload_raw_data(self):
        """Upload and preview raw health data"""
        st.sidebar.header("📁 Data Upload")
        
        # File upload options
        uploaded_file = st.sidebar.file_uploader(
            "Upload Health Data (CSV/Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your FitPulse health data in CSV or Excel format"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Store the raw data
                self.raw_data['uploaded'] = df
                
                # Show data preview
                st.subheader("📊 Raw Data Preview")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    st.metric("Data Types", f"{len(df.select_dtypes(include=[np.number]).columns)} numeric")
                
                # Show data sample
                st.write("**Data Sample (First 10 rows):**")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Show data info
                with st.expander("📋 Data Information"):
                    st.write("**Column Names and Data Types:**")
                    st.write(df.dtypes)
                    
                    st.write("**Basic Statistics:**")
                    st.write(df.describe())
                
                # Show missing values
                with st.expander("🔍 Missing Values Analysis"):
                    missing_data = df.isnull().sum()
                    if missing_data.sum() > 0:
                        st.warning(f"Found {missing_data.sum()} missing values")
                        missing_df = pd.DataFrame({
                            'Column': missing_data.index,
                            'Missing_Values': missing_data.values,
                            'Percentage': (missing_data.values / len(df)) * 100
                        })
                        st.dataframe(missing_df[missing_df['Missing_Values'] > 0], use_container_width=True)
                    else:
                        st.success("No missing values found!")
                
                return df
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return None
        
        return None

class Milestone1Preprocessor:
    """Milestone 1 data preprocessing functionality"""
    
    def __init__(self):
        self.processed_data = {}
        self.preprocessing_reports = {}
    
    def preprocess_health_data(self, raw_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Preprocess raw health data (Milestone 1 functionality)
        
        Args:
            raw_df: Raw uploaded dataframe
            
        Returns:
            Dictionary of processed dataframes by data type
        """
        st.header("🔧 Milestone 1: Data Preprocessing")
        
        if raw_df is None or raw_df.empty:
            st.error("No data available for preprocessing")
            return {}
        
        try:
            # Data cleaning and preparation
            processed_df = self._clean_data(raw_df)
            
            # Identify and separate different data types
            separated_data = self._separate_data_types(processed_df)
            
            # Generate preprocessing report
            self._generate_preprocessing_report(raw_df, separated_data)
            
            st.success("✅ Data preprocessing completed successfully!")
            return separated_data
            
        except Exception as e:
            st.error(f"❌ Preprocessing failed: {str(e)}")
            return {}
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the raw data"""
        st.write("**Step 1: Data Cleaning**")
        
        df_clean = df.copy()
        
        # Identify timestamp column
        timestamp_col = self._identify_timestamp_column(df_clean)
        if timestamp_col:
            df_clean[timestamp_col] = pd.to_datetime(df_clean[timestamp_col])
            df_clean = df_clean.rename(columns={timestamp_col: 'timestamp'})
        else:
            # Create synthetic timestamp if none exists
            df_clean['timestamp'] = pd.date_range(
                start='2024-01-01', 
                periods=len(df_clean), 
                freq='1min'
            )
            st.warning("No timestamp column found. Created synthetic timestamps.")
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                st.info(f"Filled missing values in {col} with median")
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        if len(df_clean) < initial_rows:
            st.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Sort by timestamp
        df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
        
        return df_clean
    
    def _identify_timestamp_column(self, df: pd.DataFrame) -> str:
        """Identify timestamp column in the dataframe"""
        timestamp_indicators = ['timestamp', 'time', 'date', 'datetime', 'created_at']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in timestamp_indicators):
                return col
            
            # Check if column contains datetime data
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(10))
                    return col
                except:
                    continue
        
        return None
    
    def _separate_data_types(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Separate data into different health metric types"""
        st.write("**Step 2: Identifying Health Metrics**")
        
        separated_data = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove timestamp from numeric columns
        if 'timestamp' in numeric_cols:
            numeric_cols.remove('timestamp')
        
        # Common health metric patterns
        metric_patterns = {
            'heart_rate': ['heart', 'hr', 'bpm', 'pulse'],
            'steps': ['step', 'walk', 'distance'],
            'calories': ['calorie', 'energy', 'burn'],
            'sleep': ['sleep', 'rest', 'bedtime'],
            'activity': ['activity', 'active', 'move']
        }
        
        # Assign columns to data types
        for col in numeric_cols:
            col_lower = col.lower()
            assigned = False
            
            for data_type, patterns in metric_patterns.items():
                if any(pattern in col_lower for pattern in patterns):
                    separated_data[data_type] = df[['timestamp', col]].rename(columns={col: 'value'})
                    st.success(f"✅ Identified {col} as {data_type}")
                    assigned = True
                    break
            
            if not assigned:
                # Assign to generic type
                data_type = f"metric_{len([k for k in separated_data.keys() if k.startswith('metric_')]) + 1}"
                separated_data[data_type] = df[['timestamp', col]].rename(columns={col: 'value'})
                st.info(f"📊 Assigned {col} to {data_type}")
        
        # If no columns identified, use all numeric columns
        if not separated_data and len(numeric_cols) > 0:
            st.warning("No specific health metrics identified. Using all numeric columns.")
            for i, col in enumerate(numeric_cols):
                separated_data[f'metric_{i+1}'] = df[['timestamp', col]].rename(columns={col: 'value'})
        
        return separated_data
    
    def _generate_preprocessing_report(self, raw_df: pd.DataFrame, processed_data: Dict):
        """Generate preprocessing report"""
        st.write("**Step 3: Preprocessing Report**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Original Rows", len(raw_df))
        
        with col2:
            st.metric("Processed Metrics", len(processed_data))
        
        with col3:
            total_processed = sum(len(df) for df in processed_data.values())
            st.metric("Total Data Points", total_processed)
        
        with col4:
            numeric_cols = len(raw_df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", numeric_cols)
        
        # Show processed data types
        with st.expander("📋 Processed Data Overview"):
            for data_type, df in processed_data.items():
                st.write(f"**{data_type.replace('_', ' ').title()}:** {len(df)} data points")
                st.dataframe(df.head(5), use_container_width=True)

# =============================================================================
# MILESTONE 2 CLASSES - ADDED BELOW
# =============================================================================

class FitPulseFeatureExtractor:
    """Extract time-series features for FitPulse health data"""
    
    def __init__(self, feature_complexity: str = 'efficient'):
        self.feature_complexity = feature_complexity
        self.feature_matrix = None
        self.feature_names = []
        self.extraction_report = {}
        
    def extract_fitpulse_features(self, df: pd.DataFrame, data_type: str, 
                                window_size: int = 60) -> Tuple[pd.DataFrame, Dict]:
        """
        Extract features from FitPulse health data
        """
        st.info(f"🔬 Extracting {data_type} features for FitPulse...")
        
        report = {
            'data_type': data_type,
            'original_rows': len(df),
            'window_size': window_size,
            'features_extracted': 0,
            'success': False
        }
        
        try:
            # Prepare FitPulse data for TSFresh
            df_prepared = self._prepare_fitpulse_data(df, data_type, window_size)
            
            if df_prepared is None or len(df_prepared) == 0:
                report['error'] = "No FitPulse data available for feature extraction"
                return pd.DataFrame(), report
            
            # Select feature parameters for health data
            fc_parameters = self._get_fitpulse_parameters()
            
            # Extract features
            feature_matrix = extract_features(
                df_prepared,
                column_id='window_id',
                column_sort='timestamp',
                default_fc_parameters=fc_parameters,
                disable_progressbar=False,
                n_jobs=1
            )
            
            # Handle missing values
            feature_matrix = impute(feature_matrix)
            
            # Remove constant features
            feature_matrix = self._remove_constant_features(feature_matrix)
            
            self.feature_matrix = feature_matrix
            self.feature_names = list(feature_matrix.columns)
            
            report['features_extracted'] = len(self.feature_names)
            report['feature_windows'] = len(feature_matrix)
            report['success'] = True
            
            st.success(f"✅ Extracted {report['features_extracted']} features from {data_type}")
            
            return feature_matrix, report
            
        except Exception as e:
            report['error'] = str(e)
            st.error(f"❌ FitPulse feature extraction failed: {str(e)}")
            return pd.DataFrame(), report
    
    def _prepare_fitpulse_data(self, df: pd.DataFrame, data_type: str, 
                             window_size: int) -> pd.DataFrame:
        """Prepare FitPulse data in TSFresh format"""
        
        # FitPulse metric columns mapping
        fitpulse_metrics = {
            'heart_rate': 'value',
            'steps': 'steps',
            'calories': 'calories',
            'sleep': 'sleep_duration',
            'activity': 'activity_level'
        }
        
        if data_type not in fitpulse_metrics:
            st.warning(f"Unknown FitPulse data type: {data_type}")
            return None
        
        metric_col = fitpulse_metrics[data_type]
        
        # Use 'value' as default if specific column not found
        if metric_col not in df.columns and 'value' in df.columns:
            metric_col = 'value'
        elif metric_col not in df.columns:
            # Try to find any numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:  # Skip timestamp
                metric_col = numeric_cols[1]  # Use first numeric column after timestamp
            else:
                st.warning(f"No suitable metric column found in FitPulse data")
                return None
        
        # Create rolling windows for FitPulse data
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Ensure we have enough data for windows
        if len(df_sorted) < window_size:
            st.warning(f"Not enough data points for window size {window_size}. Need at least {window_size} points, have {len(df_sorted)}")
            return None
        
        prepared_data = []
        window_id = 0
        
        # Create overlapping windows (50% overlap)
        step_size = max(1, window_size // 2)  # Ensure at least 1
        
        for i in range(0, len(df_sorted) - window_size + 1, step_size):
            window_data = df_sorted.iloc[i:i+window_size].copy()
            window_data['window_id'] = window_id
            prepared_data.append(window_data[['window_id', 'timestamp', metric_col]])
            window_id += 1
        
        if not prepared_data:
            return None
        
        df_prepared = pd.concat(prepared_data, ignore_index=True)
        df_prepared = df_prepared.rename(columns={metric_col: 'value'})
        
        return df_prepared
    
    def _get_fitpulse_parameters(self) -> Dict:
        """Get feature parameters optimized for health data"""
        return {
            # Basic statistical features
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "variance": None,
            "minimum": None,
            "maximum": None,
            "mean_abs_change": None,
            "mean_change": None,
            
            # Distribution features
            "skewness": None,
            "kurtosis": None,
            "quantile": [{"q": 0.25}, {"q": 0.5}, {"q": 0.75}],
            
            # Energy and change features
            "abs_energy": None,
            "absolute_sum_of_changes": None,
            
            # Pattern features
            "count_above_mean": None,
            "count_below_mean": None,
            "longest_strike_above_mean": None,
            "longest_strike_below_mean": None,
            
            # Trend features
            "linear_trend": [{"attr": "slope"}, {"attr": "intercept"}],
            
            # Autocorrelation features
            "autocorrelation": [{"lag": 1}, {"lag": 2}, {"lag": 5}],
            
            # Complexity features
            "approximate_entropy": [{"m": 2, "r": 0.2}],
            "cid_ce": [{"normalize": True}],
            
            # Additional valid health features
            "number_peaks": [{"n": 3}],
            "range_count": [{"min": -1, "max": 1}],
            "ratio_beyond_r_sigma": [{"r": 2}],
            
            # Time series characteristics
            "fft_aggregated": [{"aggtype": "centroid"}],
            "time_reversal_asymmetry_statistic": [{"lag": 1}],
        }
    
    def _remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with zero variance"""
        constant_features = [col for col in df.columns if df[col].std() == 0]
        if constant_features:
            st.info(f"Removed {len(constant_features)} constant features")
            df = df.drop(columns=constant_features)
        return df
    
    def get_health_insights(self, n_features: int = 15) -> pd.DataFrame:
        """Get top features with health insights"""
        if self.feature_matrix is None or self.feature_matrix.empty:
            return pd.DataFrame()
        
        # Calculate variance for each feature
        feature_variance = self.feature_matrix.var().sort_values(ascending=False)
        top_features = feature_variance.head(n_features)
        
        insights_df = pd.DataFrame({
            'Feature': top_features.index,
            'Variance': top_features.values,
            'Mean': [self.feature_matrix[feat].mean() for feat in top_features.index],
            'Std': [self.feature_matrix[feat].std() for feat in top_features.index],
            'Health_Relevance': self._categorize_health_relevance(top_features.index)
        })
        
        return insights_df
    
    def _categorize_health_relevance(self, features: List[str]) -> List[str]:
        """Categorize features by health relevance"""
        categories = []
        
        for feature in features:
            feature_lower = feature.lower()
            
            if any(term in feature_lower for term in ['trend', 'slope', 'linear']):
                categories.append("Trend Pattern")
            elif any(term in feature_lower for term in ['energy', 'entropy', 'complexity']):
                categories.append("Activity Level")
            elif any(term in feature_lower for term in ['autocorrelation', 'recurrence']):
                categories.append("Rhythm Pattern")
            elif any(term in feature_lower for term in ['quantile', 'skewness', 'kurtosis']):
                categories.append("Distribution Shape")
            elif any(term in feature_lower for term in ['mean', 'median', 'std']):
                categories.append("Central Tendency")
            elif any(term in feature_lower for term in ['peak', 'fft']):
                categories.append("Spectral Pattern")
            else:
                categories.append("General Pattern")
        
        return categories

class FitPulseTrendAnalyzer:
    """Analyze health trends using Prophet for FitPulse data"""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.health_anomalies = {}
        self.analysis_reports = {}
    
    def analyze_health_trends(self, df: pd.DataFrame, data_type: str,
                            forecast_hours: int = 24) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze health trends for FitPulse data
        """
        st.info(f"📈 Analyzing {data_type} trends for FitPulse...")
        
        report = {
            'data_type': data_type,
            'training_points': len(df),
            'forecast_hours': forecast_hours,
            'success': False
        }
        
        try:
            # Prepare data for Prophet
            prophet_df = self._prepare_prophet_data(df, data_type)
            
            if len(prophet_df) < 10:  # Minimum data points
                report['error'] = f"Insufficient {data_type} data for trend analysis"
                return pd.DataFrame(), report
            
            # Configure Prophet for health data
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,
                interval_width=0.90
            )
            
            # Add health-specific seasonality
            if data_type == 'heart_rate':
                model.add_seasonality(name='hourly', period=1, fourier_order=3)
            
            st.write(f"Training {data_type} trend model...")
            model.fit(prophet_df)
            
            # Create future dataframe
            future = model.make_future_dataframe(
                periods=forecast_hours * 60,
                freq='min',
                include_history=True
            )
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Detect health anomalies
            anomalies = self._detect_health_anomalies(prophet_df, forecast, data_type)
            
            # Store results
            self.models[data_type] = model
            self.forecasts[data_type] = forecast
            self.health_anomalies[data_type] = anomalies
            
            # Calculate health metrics
            report.update(self._calculate_health_metrics(prophet_df, forecast, data_type))
            report['anomalies_detected'] = len(anomalies)
            report['success'] = True
            
            self.analysis_reports[data_type] = report
            
            st.success(f"✅ {data_type} trend analysis complete")
            st.write(f"📊 Detected {len(anomalies)} potential health anomalies")
            
            return forecast, report
            
        except Exception as e:
            report['error'] = str(e)
            st.error(f"❌ {data_type} trend analysis failed: {str(e)}")
            return pd.DataFrame(), report
    
    def _prepare_prophet_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Prepare FitPulse data for Prophet"""
        # Use 'value' as default metric column
        if 'value' in df.columns:
            metric_col = 'value'
        else:
            # Find first numeric column after timestamp
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            metric_col = numeric_cols[0] if numeric_cols else 'value'
        
        prophet_df = pd.DataFrame({
            'ds': df['timestamp'],
            'y': df[metric_col]
        })
        
        return prophet_df.dropna()
    
    def _detect_health_anomalies(self, actual_df: pd.DataFrame, 
                               forecast_df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Detect health anomalies based on forecast residuals"""
        
        # Merge actual and forecast
        merged = actual_df.merge(
            forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
            on='ds', 
            how='left'
        )
        
        # Calculate residuals
        merged['residual'] = merged['y'] - merged['yhat']
        residual_std = merged['residual'].std()
        
        if residual_std == 0:  # Handle case with no variation
            return pd.DataFrame()
        
        merged['residual_std'] = np.abs(merged['residual']) / residual_std
        
        # Define anomaly thresholds based on data type
        thresholds = {
            'heart_rate': 2.5,
            'steps': 3.0,
            'calories': 3.0,
            'sleep': 2.8,
            'activity': 3.2
        }
        
        threshold = thresholds.get(data_type, 3.0)
        
        # Identify anomalies
        anomalies = merged[merged['residual_std'] > threshold].copy()
        
        if not anomalies.empty:
            anomalies['anomaly_score'] = anomalies['residual_std']
            anomalies['severity'] = self._classify_anomaly_severity(
                anomalies['residual_std'], data_type
            )
        
        return anomalies
    
    def _classify_anomaly_severity(self, residual_std: pd.Series, data_type: str) -> pd.Series:
        """Classify anomaly severity"""
        if data_type == 'heart_rate':
            return pd.cut(residual_std, bins=[0, 2.5, 3.5, float('inf')], 
                         labels=['Low', 'Medium', 'High'])
        else:
            return pd.cut(residual_std, bins=[0, 3.0, 4.5, float('inf')], 
                         labels=['Low', 'Medium', 'High'])
    
    def _calculate_health_metrics(self, actual_df: pd.DataFrame, 
                                forecast_df: pd.DataFrame, data_type: str) -> Dict:
        """Calculate health-specific metrics"""
        merged = actual_df.merge(
            forecast_df[['ds', 'yhat']], 
            on='ds', 
            how='left'
        )
        
        metrics = {
            'mae': np.mean(np.abs(merged['y'] - merged['yhat'])),
            'rmse': np.sqrt(np.mean((merged['y'] - merged['yhat'])**2)),
            'trend_stability': self._calculate_trend_stability(merged['yhat']),
            'data_volatility': merged['y'].std()
        }
        
        return metrics
    
    def _calculate_trend_stability(self, trend: pd.Series) -> float:
        """Calculate how stable the trend is"""
        changes = np.diff(trend)
        if len(changes) == 0:
            return 1.0
        return 1.0 / (1.0 + np.std(changes))

class FitPulseBehaviorClusterer:
    """Cluster health behaviors for FitPulse data"""
    
    def __init__(self):
        self.scalers = {}
        self.cluster_models = {}
        self.cluster_labels = {}
        self.cluster_insights = {}
    
    def cluster_health_behaviors(self, feature_matrix: pd.DataFrame, data_type: str,
                               method: str = 'kmeans', n_clusters: int = 4) -> Tuple[np.ndarray, Dict]:
        """
        Cluster health behavior patterns
        """
        st.info(f"👥 Clustering {data_type} health behaviors...")
        
        report = {
            'data_type': data_type,
            'method': method,
            'n_samples': len(feature_matrix),
            'n_features': len(feature_matrix.columns),
            'success': False
        }
        
        try:
            if feature_matrix.empty or len(feature_matrix) < n_clusters:
                report['error'] = "Not enough data for clustering"
                return np.array([]), report
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_matrix)
            self.scalers[data_type] = scaler
            
            # Apply clustering
            if method == 'kmeans':
                model = KMeans(n_clusters=min(n_clusters, len(feature_matrix)), 
                              random_state=42, n_init=10)
                labels = model.fit_predict(features_scaled)
                self.cluster_models[data_type] = model
                
            elif method == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=min(5, len(feature_matrix)//10))
                labels = model.fit_predict(features_scaled)
                self.cluster_models[data_type] = model
            else:
                report['error'] = f"Unknown clustering method: {method}"
                return np.array([]), report
            
            self.cluster_labels[data_type] = labels
            
            # Calculate clustering quality
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                try:
                    silhouette = silhouette_score(features_scaled, labels)
                    davies_bouldin = davies_bouldin_score(features_scaled, labels)
                    
                    report['silhouette_score'] = silhouette
                    report['davies_bouldin_score'] = davies_bouldin
                    report['clustering_quality'] = self._assess_clustering_quality(silhouette, davies_bouldin)
                except:
                    report['clustering_quality'] = "Cannot calculate metrics"
            
            report['n_clusters'] = len(unique_labels)
            report['cluster_distribution'] = {
                int(label): int(count) 
                for label, count in zip(*np.unique(labels, return_counts=True))
            }
            report['success'] = True
            
            # Generate health insights
            self.cluster_insights[data_type] = self._generate_health_insights(
                feature_matrix, labels, data_type
            )
            
            st.success(f"✅ Identified {report['n_clusters']} health behavior patterns")
            
            return labels, report
            
        except Exception as e:
            report['error'] = str(e)
            st.error(f"❌ Health behavior clustering failed: {str(e)}")
            return np.array([]), report
    
    def _assess_clustering_quality(self, silhouette: float, davies_bouldin: float) -> str:
        """Assess clustering quality for health data"""
        if silhouette > 0.6 and davies_bouldin < 0.8:
            return "Excellent"
        elif silhouette > 0.4 and davies_bouldin < 1.2:
            return "Good"
        elif silhouette > 0.2 and davies_bouldin < 2.0:
            return "Fair"
        else:
            return "Poor"
    
    def _generate_health_insights(self, features: pd.DataFrame, 
                                labels: np.ndarray, data_type: str) -> Dict:
        """Generate health insights from clusters"""
        insights = {}
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_data = features[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
                
            insights[f"cluster_{cluster_id}"] = {
                'size': len(cluster_data),
                'avg_activity_level': cluster_data.mean().mean(),
                'stability': cluster_data.std().mean(),
                'health_interpretation': self._interpret_health_cluster(cluster_data, data_type, cluster_id)
            }
        
        return insights
    
    def _interpret_health_cluster(self, cluster_data: pd.DataFrame, 
                                data_type: str, cluster_id: int) -> str:
        """Interpret health meaning of clusters"""
        if cluster_data.empty:
            return "Insufficient data"
        
        avg_values = cluster_data.mean()
        overall_avg = avg_values.mean()
        
        if data_type == 'heart_rate':
            if overall_avg < 60:
                return "Low resting heart rate pattern"
            elif overall_avg < 70:
                return "Normal heart rate pattern"
            elif overall_avg < 80:
                return "Elevated heart rate pattern"
            else:
                return "High heart rate pattern"
        
        elif data_type == 'steps':
            if overall_avg < 5000:
                return "Sedentary activity pattern"
            elif overall_avg < 10000:
                return "Moderate activity pattern"
            else:
                return "Active lifestyle pattern"
        
        else:
            return f"Cluster {cluster_id} behavior pattern"

class FitPulseMilestone2:
    """Complete Milestone 2 pipeline for FitPulse Health Analytics"""
    
    def __init__(self):
        self.feature_extractor = FitPulseFeatureExtractor()
        self.trend_analyzer = FitPulseTrendAnalyzer()
        self.behavior_clusterer = FitPulseBehaviorClusterer()
        self.results = {}
    
    def run_fitpulse_analysis(self, processed_data: Dict[str, pd.DataFrame],
                            window_size: int = 60,
                            forecast_hours: int = 24,
                            clustering_method: str = 'kmeans',
                            n_clusters: int = 4) -> Dict:
        """
        Run complete FitPulse health analysis pipeline
        """
        st.header("🏥 FitPulse Health Analytics - Milestone 2")
        st.markdown("**Advanced Health Pattern Recognition & Anomaly Detection**")
        
        results = {
            'health_features': {},
            'trend_forecasts': {},
            'behavior_clusters': {},
            'health_anomalies': {},
            'analysis_reports': {}
        }
        
        for data_type, df in processed_data.items():
            st.subheader(f"🔍 Analyzing {data_type.replace('_', ' ').title()}")
            
            # Step 1: Health Feature Extraction
            with st.expander(f"📊 Health Feature Extraction - {data_type}", expanded=True):
                features, feature_report = self.feature_extractor.extract_fitpulse_features(
                    df, data_type, window_size
                )
                
                if not features.empty:
                    results['health_features'][data_type] = features
                    results['analysis_reports'][f'{data_type}_features'] = feature_report
                    
                    # Show health insights
                    st.write("**Health Feature Insights:**")
                    health_insights = self.feature_extractor.get_health_insights(12)
                    st.dataframe(health_insights, use_container_width=True)
                else:
                    st.warning(f"No features extracted for {data_type}")
            
            # Step 2: Health Trend Analysis
            with st.expander(f"📈 Health Trend Analysis - {data_type}", expanded=True):
                forecast, trend_report = self.trend_analyzer.analyze_health_trends(
                    df, data_type, forecast_hours
                )
                
                if not forecast.empty:
                    results['trend_forecasts'][data_type] = forecast
                    results['analysis_reports'][f'{data_type}_trends'] = trend_report
                    
                    # Visualize health trends
                    self._visualize_health_trends(df, forecast, data_type)
                    
                    # Show detected anomalies
                    anomalies = self.trend_analyzer.health_anomalies.get(data_type, pd.DataFrame())
                    if not anomalies.empty:
                        st.warning(f"🚨 Detected {len(anomalies)} health anomalies in {data_type}")
                        self._show_health_anomalies(anomalies, data_type)
                    else:
                        st.success(f"✅ No anomalies detected in {data_type}")
                else:
                    st.warning(f"No trend analysis completed for {data_type}")
            
            # Step 3: Behavior Clustering
            with st.expander(f"👥 Health Behavior Clustering - {data_type}", expanded=True):
                if data_type in results['health_features']:
                    features = results['health_features'][data_type]
                    
                    clusters, cluster_report = self.behavior_clusterer.cluster_health_behaviors(
                        features, data_type, clustering_method, n_clusters
                    )
                    
                    if len(clusters) > 0:
                        results['behavior_clusters'][data_type] = clusters
                        results['analysis_reports'][f'{data_type}_clusters'] = cluster_report
                        
                        # Visualize health behaviors
                        self._visualize_health_behaviors(features, clusters, data_type)
                        
                        # Show health insights
                        self._show_behavior_insights(data_type)
                    else:
                        st.warning(f"No clusters identified for {data_type}")
                else:
                    st.warning(f"No features available for clustering {data_type}")
            
            st.markdown("---")
        
        # Generate comprehensive health report
        self._generate_fitpulse_health_report(results)
        
        self.results = results
        return results
    
    def _visualize_health_trends(self, df: pd.DataFrame, forecast: pd.DataFrame, data_type: str):
        """Visualize health trends with anomalies"""
        fig = go.Figure()
        
        # Actual values
        metric_col = 'value' if 'value' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[metric_col],
            mode='markers',
            name='Actual',
            marker=dict(size=4, color='blue', opacity=0.6)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(width=0),
            name='Confidence'
        ))
        
        # Anomalies
        anomalies = self.trend_analyzer.health_anomalies.get(data_type, pd.DataFrame())
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies['ds'],
                y=anomalies['y'],
                mode='markers',
                name='Anomalies',
                marker=dict(size=8, color='red', symbol='x')
            ))
        
        fig.update_layout(
            title=f"FitPulse {data_type.replace('_', ' ').title()} Analysis",
            xaxis_title="Time",
            yaxis_title=data_type.replace('_', ' ').title(),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_health_anomalies(self, anomalies: pd.DataFrame, data_type: str):
        """Display health anomalies with severity"""
        st.write("**Detected Health Anomalies:**")
        
        display_anomalies = anomalies[['ds', 'y', 'residual_std', 'severity']].copy()
        display_anomalies['ds'] = display_anomalies['ds'].dt.strftime('%Y-%m-%d %H:%M')
        display_anomalies = display_anomalies.rename(columns={
            'ds': 'Timestamp',
            'y': 'Value',
            'residual_std': 'Anomaly Score',
            'severity': 'Severity'
        })
        
        st.dataframe(display_anomalies.sort_values('Anomaly Score', ascending=False), 
                    use_container_width=True)
    
    def _visualize_health_behaviors(self, features: pd.DataFrame, clusters: np.ndarray, data_type: str):
        """Visualize health behavior clusters"""
        try:
            # Reduce dimensions
            pca = PCA(n_components=2, random_state=42)
            features_scaled = StandardScaler().fit_transform(features)
            features_reduced = pca.fit_transform(features_scaled)
            
            df_viz = pd.DataFrame({
                'PC1': features_reduced[:, 0],
                'PC2': features_reduced[:, 1],
                'Health Pattern': clusters.astype(str)
            })
            
            fig = px.scatter(
                df_viz,
                x='PC1',
                y='PC2',
                color='Health Pattern',
                title=f"FitPulse {data_type} Health Behavior Patterns",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_traces(marker=dict(size=10, opacity=0.7))
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not visualize clusters: {str(e)}")
    
    def _show_behavior_insights(self, data_type: str):
        """Display health behavior insights"""
        insights = self.behavior_clusterer.cluster_insights.get(data_type, {})
        
        if insights:
            st.write("**Health Behavior Insights:**")
            
            for cluster_id, insight in insights.items():
                with st.expander(f"Pattern {cluster_id}: {insight['health_interpretation']}"):
                    st.write(f"• Samples: {insight['size']}")
                    st.write(f"• Average Activity Level: {insight['avg_activity_level']:.2f}")
                    st.write(f"• Pattern Stability: {insight['stability']:.2f}")
    
    def _generate_fitpulse_health_report(self, results: Dict):
        """Generate comprehensive FitPulse health report"""
        st.header("📋 FitPulse Health Analytics Report")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Health Metrics Analyzed", len(results['health_features']))
        
        with col2:
            total_anomalies = sum(
                len(self.trend_analyzer.health_anomalies.get(dt, pd.DataFrame()))
                for dt in results['health_features'].keys()
            )
            st.metric("Health Anomalies Detected", total_anomalies)
        
        with col3:
            st.metric("Behavior Patterns Identified", len(results['behavior_clusters']))
        
        with col4:
            total_features = sum(
                len(features.columns) for features in results['health_features'].values()
            )
            st.metric("Health Features Extracted", total_features)
        
        # Health recommendations
        st.subheader("💡 Health Insights & Recommendations")
        
        for data_type in results['health_features'].keys():
            anomalies = self.trend_analyzer.health_anomalies.get(data_type, pd.DataFrame())
            
            if not anomalies.empty:
                st.warning(f"**{data_type.replace('_', ' ').title()}:** Review {len(anomalies)} detected anomalies for potential health concerns")
            else:
                st.success(f"**{data_type.replace('_', ' ').title()}:** Patterns appear normal and healthy")
        
        st.success("✅ FitPulse Health Analytics Complete!")
        
        st.info("""
        **Milestone 2 Deliverables:**
        - ✅ Health-specific feature extraction
        - ✅ Advanced trend analysis with anomaly detection
        - ✅ Health behavior pattern clustering
        - ✅ Comprehensive health insights and recommendations
        - ✅ Interactive visualizations for health monitoring
        """)

# =============================================================================
# COMPLETE PIPELINE CLASS - ADDED BELOW
# =============================================================================

class CompleteFitPulsePipeline:
    """Complete pipeline from raw data upload to Milestone 2 analysis"""
    
    def __init__(self):
        self.data_uploader = DataUploader()
        self.preprocessor = Milestone1Preprocessor()
        self.milestone2 = FitPulseMilestone2()  # NOW THIS IS DEFINED!
    
    def run_complete_pipeline(self):
        """Run complete pipeline from data upload to analysis"""
        st.title("🏥 FitPulse - Complete Health Analytics Pipeline")
        
        # Step 1: Data Upload
        st.header("📁 Step 1: Upload Raw Data")
        raw_df = self.data_uploader.upload_raw_data()
        
        if raw_df is not None:
            # Step 2: Milestone 1 Preprocessing
            if st.button("🔧 Run Milestone 1 Preprocessing", type="primary"):
                with st.spinner("Preprocessing data..."):
                    processed_data = self.preprocessor.preprocess_health_data(raw_df)
                    
                    if processed_data:
                        st.session_state.processed_data = processed_data
                        st.success("✅ Ready for Milestone 2 Analysis!")
            
            # Step 3: Milestone 2 Analysis
            if 'processed_data' in st.session_state:
                st.header("🔬 Step 2: Milestone 2 - Feature Extraction & Analysis")
                
                # Analysis configuration
                st.sidebar.header("⚙️ Analysis Configuration")
                
                window_size = st.sidebar.slider(
                    "Feature Window Size (minutes)",
                    min_value=30, max_value=120, value=60, step=10
                )
                
                forecast_hours = st.sidebar.slider(
                    "Trend Forecast (hours)",
                    min_value=6, max_value=48, value=24, step=6
                )
                
                clustering_method = st.sidebar.selectbox(
                    "Clustering Method",
                    options=['kmeans', 'dbscan']
                )
                
                n_clusters = st.sidebar.slider(
                    "Health Patterns to Identify",
                    min_value=2, max_value=6, value=4
                )
                
                # Run Milestone 2 analysis
                if st.button("🚀 Run Milestone 2 Analysis", type="primary"):
                    with st.spinner("Performing advanced health analytics..."):
                        results = self.milestone2.run_fitpulse_analysis(
                            processed_data=st.session_state.processed_data,
                            window_size=window_size,
                            forecast_hours=forecast_hours,
                            clustering_method=clustering_method,
                            n_clusters=n_clusters
                        )
                        
                        st.session_state.milestone2_results = results
                        st.balloons()

def create_fitpulse_sample_data() -> pd.DataFrame:
    """Create sample FitPulse health data for demonstration"""
    
    # Generate realistic health data
    timestamps = pd.date_range(
        start='2024-01-15 06:00:00', 
        end='2024-01-15 22:00:00', 
        freq='1min'
    )
    
    # Heart rate data
    base_hr = 65
    hr_data = []
    
    for i, ts in enumerate(timestamps):
        hour = ts.hour + ts.minute / 60
        
        # Simulate daily heart rate pattern
        if 6 <= hour < 8:   # Morning rise
            activity_factor = 1.1 + 0.1 * np.sin(hour * np.pi / 12)
        elif 8 <= hour < 12:  # Active morning
            activity_factor = 1.3 + 0.2 * np.random.random()
        elif 12 <= hour < 14:  # Lunch dip
            activity_factor = 1.1
        elif 14 <= hour < 18:  # Afternoon activity
            activity_factor = 1.4 + 0.1 * np.random.random()
        else:  # Evening rest
            activity_factor = 1.0
        
        noise = np.random.normal(0, 2)
        hr = base_hr * activity_factor + noise
        hr_data.append(max(50, min(120, hr)))
    
    # Steps data
    steps_data = []
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        if 8 <= hour < 9 or 17 <= hour < 18:  # Walking periods
            steps = np.random.poisson(100)
        elif 9 <= hour < 12 or 14 <= hour < 17:  # Active periods
            steps = np.random.poisson(50)
        else:  # Rest periods
            steps = np.random.poisson(5)
        steps_data.append(steps)
    
    # Create sample dataframe
    sample_df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate_bpm': hr_data,
        'step_count': steps_data,
        'calories_burned': [hr * 0.1 + steps * 0.05 for hr, steps in zip(hr_data, steps_data)],
        'activity_level': [min(100, (hr - 60) * 2 + steps * 0.1) for hr, steps in zip(hr_data, steps_data)]
    })
    
    return sample_df

def main():
    st.set_page_config(
        page_title="FitPulse - Complete Pipeline",
        page_icon="🏥",
        layout="wide"
    )
    
    # Initialize the complete pipeline
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = CompleteFitPulsePipeline()
    
    # Sidebar options
    st.sidebar.header("🎯 Pipeline Options")
    
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=False, 
                                         help="Use built-in sample data for demonstration")
    
    if use_sample_data:
        st.sidebar.info("Using sample FitPulse data")
        sample_df = create_fitpulse_sample_data()
        st.session_state.sample_data = sample_df
        
        # Show sample data info
        with st.sidebar.expander("Sample Data Info"):
            st.write(f"Rows: {len(sample_df)}")
            st.write(f"Columns: {list(sample_df.columns)}")
            st.write("Time range:", 
                    f"{sample_df['timestamp'].min().strftime('%H:%M')} to "
                    f"{sample_df['timestamp'].max().strftime('%H:%M')}")
    
    # Run the complete pipeline
    st.session_state.pipeline.run_complete_pipeline()
    
    # Quick start with sample data
    if use_sample_data and 'sample_data' in st.session_state:
        if st.sidebar.button("🚀 Quick Start with Sample Data"):
            with st.spinner("Processing sample data..."):
                # Run preprocessing on sample data
                processed_data = st.session_state.pipeline.preprocessor.preprocess_health_data(
                    st.session_state.sample_data
                )
                
                if processed_data:
                    st.session_state.processed_data = processed_data
                    st.success("✅ Sample data processed! Ready for analysis.")

if __name__ == "__main__":
    main()