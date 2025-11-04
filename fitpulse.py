# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import warnings

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from prophet import Prophet

warnings.filterwarnings('ignore')

# === Page config ===
st.set_page_config(
    page_title="FitPulse — Fitness Anomaly Detection",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Demo auth (local only) ===
CREDENTIALS = {"Vasavi": "Infosys@123"}

def authenticate():
    """
    Simple demo auth that does NOT call st.experimental_rerun().
    Streamlit will automatically rerun when widgets change.
    """
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user = None

    # Already logged in
    if st.session_state.logged_in:
        st.sidebar.success(f"Signed in as: {st.session_state.user}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            return False, None
        return True, st.session_state.user

    # Sidebar data
    st.sidebar.title("💖FitPulse")
    
    # Login form
    st.sidebar.subheader("🔒 Sign in (demo)")
    user = st.sidebar.text_input("Username")
    pwd = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if user in CREDENTIALS and CREDENTIALS[user] == pwd:
            st.session_state.logged_in = True
            st.session_state.user = user
            return True, user
        else:
            st.sidebar.error("Invalid credentials")
            return False, None

    st.sidebar.info("Demo credentials")
    return False, None

logged_in, current_user = authenticate()
if not logged_in:
    st.title("💖FitPulse — Please sign in")
    st.markdown("This demo app requires a simple sign-in. Use the sidebar to authenticate.")
    st.stop()
    

# === Analyzer class ===
class FitnessDataAnalyzer:
    def __init__(self):
        self.data = None
        self.cleaned_data = None
        self.features = None
        self.anomalies = None
        self.prophet_results = {}
        self.cluster_results = None

    def load_data(self, file, file_type):
        try:
            if file_type == "CSV":
                self.data = pd.read_csv(file)
            else:
                self.data = pd.read_json(file)
            return True, f"Data loaded successfully! Shape: {self.data.shape}"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"

    def clean_and_preprocess(self, timestamp_col, value_cols, resample_freq='1H'):
        try:
            if self.data is None:
                return False, "No data loaded"

            df = self.data.copy()

            # Parse timestamps robustly
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            df = df.dropna(subset=[timestamp_col])
            df = df.sort_values(timestamp_col)
            df.set_index(timestamp_col, inplace=True)

            # Ensure numeric and interpolate selected columns
            for col in value_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

            freq_map = {'1H': '1H', '30min': '30min', '1D': '1D', '1W': '1W'}
            freq = freq_map.get(resample_freq, '1H')

            resampled = {}
            for col in value_cols:
                if col in df.columns:
                    resampled[col] = df[col].resample(freq).mean()

            if not resampled:
                return False, "No valid numeric columns found for resampling."

            self.cleaned_data = pd.DataFrame(resampled)
            if self.cleaned_data.empty:
                return False, "Resampled data is empty. Check columns and timestamps."

            self.cleaned_data = self.cleaned_data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

            # ensure index has a name (helps with Prophet)
            if self.cleaned_data.index.name is None:
                self.cleaned_data.index.name = 'timestamp'

            return True, "Data cleaned and preprocessed successfully!"
        except Exception as e:
            return False, f"Error in preprocessing: {str(e)}"

    def extract_features_basic(self, column_name):
        try:
            if self.cleaned_data is None or column_name not in self.cleaned_data.columns:
                return False, "No cleaned data available or column not found"

            data_series = self.cleaned_data[column_name].dropna()
            features = {
                'mean': float(data_series.mean()),
                'std': float(data_series.std()),
                'min': float(data_series.min()),
                'max': float(data_series.max()),
                'median': float(data_series.median()),
                'variance': float(data_series.var()),
                'skewness': float(data_series.skew()),
                'kurtosis': float(data_series.kurtosis()),
                'rolling_mean_24': float(data_series.rolling(window=24, min_periods=1).mean().iloc[-1]),
                'rolling_std_24': float(data_series.rolling(window=24, min_periods=1).std().iloc[-1]),
                'trend_slope': float(np.polyfit(np.arange(len(data_series)), data_series.values, 1)[0]) if len(data_series) > 1 else 0.0
            }

            self.features = pd.DataFrame([features])
            return True, f"Extracted {len(features)} basic features"
        except Exception as e:
            return False, f"Error in feature extraction: {str(e)}"

    def prophet_analysis(self, column_name):
        try:
            if self.cleaned_data is None or column_name not in self.cleaned_data.columns:
                return False, "No cleaned data available or column not found", None, None

            prophet_df = self.cleaned_data[[column_name]].reset_index()
            prophet_df.columns = ['ds', 'y']
            prophet_df = prophet_df.dropna()

            if len(prophet_df) < 10:
                return False, "Not enough data points for Prophet analysis (need >=10)", None, None

            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, changepoint_prior_scale=0.05)
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=0, freq='H')
            forecast = model.predict(future)

            results = prophet_df.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
            results['residual'] = results['y'] - results['yhat']
            residual_std = results['residual'].std() if not results['residual'].empty else 0.0
            results['anomaly_prophet'] = np.abs(results['residual']) > 2 * residual_std

            self.prophet_results[column_name] = {'results': results, 'forecast': forecast, 'model': model}
            return True, "Prophet analysis completed", results, forecast
        except Exception as e:
            return False, f"Error in Prophet analysis: {str(e)}", None, None

    def behavioral_clustering(self, n_clusters=3):
        try:
            if self.cleaned_data is None:
                return False, "No cleaned data available", None, None, None, None

            cluster_data = self.cleaned_data.copy().dropna()
            if cluster_data.empty or len(cluster_data) < n_clusters:
                return False, "Not enough data points for clustering", None, None, None, None

            scaler = StandardScaler()
            scaled = scaler.fit_transform(cluster_data)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(scaled)

            dbscan = DBSCAN(eps=0.5, min_samples=5)
            outlier_labels = dbscan.fit_predict(scaled)

            iso = IsolationForest(contamination=0.1, random_state=42)
            iso_labels = iso.fit_predict(scaled)

            results = cluster_data.copy()
            results['cluster_kmeans'] = labels
            results['cluster_dbscan'] = outlier_labels
            results['is_outlier_dbscan'] = outlier_labels == -1
            results['is_outlier_iso'] = iso_labels == -1

            n_comp = min(3, scaled.shape[1])
            if n_comp >= 2:
                pca = PCA(n_components=n_comp)
                pca_res = pca.fit_transform(scaled)
                results['pca1'] = pca_res[:, 0]
                results['pca2'] = pca_res[:, 1]
                results['pca3'] = pca_res[:, 2] if n_comp == 3 else 0.0
            else:
                results['pca1'] = scaled[:, 0]
                results['pca2'] = 0.0
                results['pca3'] = 0.0
                pca = None

            self.cluster_results = results
            return True, "Clustering completed", results, kmeans, dbscan, pca
        except Exception as e:
            return False, f"Error in clustering: {str(e)}", None, None, None, None

    def detect_anomalies(self, column_name, threshold_std=2.0):
        try:
            if self.cleaned_data is None:
                return False, "No cleaned data available"

            anomalies_data = self.cleaned_data.copy()

            # threshold method
            if column_name in anomalies_data.columns:
                mean_val = anomalies_data[column_name].mean()
                std_val = anomalies_data[column_name].std()
                upper = mean_val + threshold_std * std_val
                lower = mean_val - threshold_std * std_val
                anomalies_data['anomaly_threshold'] = ((anomalies_data[column_name] > upper) | (anomalies_data[column_name] < lower))
            else:
                anomalies_data['anomaly_threshold'] = False

            # prophet
            if column_name in self.prophet_results:
                pr = self.prophet_results[column_name]['results'].set_index('ds')
                pa = pr['anomaly_prophet']
                anomalies_data = anomalies_data.merge(pa.rename('anomaly_prophet'), left_index=True, right_index=True, how='left')
                anomalies_data['anomaly_prophet'] = anomalies_data['anomaly_prophet'].fillna(False)
            else:
                anomalies_data['anomaly_prophet'] = False

            # clustering
            if self.cluster_results is not None:
                flags = self.cluster_results[['is_outlier_dbscan', 'is_outlier_iso']].any(axis=1)
                anomalies_data = anomalies_data.merge(flags.rename('is_outlier'), left_index=True, right_index=True, how='left')
                anomalies_data['is_outlier'] = anomalies_data['is_outlier'].fillna(False)
            else:
                anomalies_data['is_outlier'] = False

            anomalies_data['anomaly_score'] = anomalies_data[['anomaly_threshold', 'anomaly_prophet', 'is_outlier']].sum(axis=1).astype(int)
            anomalies_data['final_anomaly'] = anomalies_data['anomaly_score'] >= 2

            self.anomalies = anomalies_data
            return True, "Anomaly detection completed successfully!"
        except Exception as e:
            return False, f"Error in anomaly detection: {str(e)}"

    def create_visualizations(self, column_name):
        try:
            if self.cleaned_data is None or column_name not in self.cleaned_data.columns:
                return None, None, None, None

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=self.cleaned_data.index, y=self.cleaned_data[column_name], mode='lines', name=column_name))

            if self.anomalies is not None and 'final_anomaly' in self.anomalies.columns:
                anomaly_points = self.anomalies[self.anomalies['final_anomaly']]
                if not anomaly_points.empty:
                    fig_ts.add_trace(go.Scatter(x=anomaly_points.index, y=anomaly_points[column_name], mode='markers', name='Anomalies',
                                                marker=dict(color='red', size=8, symbol='x')))

            fig_ts.update_layout(title=f"{column_name} — Time Series with Anomalies", xaxis_title='Time', yaxis_title=column_name, height=400)

            fig_dist = px.histogram(self.cleaned_data, x=column_name, nbins=50, title=f'Distribution of {column_name}')
            fig_dist.update_layout(height=400)

            if self.anomalies is not None and 'final_anomaly' in self.anomalies.columns:
                breakdown = pd.DataFrame({
                    'Method': ['Threshold', 'Prophet', 'Clustering', 'Final'],
                    'Count': [
                        int(self.anomalies['anomaly_threshold'].sum()) if 'anomaly_threshold' in self.anomalies.columns else 0,
                        int(self.anomalies['anomaly_prophet'].sum()) if 'anomaly_prophet' in self.anomalies.columns else 0,
                        int(self.anomalies['is_outlier'].sum()) if 'is_outlier' in self.anomalies.columns else 0,
                        int(self.anomalies['final_anomaly'].sum()) if 'final_anomaly' in self.anomalies.columns else 0
                    ]
                })
                fig_break = px.bar(breakdown, x='Method', y='Count', title='Anomaly Breakdown')
                fig_break.update_layout(height=400)
            else:
                fig_break = go.Figure()
                fig_break.add_annotation(text="No anomaly data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                fig_break.update_layout(height=400, title="Anomaly Breakdown")

            return fig_ts, fig_dist, fig_break, None
        except Exception as e:
            st.error(f"Error creating visualizations: {str(e)}")
            return None, None, None, None

# === sample data utility ===
def create_sample_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    np.random.seed(42)
    heart_rate = 65 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + np.random.normal(0, 3, len(dates))
    steps = np.maximum(0, np.random.poisson(50, len(dates)) + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24))
    sleep_hours = np.clip(np.random.normal(7, 1, len(dates)), 4, 10)
    # anomalies
    heart_rate[100] = 120
    heart_rate[200] = 45
    steps[150] = 300
    sleep_hours[250] = 12
    steps[50] = 0
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'heart_rate': np.round(heart_rate, 1),
        'steps': steps.astype(int),
        'sleep_hours': np.round(sleep_hours, 1)
    })
    return sample_data

# === UI stepper ===
if 'step' not in st.session_state:
    st.session_state.step = 0
steps = ["Data Upload & Preprocessing", "Feature Extraction & Modeling", "Anomaly Detection & Visualization", "Dashboard & Reports"]

def step_next():
    st.session_state.step = min(st.session_state.step + 1, len(steps)-1)

def step_prev():
    st.session_state.step = max(st.session_state.step - 1, 0)

st.title("💖 FitPulse — Fitness Health Anomaly Detection")
st.caption("This application analyzes fitness tracker data (heart rate, steps, sleep) to detect anomalies and identify behavioral patterns using advanced time-series analysis and machine learning.")

col_nav1, col_nav2, col_nav3 = st.columns([1, 6, 1])
with col_nav1:
    if st.button("◀ Prev"):
        step_prev()
with col_nav3:
    if st.button("Next ▶"):
        step_next()

st.sidebar.markdown("### Steps")
selected_step = st.sidebar.radio("Navigate steps", steps, index=st.session_state.step, key="side_step")
st.session_state.step = steps.index(selected_step)

if 'analyzer' not in st.session_state:
    st.session_state.analyzer = FitnessDataAnalyzer()
analyzer = st.session_state.analyzer

# Step 0: Data upload & preprocessing
if st.session_state.step == 0:
    st.header("📊 Data Upload & Preprocessing")
    left, right = st.columns(2)

    with left:
        st.subheader("Upload or load sample data")
        file_type = st.radio("Select file type:", ["CSV", "JSON"])
        uploaded_file = st.file_uploader(f"Upload {file_type} file", type=['csv', 'json'])
        if st.button("Load Sample Data"):
            analyzer.data = create_sample_data()
            st.success("Sample data loaded to memory")

        if uploaded_file is not None:
            ok, msg = analyzer.load_data(uploaded_file, file_type)
            if ok:
                st.success(msg)
                st.dataframe(analyzer.data.head(10))
                st.write("Columns:", list(analyzer.data.columns))
            else:
                st.error(msg)

    with right:
        st.subheader("Preprocessing")
        if analyzer.data is None:
            st.info("Upload data or load sample data first.")
        else:
            timestamp_candidates = [c for c in analyzer.data.columns if any(k in c.lower() for k in ['time', 'date', 'timestamp'])]
            default_ts = timestamp_candidates[0] if timestamp_candidates else analyzer.data.columns[0]
            timestamp_col = st.selectbox("Timestamp column", analyzer.data.columns, index=list(analyzer.data.columns).index(default_ts))
            numeric_cols = analyzer.data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                for c in analyzer.data.columns:
                    try:
                        pd.to_numeric(analyzer.data[c])
                        numeric_cols.append(c)
                    except Exception:
                        pass

            value_cols = st.multiselect("Metrics to analyze", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
            resample_freq = st.selectbox("Resample frequency", ['1H', '30min', '1D', '1W'], index=0)
            if st.button("Preprocess Data"):
                if not value_cols:
                    st.warning("Select at least one numeric metric")
                else:
                    with st.spinner("Preprocessing..."):
                        ok, msg = analyzer.clean_and_preprocess(timestamp_col, value_cols, resample_freq)
                        if ok:
                            st.success(msg)
                            st.subheader("Cleaned data preview (index = timestamp)")
                            preview = analyzer.cleaned_data.reset_index()
                            if 'timestamp' not in preview.columns:
                                preview = preview.rename(columns={preview.columns[0]: 'timestamp'})
                            st.dataframe(preview.head(10))
                            st.subheader("Basic stats")
                            st.dataframe(analyzer.cleaned_data.describe().round(3))
                            col1, col2 = st.columns(2)
                            with col1:
                                sel = st.selectbox("Select metric for quick plot", analyzer.cleaned_data.columns, key='quick_sel')
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=analyzer.cleaned_data.index, y=analyzer.cleaned_data[sel], mode='lines'))
                                fig.update_layout(title=f"{sel} over time", height=350)
                                st.plotly_chart(fig, use_container_width=True)
                            with col2:
                                fighist = px.histogram(analyzer.cleaned_data, x=analyzer.cleaned_data.columns[0], nbins=30)
                                fighist.update_layout(height=350)
                                st.plotly_chart(fighist, use_container_width=True)
                        else:
                            st.error(msg)

# Step 1: Features & modeling
elif st.session_state.step == 1:
    st.header("🔍 Feature Extraction & Modeling")
    if analyzer.cleaned_data is None:
        st.warning("Run preprocessing first (Step 0).")
    else:
        left, right = st.columns([2, 2])
        with left:
            st.subheader("Feature Extraction")
            selected_column = st.selectbox("Column for features", analyzer.cleaned_data.columns, key='feat_col')
            if st.button("Extract Basic Features"):
                ok, msg = analyzer.extract_features_basic(selected_column)
                if ok:
                    st.success(msg)
                    st.table(analyzer.features.T.rename(columns={0: 'Value'}).round(4))
                else:
                    st.error(msg)

        with right:
            st.subheader("Prophet Trend Analysis")
            prophet_column = st.selectbox("Column for Prophet", analyzer.cleaned_data.columns, key='prophet_col')
            if st.button("Run Prophet Analysis"):
                with st.spinner("Running Prophet..."):
                    ok, msg, results, forecast = analyzer.prophet_analysis(prophet_column)
                    if ok:
                        st.success(msg)
                        if results is not None and forecast is not None:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=results['ds'], y=results['y'], mode='lines', name='Actual'))
                            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted', line=dict(dash='dash')))
                            if 'yhat_upper' in forecast.columns and 'yhat_lower' in forecast.columns:
                                fig.add_trace(go.Scatter(
                                    x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                                    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                                    fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,255,255,0)'), showlegend=True, name='CI'
                                ))
                            fig.update_layout(title=f"Prophet Forecast - {prophet_column}", height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            if 'anomaly_prophet' in results.columns:
                                anoms = results[results['anomaly_prophet']]
                                st.markdown(f"**Prophet anomalies detected:** {len(anoms)}")
                                if not anoms.empty:
                                    st.dataframe(anoms[['ds', 'y', 'residual']].head(10))
                    else:
                        st.error(msg)

        st.subheader("Behavioral Pattern Clustering")
        n_clusters = st.slider("Number of clusters:", 2, 6, 3)
        if st.button("Perform Clustering"):
            with st.spinner("Clustering..."):
                ok, msg, results, kmeans, dbscan, pca = analyzer.behavioral_clustering(n_clusters)
                if ok:
                    st.success(msg)
                    plot_df = results.reset_index(drop=True).copy()
                    plot_df['timestamp'] = results.index.values
                    if 'pca1' in plot_df.columns and 'pca2' in plot_df.columns:
                        fig2d = px.scatter(plot_df, x='pca1', y='pca2', color='cluster_kmeans',
                                           title='KMeans Clustering (PCA 2D)',
                                           hover_data=['timestamp'] + analyzer.cleaned_data.columns.tolist())
                        st.plotly_chart(fig2d, use_container_width=True)

                    if 'pca3' in results.columns:
                        fig3d = go.Figure()
                        fig3d.add_trace(go.Scatter3d(
                            x=results['pca1'], y=results['pca2'], z=results['pca3'],
                            mode='markers',
                            marker=dict(size=4, color=results['cluster_kmeans'], colorscale='Viridis', showscale=True),
                            text=[str(ts) for ts in results.index]
                        ))
                        fig3d.update_layout(title='KMeans Clustering (PCA 3D)', height=600)
                        st.plotly_chart(fig3d, use_container_width=True)

                    out_counts = {'DBSCAN outliers': int(results['is_outlier_dbscan'].sum()), 'IsolationForest outliers': int(results['is_outlier_iso'].sum())}
                    st.bar_chart(pd.Series(out_counts))

                    st.subheader("Cluster statistics (by mean & std)")
                    try:
                        cluster_stats = results.reset_index().groupby('cluster_kmeans').agg({col: ['mean', 'std'] for col in analyzer.cleaned_data.columns})
                        st.dataframe(cluster_stats.round(3))
                    except Exception:
                        st.info("Could not compute cluster stats (check data).")
                else:
                    st.error(msg)

# Step 2: Anomaly detection & viz
elif st.session_state.step == 2:
    st.header("🚨 Anomaly Detection & Visualization")
    if analyzer.cleaned_data is None:
        st.warning("Run preprocessing first.")
    else:
        left, right = st.columns([2,1])
        with left:
            st.subheader("Settings")
            anomaly_column = st.selectbox("Select column for anomaly detection", analyzer.cleaned_data.columns, key='anomaly_col')
            threshold_std = st.slider("Threshold (std devs)", 1.0, 3.0, 2.0, 0.1)

            if st.button("Run Complete Analysis"):
                with st.spinner("Running Prophet + Clustering + Anomaly detection..."):
                    okp, msgp, _, _ = analyzer.prophet_analysis(anomaly_column)
                    if okp:
                        st.success("Prophet analysis done")
                    else:
                        st.warning(msgp)
                    okc, msgc, _, _, _, _ = analyzer.behavioral_clustering()
                    if okc:
                        st.success("Clustering done")
                    else:
                        st.info(msgc)
                    oka, msga = analyzer.detect_anomalies(anomaly_column, threshold_std)
                    if oka:
                        st.success("Anomaly detection done")
                    else:
                        st.error(msga)

            if st.button("Detect Anomalies Only"):
                with st.spinner("Detecting anomalies..."):
                    oka, msga = analyzer.detect_anomalies(anomaly_column, threshold_std)
                    if oka:
                        st.success(msga)
                    else:
                        st.error(msga)

        with right:
            st.subheader("Summary")
            if analyzer.anomalies is not None and 'final_anomaly' in analyzer.anomalies.columns:
                total_points = len(analyzer.anomalies)
                anomaly_count = int(analyzer.anomalies['final_anomaly'].sum())
                anomaly_percentage = (anomaly_count / total_points) * 100 if total_points > 0 else 0.0
                st.metric("Total points", total_points)
                st.metric("Anomalies", anomaly_count)
                st.metric("Anomaly %", f"{anomaly_percentage:.2f}%")
            else:
                st.info("Run anomaly detection to see summary.")

        st.subheader("Visualizations")
        if analyzer.cleaned_data is not None:
            fig_ts, fig_dist, fig_break, _ = analyzer.create_visualizations(anomaly_column)
            if fig_ts is not None:
                st.plotly_chart(fig_ts, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                if fig_dist is not None:
                    st.plotly_chart(fig_dist, use_container_width=True)
            with col2:
                if fig_break is not None:
                    st.plotly_chart(fig_break, use_container_width=True)

            if analyzer.anomalies is not None and 'final_anomaly' in analyzer.anomalies.columns:
                timeline = analyzer.anomalies['final_anomaly'].astype(int).resample('1D').sum()
                fig_tl = px.line(x=timeline.index, y=timeline.values, title="Daily Anomaly Count", labels={'x': 'Date', 'y': 'Anomaly Count'})
                st.plotly_chart(fig_tl, use_container_width=True)

# Step 3: Dashboard & Reports
elif st.session_state.step == 3:
    st.header("📈 Dashboard & Reports")
    if analyzer.cleaned_data is None:
        st.warning("Run preprocessing first.")
    else:
        st.subheader("Overall Summary")
        cols = st.columns(4)
        cols[0].metric("Total records", len(analyzer.cleaned_data))
        cols[1].metric("Metrics tracked", len(analyzer.cleaned_data.columns))
        date_range = analyzer.cleaned_data.index.max() - analyzer.cleaned_data.index.min()
        cols[2].metric("Data duration", f"{date_range.days} days")
        if analyzer.anomalies is not None and 'final_anomaly' in analyzer.anomalies.columns:
            cols[3].metric("Total anomalies", int(analyzer.anomalies['final_anomaly'].sum()))
        else:
            cols[3].metric("Total anomalies", "N/A")

        st.subheader("Metric Analysis")
        selected_metric = st.selectbox("Select metric", analyzer.cleaned_data.columns, key='dashboard_metric')

        left, right = st.columns(2)
        with left:
            fig_rolling = go.Figure()
            fig_rolling.add_trace(go.Scatter(x=analyzer.cleaned_data.index, y=analyzer.cleaned_data[selected_metric], mode='lines', name=selected_metric))
            rolling_avg = analyzer.cleaned_data[selected_metric].rolling(window=24, min_periods=1).mean()
            fig_rolling.add_trace(go.Scatter(x=analyzer.cleaned_data.index, y=rolling_avg, mode='lines', name='24h rolling'))
            fig_rolling.update_layout(title=f"{selected_metric} with rolling avg", height=400)
            st.plotly_chart(fig_rolling, use_container_width=True)

            if hasattr(analyzer.cleaned_data.index, 'hour'):
                hourly_avg = analyzer.cleaned_data.groupby(analyzer.cleaned_data.index.hour)[selected_metric].mean()
                fig_hour = px.bar(x=hourly_avg.index, y=hourly_avg.values, labels={'x': 'Hour', 'y': selected_metric}, title=f"{selected_metric} avg by hour")
                st.plotly_chart(fig_hour, use_container_width=True)

        with right:
            fig_box = px.box(analyzer.cleaned_data, y=selected_metric, title=f"{selected_metric} distribution")
            st.plotly_chart(fig_box, use_container_width=True)
            if len(analyzer.cleaned_data.columns) > 1:
                corr = analyzer.cleaned_data.corr()
                fig_corr = px.imshow(corr, title="Correlation heatmap", aspect='auto')
                st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Generate Report")
        if st.button("Generate Comprehensive Report"):
            with st.spinner("Preparing report..."):
                report = {
                    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data_shape": analyzer.cleaned_data.shape,
                    "data_columns": list(analyzer.cleaned_data.columns),
                    "date_range": {"start": analyzer.cleaned_data.index.min().strftime("%Y-%m-%d"), "end": analyzer.cleaned_data.index.max().strftime("%Y-%m-%d")},
                    "summary_statistics": analyzer.cleaned_data.describe().round(3).to_dict()
                }
                if analyzer.anomalies is not None and 'final_anomaly' in analyzer.anomalies.columns:
                    total = int(analyzer.anomalies['final_anomaly'].sum())
                    pct = f"{(total / len(analyzer.anomalies) * 100):.2f}%"
                    report["anomaly_summary"] = {
                        "total_anomalies": total,
                        "anomaly_percentage": pct,
                        "breakdown": {
                            "threshold": int(analyzer.anomalies['anomaly_threshold'].sum()) if 'anomaly_threshold' in analyzer.anomalies.columns else 0,
                            "prophet": int(analyzer.anomalies['anomaly_prophet'].sum()) if 'anomaly_prophet' in analyzer.anomalies.columns else 0,
                            "clustering": int(analyzer.anomalies['is_outlier'].sum()) if 'is_outlier' in analyzer.anomalies.columns else 0
                        }
                    }
                report_json = json.dumps(report, indent=2)
                st.download_button("Download Report (JSON)", data=report_json, file_name=f"fitpulse_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
                st.success("Report ready!")
                with st.expander("Report preview"):
                    st.json(report)
                    
                
