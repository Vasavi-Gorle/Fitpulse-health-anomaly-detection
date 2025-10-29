"""
two.py - Milestone 3: Anomaly Detection & Visualization
Single-file upload version (heart_rate, step_count, duration_minutes in one CSV).
Run: streamlit run two.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Optional
import json
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

# Try Prophet (optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


# ---------------------------
# Utility: aligned boolean mask
# ---------------------------
def bool_mask_for_column(df: pd.DataFrame, colname: str) -> pd.Series:
    if colname in df.columns:
        s = df[colname]
        s = s.reindex(df.index).fillna(False).astype(bool)
        return s
    else:
        return pd.Series(False, index=df.index, dtype=bool)


# ---------------------------
# SAMPLE DATA GENERATOR
# ---------------------------
def create_sample_data_with_anomalies():
    rng_hr = pd.date_range('2024-01-15 08:00:00', '2024-01-15 20:00:00', freq='1min')
    hr_vals = []
    for ts in rng_hr:
        tod = ts.hour + ts.minute / 60.0
        if 9 <= tod < 10:
            hr = 105 + np.random.normal(0, 4)
        elif 14 <= tod < 15:
            hr = 95 + np.random.normal(0, 4)
        else:
            hr = 70 + np.random.normal(0, 2.5)
        if 11.5 <= tod < 12:
            hr = 135 + np.random.normal(0, 3)
        if 16 <= tod < 16.3:
            hr = 35 + np.random.normal(0, 2)
        if 18.5 <= tod < 18.6:
            hr = 150
        hr_vals.append(float(max(20, min(220, hr))))
    heart_rate_df = pd.DataFrame({'timestamp': rng_hr, 'heart_rate': hr_vals})

    rng_steps = pd.date_range('2024-01-15 08:00:00', '2024-01-15 20:00:00', freq='5min')
    step_vals = []
    for ts in rng_steps:
        tod = ts.hour + ts.minute / 60.0
        if 8 <= tod < 9:
            s = 50 + np.random.randint(-10, 10)
        elif 12 <= tod < 13:
            s = 80 + np.random.randint(-15, 15)
        elif 17 <= tod < 18:
            s = 100 + np.random.randint(-20, 20)
        else:
            s = 20 + np.random.randint(-5, 5)
        if 15 <= tod < 15.2:
            s = 1200
        step_vals.append(int(max(0, s)))
    steps_df = pd.DataFrame({'timestamp': rng_steps, 'step_count': step_vals})

    sleep_dates = pd.date_range('2024-01-01', periods=30, freq='D')
    durations = []
    for _ in sleep_dates:
        dur_hours = 7 + np.random.normal(0, 1.0)
        if np.random.rand() < 0.05:
            dur_hours = 2.0
        if np.random.rand() < 0.03:
            dur_hours = 13.0
        durations.append(max(0.5, dur_hours * 60))
    sleep_df = pd.DataFrame({'timestamp': sleep_dates, 'duration_minutes': durations})

    return {'heart_rate': heart_rate_df, 'steps': steps_df, 'sleep': sleep_df}


# ---------------------------
# THRESHOLD DETECTOR
# ---------------------------
class ThresholdAnomalyDetector:
    def __init__(self):
        self.rules = {
            'heart_rate': {'col': 'heart_rate', 'min': 40, 'max': 120, 'sustained_min': 10},
            'steps': {'col': 'step_count', 'min': 0, 'max': 1000, 'sustained_min': 5},
            'sleep': {'col': 'duration_minutes', 'min': 180, 'max': 720, 'sustained_min': 0}
        }

    def detect(self, df: pd.DataFrame, data_type: str):
        report = {'method': 'threshold', 'data_type': data_type, 'anomalies_detected': 0}
        if data_type not in self.rules:
            return df, report
        rule = self.rules[data_type]
        col = rule['col']
        if col not in df.columns:
            return df, report

        out = df.copy().sort_values('timestamp').reset_index(drop=True)
        out['threshold_anomaly'] = False
        out['threshold_reason'] = ''
        out['threshold_severity'] = 'normal'

        high_mask = out[col] > rule['max']
        low_mask = out[col] < rule['min']

        if rule['sustained_min'] > 0:
            window = rule['sustained_min']
            high_sustained = high_mask.rolling(window=window, min_periods=window).sum() >= window
            low_sustained = low_mask.rolling(window=window, min_periods=window).sum() >= window
            out.loc[high_sustained.fillna(False), ['threshold_anomaly', 'threshold_reason', 'threshold_severity']] = \
                [True, f'{col} > {rule["max"]} (sustained)', 'high']
            out.loc[low_sustained.fillna(False), ['threshold_anomaly', 'threshold_reason', 'threshold_severity']] = \
                [True, f'{col} < {rule["min"]} (sustained)', 'medium']
        else:
            out.loc[high_mask, ['threshold_anomaly', 'threshold_reason', 'threshold_severity']] = \
                [True, f'{col} > {rule["max"]}', 'medium']
            out.loc[low_mask, ['threshold_anomaly', 'threshold_reason', 'threshold_severity']] = \
                [True, f'{col} < {rule["min"]}', 'high']

        report['anomalies_detected'] = int(out['threshold_anomaly'].sum())
        report['anomaly_percentage'] = float((out['threshold_anomaly'].sum() / max(1, len(out))) * 100)
        report['threshold'] = rule
        return out, report


# ---------------------------
# RESIDUAL (MODEL-BASED) DETECTOR
# ---------------------------
class ResidualAnomalyDetector:
    def __init__(self, threshold_std: float = 3.0):
        self.threshold_std = float(threshold_std)
        self.use_prophet = PROPHET_AVAILABLE

    def _prophet_forecast(self, df: pd.DataFrame, metric_col: str):
        tmp = df[['timestamp', metric_col]].rename(columns={'timestamp': 'ds', metric_col: 'y'}).copy()
        model = Prophet()
        model.fit(tmp)
        future = tmp[['ds']].rename(columns={'ds': 'ds'})
        forecast = model.predict(future)
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'timestamp', 'yhat': 'predicted'})
        return forecast

    def _stl_predict(self, df: pd.DataFrame, metric_col: str):
        tmp = df.set_index('timestamp').copy()
        try:
            ts = tmp[metric_col].asfreq('T')
        except Exception:
            ts = tmp[metric_col]
        ts = ts.interpolate().fillna(method='bfill').fillna(method='ffill')
        period = 1440 if len(ts) >= 1440 else max(3, int(len(ts) / 2))
        stl = STL(ts, period=period, robust=True)
        res = stl.fit()
        predicted = res.trend + res.seasonal
        residuals = ts.values - predicted.values
        sigma = float(np.nanstd(residuals)) if len(residuals) > 0 else 1.0
        out = pd.DataFrame({
            'timestamp': predicted.index,
            'predicted': predicted.values,
            'yhat_lower': predicted.values - 1.96 * sigma,
            'yhat_upper': predicted.values + 1.96 * sigma
        }).reset_index(drop=True)
        return out

    def detect(self, df: pd.DataFrame, data_type: str, metric_col: Optional[str] = None):
        metric_map = {'heart_rate': 'heart_rate', 'steps': 'step_count', 'sleep': 'duration_minutes'}
        if metric_col is None:
            metric_col = metric_map.get(data_type)
        report = {'method': 'residual', 'data_type': data_type, 'anomalies_detected': 0}
        if metric_col is None or metric_col not in df.columns:
            return df, report

        df_sorted = df.copy().sort_values('timestamp').reset_index(drop=True)
        forecast = None
        if self.use_prophet and PROPHET_AVAILABLE:
            try:
                forecast = self._prophet_forecast(df_sorted, metric_col)
            except Exception:
                forecast = self._stl_predict(df_sorted, metric_col)
        else:
            forecast = self._stl_predict(df_sorted, metric_col)

        merged = df_sorted.merge(forecast[['timestamp', 'predicted', 'yhat_lower', 'yhat_upper']],
                                 on='timestamp', how='left')
        merged['predicted'] = merged['predicted'].fillna(method='ffill').fillna(method='bfill')
        merged['yhat_lower'] = merged['yhat_lower'].fillna(merged['predicted'] - 1.0)
        merged['yhat_upper'] = merged['yhat_upper'].fillna(merged['predicted'] + 1.0)

        merged['residual'] = merged[metric_col] - merged['predicted']
        res_mean = merged['residual'].mean()
        res_std = merged['residual'].std(ddof=0) if merged['residual'].std(ddof=0) != 0 else 1.0
        threshold = self.threshold_std * res_std
        merged['residual_anomaly'] = np.abs(merged['residual'] - res_mean) > threshold
        outside_interval = (merged[metric_col] > merged['yhat_upper']) | (merged[metric_col] < merged['yhat_lower'])
        merged['residual_anomaly'] = merged['residual_anomaly'] | outside_interval
        merged['residual_reason'] = ''
        merged.loc[merged['residual_anomaly'], 'residual_reason'] = 'Deviates from predicted trend'

        report['anomalies_detected'] = int(merged['residual_anomaly'].sum())
        report['anomaly_percentage'] = float((merged['residual_anomaly'].sum() / max(1, len(merged))) * 100)
        report['residual_stats'] = {'mean': float(res_mean), 'std': float(res_std), 'threshold': float(threshold)}
        return merged, report


# ---------------------------
# CLUSTER-BASED DETECTOR
# ---------------------------
class ClusterAnomalyDetector:
    def __init__(self):
        pass

    def detect(self, feature_df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5):
        report = {'method': 'cluster', 'anomalies_detected': 0}
        if feature_df is None or feature_df.shape[0] == 0:
            return feature_df, report
        f = feature_df.fillna(0)
        scaler = StandardScaler()
        X = scaler.fit_transform(f.values)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        out = feature_df.copy().reset_index(drop=True)
        out['cluster'] = labels
        out['cluster_anomaly'] = out['cluster'] == -1
        report['anomalies_detected'] = int(out['cluster_anomaly'].sum())
        report['cluster_counts'] = dict(pd.Series(labels).value_counts().to_dict())
        return out, report


# ---------------------------
# VISUALIZER
# ---------------------------
class AnomalyVisualizer:
    def __init__(self):
        self.colors = {
            'normal': '#1f77b4',
            'threshold': '#ff7f0e',
            'residual': '#d62728',
            'cluster': '#9467bd'
        }

    def plot_heart_rate(self, df: pd.DataFrame, title: str = "Heart Rate Anomalies"):
        fig = go.Figure()
        threshold_mask = bool_mask_for_column(df, 'threshold_anomaly')
        residual_mask = bool_mask_for_column(df, 'residual_anomaly')
        baseline_mask = ~threshold_mask

        if not df.loc[baseline_mask].empty:
            fig.add_trace(go.Scatter(
                x=df.loc[baseline_mask, 'timestamp'],
                y=df.loc[baseline_mask, 'heart_rate'],
                mode='lines',
                name='Baseline',
                line=dict(color=self.colors['normal'])
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['heart_rate'],
                mode='lines',
                name='Heart Rate',
                line=dict(color=self.colors['normal'])
            ))

        if 'predicted' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['predicted'], mode='lines', name='Predicted',
                                     line=dict(color='green', dash='dash')))
            if 'yhat_upper' in df.columns and 'yhat_lower' in df.columns:
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['yhat_upper'], mode='lines', name='Upper', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['yhat_lower'], mode='lines', name='Lower', line=dict(width=0), fill='tonexty', fillcolor='rgba(144,238,144,0.2)', showlegend=False))

        if threshold_mask.any():
            th = df.loc[threshold_mask]
            fig.add_trace(go.Scatter(x=th['timestamp'], y=th['heart_rate'], mode='markers', name='Threshold',
                                     marker=dict(color=self.colors['threshold'], size=9, symbol='x')))

        if residual_mask.any():
            res = df.loc[residual_mask]
            fig.add_trace(go.Scatter(x=res['timestamp'], y=res['heart_rate'], mode='markers', name='Residual',
                                     marker=dict(color=self.colors['residual'], size=10, symbol='diamond')))

        fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Heart Rate (bpm)', height=600)
        st.plotly_chart(fig, use_container_width=True)

    def plot_steps(self, df: pd.DataFrame, title: str = "Steps Anomalies"):
        fig = go.Figure()
        threshold_mask = bool_mask_for_column(df, 'threshold_anomaly')
        baseline_mask = ~threshold_mask

        if not df.loc[baseline_mask].empty:
            fig.add_trace(go.Bar(x=df.loc[baseline_mask, 'timestamp'], y=df.loc[baseline_mask, 'step_count'], name='Steps'))
        else:
            fig.add_trace(go.Bar(x=df['timestamp'], y=df['step_count'], name='Steps'))

        if threshold_mask.any():
            th = df.loc[threshold_mask]
            fig.add_trace(go.Scatter(x=th['timestamp'], y=th['step_count'], mode='markers', name='Threshold Anomaly',
                                     marker=dict(color='red', size=12, symbol='star')))

        if 'predicted' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['predicted'], mode='lines', name='Predicted', line=dict(color='orange', dash='dot')))

        fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Step Count', height=500)
        st.plotly_chart(fig, use_container_width=True)

    def plot_sleep(self, df: pd.DataFrame, title: str = "Sleep Anomalies"):
        d = df.copy()
        if 'duration_minutes' in d.columns:
            d['duration_hours'] = d['duration_minutes'] / 60.0
        else:
            d['duration_hours'] = 0.0

        fig = go.Figure()
        threshold_mask = bool_mask_for_column(d, 'threshold_anomaly')
        baseline_mask = ~threshold_mask

        if not d.loc[baseline_mask].empty:
            fig.add_trace(go.Scatter(x=d.loc[baseline_mask, 'timestamp'], y=d.loc[baseline_mask, 'duration_hours'], mode='lines+markers', name='Sleep (hrs)'))
        else:
            fig.add_trace(go.Scatter(x=d['timestamp'], y=d['duration_hours'], mode='lines+markers', name='Sleep (hrs)'))

        if threshold_mask.any():
            an = d.loc[threshold_mask]
            fig.add_trace(go.Scatter(x=an['timestamp'], y=an['duration_hours'], mode='markers', name='Sleep Anomaly', marker=dict(color='red', size=10)))

        fig.add_hline(y=7, line_dash='dash', annotation_text='Recommended 7h', annotation_position='right')
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Hours', height=450)
        st.plotly_chart(fig, use_container_width=True)

    def mat_summary(self, reports: dict):
        labels = []
        counts = []
        for dtype, methods in reports.items():
            for method_key, rep in methods.items():
                labels.append(f"{dtype}-{method_key}")
                counts.append(rep.get('anomalies_detected', 0))
        if not labels:
            st.info("No anomaly reports to summarize.")
            return
        fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.3)))
        ax.barh(labels, counts)
        ax.set_title('Anomalies by data-method')
        ax.set_xlabel('Count')
        plt.tight_layout()
        st.pyplot(fig)


# ---------------------------
# PIPELINE
# ---------------------------
class AnomalyDetectionPipeline:
    def __init__(self):
        self.threshold_detector = ThresholdAnomalyDetector()
        self.residual_detector = ResidualAnomalyDetector()
        self.cluster_detector = ClusterAnomalyDetector()
        self.visualizer = AnomalyVisualizer()
        self.reports = {}
        self.processed = {}

    def run(self,
            preprocessed_data: Dict[str, pd.DataFrame],
            use_prophet: bool,
            residual_std: float,
            cluster_eps: float,
            cluster_min_samples: int):
        self.reports = {}
        self.processed = {}
        self.residual_detector.threshold_std = float(residual_std)
        self.residual_detector.use_prophet = bool(use_prophet and PROPHET_AVAILABLE)

        for dtype, df in preprocessed_data.items():
            st.subheader(f"🔍 Processing: {dtype}")
            self.reports[dtype] = {}

            df_thresh, rep_thresh = self.threshold_detector.detect(df, dtype)
            self.reports[dtype]['threshold'] = rep_thresh

            metric_map = {'heart_rate': 'heart_rate', 'steps': 'step_count', 'sleep': 'duration_minutes'}
            metric_col = metric_map.get(dtype)
            df_res, rep_res = self.residual_detector.detect(df_thresh, dtype, metric_col)
            self.reports[dtype]['residual'] = rep_res

            df_final = df_res.copy()

            if metric_col and metric_col in df_final.columns and len(df_final) >= max(10, cluster_min_samples):
                feat = pd.DataFrame()
                feat['value'] = df_final[metric_col].fillna(method='ffill').fillna(method='bfill')
                feat['roll_mean_5'] = feat['value'].rolling(window=5, min_periods=1).mean()
                feat['roll_std_5'] = feat['value'].rolling(window=5, min_periods=1).std().fillna(0)
                cluster_out, rep_cluster = self.cluster_detector.detect(feat, eps=cluster_eps, min_samples=cluster_min_samples)
                cluster_out = cluster_out.reset_index(drop=True)
                df_final = df_final.reset_index(drop=True)
                df_final = pd.concat([df_final, cluster_out[['cluster', 'cluster_anomaly']].reset_index(drop=True)], axis=1)
                self.reports[dtype]['cluster'] = rep_cluster
            else:
                self.reports[dtype]['cluster'] = {'method': 'cluster', 'anomalies_detected': 0}

            self.processed[dtype] = df_final

            if dtype == 'heart_rate':
                self.visualizer.plot_heart_rate(df_final)
            elif dtype == 'steps':
                self.visualizer.plot_steps(df_final)
            elif dtype == 'sleep':
                self.visualizer.plot_sleep(df_final)

        st.markdown("---")
        st.subheader("📊 Summary Dashboard")
        self.visualizer.mat_summary(self.reports)
        st.write("Detailed Reports (JSON):")
        st.json(self.reports)
        return {'reports': self.reports, 'processed': self.processed}


# ---------------------------
# Helper: parse single uploaded CSV into three datasets
# ---------------------------
def parse_single_file_to_dataframes(df_raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Accepts a DataFrame (from uploaded CSV) that has a 'timestamp' column and optionally:
    - heart_rate
    - step_count
    - duration_minutes
    Returns dict with keys present among 'heart_rate','steps','sleep' mapped to DataFrames.
    Performs resampling/aggregation for steps if necessary.
    """
    df = df_raw.copy()
    if 'timestamp' not in df.columns:
        raise ValueError("Uploaded CSV must contain a 'timestamp' column.")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    outputs = {}

    # HEART RATE: take rows with heart_rate column
    if 'heart_rate' in df.columns:
        hr = df[['timestamp', 'heart_rate']].dropna(subset=['heart_rate']).copy()
        # Ideally heart_rate should be minute frequency; we won't force-resample, we pass as-is.
        outputs['heart_rate'] = hr

    # STEPS: if step_count exists, try to ensure 5-min aggregation
    if 'step_count' in df.columns:
        stp = df[['timestamp', 'step_count']].dropna(subset=['step_count']).copy()
        # set index and check freq — if data is finer than 5-min, aggregate to 5-min sums
        stp = stp.set_index('timestamp')
        # If there are duplicate timestamps or very fine frequency, resample to 5-min and sum
        try:
            inferred = pd.infer_freq(stp.index)
        except Exception:
            inferred = None
        # If freq is None or freq finer than 5T, resample to 5T sums
        if inferred is None:
            # just resample to 5T with sum (this will group sparse timestamps into 5-min bins)
            stp_5 = stp.resample('5T').sum().reset_index()
            stp_5 = stp_5[stp_5['step_count'].notna()]
            outputs['steps'] = stp_5
        else:
            # If freq is minute or smaller, resample; else keep as-is
            if inferred.endswith('T') and int(inferred[:-1]) <= 5:
                stp_5 = stp.resample('5T').sum().reset_index()
                stp_5 = stp_5[stp_5['step_count'].notna()]
                outputs['steps'] = stp_5
            else:
                outputs['steps'] = stp.reset_index().rename(columns={'index': 'timestamp'})

    # SLEEP: if duration_minutes exists (assumed daily), group by date if multiple entries
    if 'duration_minutes' in df.columns:
        sl = df[['timestamp', 'duration_minutes']].dropna(subset=['duration_minutes']).copy()
        # If multiple rows per day, take sum or mean — here we take sum (total sleep minutes)
        sl['date'] = sl['timestamp'].dt.normalize()
        sl_agg = sl.groupby('date', as_index=False)['duration_minutes'].sum()
        sl_agg = sl_agg.rename(columns={'date': 'timestamp'})
        outputs['sleep'] = sl_agg

    return outputs


# ---------------------------
# STREAMLIT APP
# ---------------------------
def main():
    st.set_page_config(page_title="Milestone 3 - Anomaly Detection", layout='wide', page_icon='🚨')
    st.title("🚨 Milestone 3 - Anomaly Detection & Visualization")
    st.markdown("Upload a single CSV containing `timestamp` and any of: `heart_rate`, `step_count`, `duration_minutes`.")

    st.sidebar.header("Configuration")
    use_sample = st.sidebar.checkbox("Use Sample Data (synthetic anomalies)", value=False)
    use_prophet = st.sidebar.checkbox("Use Prophet (if installed)", value=PROPHET_AVAILABLE)
    if use_prophet and not PROPHET_AVAILABLE:
        st.sidebar.warning("Prophet not installed — STL fallback will be used.")
    residual_std = st.sidebar.slider("Residual threshold (std deviations)", 1.0, 5.0, 3.0, 0.5)
    cluster_eps = st.sidebar.number_input("DBSCAN eps", value=0.5, step=0.1)
    cluster_min_samples = st.sidebar.number_input("DBSCAN min_samples", value=5, step=1, min_value=1)
    st.sidebar.markdown("---")
    st.sidebar.info("Upload one CSV that includes timestamp plus the metric columns you have.")

    pipeline = AnomalyDetectionPipeline()
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = pipeline

    data = None
    if use_sample:
        data = create_sample_data_with_anomalies()
    else:
        uploaded_file = st.file_uploader("Upload single CSV ", type=['csv'])
        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file)
                parsed = parse_single_file_to_dataframes = None
                try:
                    parsed = parse_single_file_to_dataframes = None  # placeholder to avoid linter issues
                except Exception:
                    pass
                # Use the helper to parse
                parsed = parse_single_file_to_dataframes if False else None  # dummy to avoid undefined error
                # actually call the function
                parsed = parse_single_file_to_dataframes_fn = None  # dummy
            except Exception as e:
                st.error(f"Failed reading uploaded CSV: {e}")
                df_raw = None

            if 'df_raw' in locals() and df_raw is not None:
                try:
                    parsed_outputs = parse_single_file_to_dataframes(df_raw)
                    # If parsed_outputs empty, warn and fallback to sample
                    if not parsed_outputs:
                        st.warning("Uploaded file doesn't contain heart_rate, step_count, or duration_minutes. Using sample data.")
                        data = create_sample_data_with_anomalies()
                    else:
                        data = parsed_outputs
                        st.success(f"Parsed columns: {', '.join([k for k in data.keys()])}")
                except Exception as e:
                    st.error(f"Error parsing uploaded file: {e}")
                    data = create_sample_data_with_anomalies()
        else:
            st.info("No upload yet. Toggle 'Use Sample Data' or upload a CSV.")
            # default nothing; user must upload or choose sample
    # If still None, use sample to ensure pipeline runs
    if data is None:
        data = create_sample_data_with_anomalies()

    # Run pipeline button
    if st.button("Run Milestone 3 Pipeline 🚀"):
        with st.spinner("Running anomaly detection pipeline..."):
            results = st.session_state.pipeline.run(
                preprocessed_data=data,
                use_prophet=use_prophet,
                residual_std=residual_std,
                cluster_eps=float(cluster_eps),
                cluster_min_samples=int(cluster_min_samples)
            )
            st.success("Pipeline complete.")
            st.session_state.milestone3_results = results

    # Export area
    if 'milestone3_results' in st.session_state:
        st.markdown("---")
        st.subheader("Export Results")
        results = st.session_state.milestone3_results
        json_report = json.dumps(results['reports'], default=str, indent=2)
        st.download_button("Download Reports (JSON)", data=json_report,
                           file_name=f"m3_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                           mime="application/json")

        all_anoms = []
        for dtype, df in results['processed'].items():
            df2 = df.copy()
            anomaly_cols = [c for c in df2.columns if 'anomaly' in c.lower()]
            if anomaly_cols:
                mask = df2[anomaly_cols].any(axis=1)
                anom = df2[mask].copy()
                if not anom.empty:
                    anom['data_type'] = dtype
                    all_anoms.append(anom)
        if all_anoms:
            combined = pd.concat(all_anoms, ignore_index=True)
            csv_bytes = combined.to_csv(index=False).encode('utf-8')
            st.download_button("Download Anomalies (CSV)", data=csv_bytes,
                               file_name=f"m3_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv")
        else:
            st.info("No anomalies found (or none to export).")


if __name__ == "__main__":
    main()
