# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="FitPulse — Data Pipeline", layout="wide")

# -----------------------
# Helper functions
# -----------------------
def generate_sample_data(rows=1000):
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    idx = pd.date_range(start, periods=rows, freq="T")
    heart = (60 + 10*np.sin(np.linspace(0, 12*np.pi, rows)) + np.random.normal(0, 3, rows)).astype(int)
    steps = np.maximum(0, (np.random.poisson(0.2, rows) * (np.random.rand(rows) > 0.85)).astype(int))
    sleep = ((idx.hour < 7) | (idx.hour > 12)).astype(int)
    df = pd.DataFrame({"timestamp": idx, "heart_rate": heart, "steps": steps, "sleep": sleep})
    df.loc[50:60, "heart_rate"] = np.nan
    df = pd.concat([df, df.iloc[10:13]])
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    return df

def load_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            return pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type. Use CSV or JSON.")
            return None
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

def ensure_timestamp(df):
    for c in df.columns:
        if c.lower() in ("timestamp", "time", "datetime", "date", "ts"):
            try:
                df[c] = pd.to_datetime(df[c])
                return df, c
            except:
                pass
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() > len(df)*0.5:
                df[c] = parsed
                return df, c
        except:
            pass
    return df, None

def data_quality_report(df):
    total = len(df)
    missing = df.isna().sum().to_frame("missing_count")
    missing["missing_pct"] = (missing["missing_count"]/total*100).round(2)
    dtypes = df.dtypes.astype(str).to_frame("dtype")
    duplicates = df.duplicated().sum()
    ranges = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            valid = df[c].dropna()
            if len(valid):
                ranges[c] = (float(valid.min()), float(valid.max()))
    dq = {"rows": total, "columns": len(df.columns), "duplicates": int(duplicates)}
    return missing, dtypes, ranges, dq

def simple_validation(df):
    issues = []
    if "heart_rate" in df.columns:
        hr = df["heart_rate"]
        invalid = hr[(hr.notna()) & ((hr < 30) | (hr > 20))]
        for idx, val in invalid.items():
            issues.append({"index": idx, "column": "heart_rate", "value": val, "issue": "heart_rate outside plausible range (30-220)"})
    if "steps" in df.columns:
        s = df["steps"]
        invalid = s[(s.notna()) & (s < 0)]
        for idx, val in invalid.items():
            issues.append({"index": idx, "column": "steps", "value": val, "issue": "negative steps"})
    return pd.DataFrame(issues)

def clip_outliers(series, z_thresh=4.0):
    if series.dropna().shape[0] < 3 or not pd.api.types.is_numeric_dtype(series):
        return series
    z = (series - series.mean()) / (series.std(ddof=0) + 1e-9)
    out = series.copy()
    out[np.abs(z) > z_thresh] = np.nan
    return out

def resample_and_fill(df, ts_col, freq, agg_map, fill_method):
    df = df.set_index(ts_col).sort_index()
    res = df.resample(freq).agg(agg_map)
    if fill_method == "ffill":
        res = res.ffill()
    elif fill_method == "bfill":
        res = res.bfill()
    elif fill_method == "interpolate":
        res = res.interpolate(limit_direction="both")
    elif fill_method == "zero":
        num_cols = res.select_dtypes("number").columns
        res[num_cols] = res[num_cols].fillna(0)
    return res.reset_index()

def zscore_anomaly(series, window=60, z_thresh=3.5):
    roll_mean = series.rolling(window, min_periods=1, center=True).mean()
    roll_std = series.rolling(window, min_periods=1, center=True).std(ddof=0).replace(0, np.nan)
    z = (series - roll_mean) / roll_std
    return np.abs(z) > z_thresh

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.title("Fitness Tracker")
uploaded_file = st.sidebar.file_uploader("Upload fitness CSV or JSON", type=["csv", "json"])
use_sample = st.sidebar.checkbox("Use sample dataset", value=False)
resample_freq = st.sidebar.selectbox("Resample frequency", ["None", "1T", "5T", "15T", "30T", "1H", "1D"])
fill_method = st.sidebar.selectbox("Fill / Impute method", ["interpolate", "ffill", "bfill", "zero"])
outlier_clip = st.sidebar.slider("Outlier", 2.0, 8.0, 4.0, 0.1)
drop_duplicates_opt = st.sidebar.checkbox("Drop duplicates", value=True)
drop_na_prop = st.sidebar.slider("Drop rows with > X% missing values", 0, 100, 90)

st.sidebar.markdown("---")
st.sidebar.markdown("**Visualization options**")
show_missing_heatmap = st.sidebar.checkbox("Show missing-data heatmap", value=True)
show_anomalies = st.sidebar.checkbox("Detect & show anomalies (z-score)", value=True)

# -----------------------
# Wait until file upload or sample selection
# -----------------------
if uploaded_file is None and not use_sample:
    st.info("Please upload a CSV/JSON file to continue, or check 'Use sample dataset' to try demo data.")
    st.stop()

# -----------------------
# Load data
# -----------------------
if uploaded_file is not None:
    raw_df = load_uploaded_file(uploaded_file)
    if raw_df is None:
        st.stop()
    st.success(f"Uploaded file: {uploaded_file.name}")
elif use_sample:
    raw_df = generate_sample_data()
    st.info("Sample dataset loaded.")

# -----------------------
# Pipeline logic continues...
# -----------------------
st.title("📊 FitPulse Anomaly Detection")
with st.expander("Raw data (first 200 rows)"):
    st.write(raw_df.head(200))

df, ts_col = ensure_timestamp(raw_df.copy())
if ts_col is None:
    st.warning("No timestamp column detected.")
    st.stop()
st.markdown(f"**Using timestamp column:** `{ts_col}`")

# -----------------------
# Data quality, validation, cleaning, resampling, visualization
# -----------------------
missing, dtypes, ranges, dq = data_quality_report(df)
st.subheader("Data Quality Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", dq["rows"])
c2.metric("Columns", dq["columns"])
c3.metric("Duplicates", dq["duplicates"])
c4.metric("Missing cells", f"{int(missing['missing_count'].sum())} ({missing['missing_pct'].sum().round(2)}%)")

col_left, col_right = st.columns([2,1])
with col_left:
    st.markdown("**Missing values by column**")
    st.dataframe(missing.sort_values("missing_count", ascending=False))
    st.markdown("**Data types**")
    st.dataframe(dtypes)
with col_right:
    st.markdown("**Numeric ranges**")
    st.write(ranges if ranges else "No numeric columns found.")

# Validation
st.subheader("Data Validation")
issues_df = simple_validation(df)
if issues_df.empty:
    st.success("No validation issues found.")
else:
    st.error(f"{len(issues_df)} issues found.")
    st.dataframe(issues_df)

# Cleaning
st.subheader("Cleaning & Missing-value handling")
clean_df = df.copy()
if drop_na_prop > 0:
    thresh = max(int((1 - drop_na_prop/100) * len(clean_df.columns)), 1)
    clean_df = clean_df.dropna(axis=0, thresh=thresh)
if drop_duplicates_opt:
    before = len(clean_df)
    clean_df = clean_df.drop_duplicates()
    st.info(f"Dropped {before - len(clean_df)} duplicate rows.")
for c in clean_df.columns:
    if pd.api.types.is_numeric_dtype(clean_df[c]) and c != ts_col:
        clean_df[c] = clip_outliers(clean_df[c], z_thresh=outlier_clip)
num_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
if num_cols:
    clean_df[num_cols] = clean_df[num_cols].interpolate(limit_direction="both")
st.write("Preview cleaned data (first 200 rows)")
st.dataframe(clean_df.head(200))

# Resampling
if resample_freq != "None":
    agg_map = {c: "mean" if pd.api.types.is_numeric_dtype(clean_df[c]) else "first" for c in clean_df.columns if c != ts_col}
    try:
        processed = resample_and_fill(clean_df, ts_col, resample_freq, agg_map, fill_method)
        st.success(f"Resampled to {resample_freq} with {fill_method}.")
    except:
        processed = clean_df.copy()
else:
    processed = clean_df.copy()

# Visualization
st.subheader("Visualizations")
plot_cols = [c for c in processed.columns if c != ts_col]
if plot_cols:
    sel_cols = st.multiselect("Select columns to plot", plot_cols, default=[c for c in plot_cols if pd.api.types.is_numeric_dtype(processed[c])][:2])
    if sel_cols:
        fig = go.Figure()
        for c in sel_cols:
            fig.add_trace(go.Scatter(x=processed[ts_col], y=processed[c], name=c, mode="lines"))
        fig.update_layout(height=420, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# Missing heatmap
if show_missing_heatmap:
    missmat = processed.isna().astype(int).T
    fig2 = px.imshow(missmat, labels=dict(x="row", y="column", color="is_missing"), aspect="auto")
    st.plotly_chart(fig2, use_container_width=True)

# Anomalies
if show_anomalies and len(num_cols):
    anomaly_col = st.selectbox("Column for anomaly detection", num_cols, index=0)
    window = st.slider("Anomaly rolling window", 5, 720, 60)
    zt = st.slider("Anomaly z-threshold", 2.0, 6.0, 3.5, 0.1)
    series = processed[anomaly_col].astype(float)
    mask = zscore_anomaly(series, window=window, z_thresh=zt)
    st.write(f"Detected {mask.sum()} anomalies in `{anomaly_col}`.")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=processed[ts_col], y=series, name=anomaly_col, mode="lines"))
    fig3.add_trace(go.Scatter(x=processed[ts_col][mask], y=series[mask], mode="markers", name="anomaly", marker=dict(size=8, symbol="x")))
    st.plotly_chart(fig3, use_container_width=True)

# Export cleaned data
st.subheader("Export cleaned dataset")
buf = processed.to_csv(index=False).encode('utf-8')
st.download_button(label="Download cleaned CSV", data=buf, file_name="fitpulse_cleaned.csv", mime="text/csv")

# Summary metrics
st.subheader("FitPulse Anomaly Detection")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Rows after processing", len(processed))
col_b.metric("Numeric columns", len(num_cols))
col_c.metric("Anomalies found", int(mask.sum()) if show_anomalies and len(num_cols) else 0)
