import streamlit_app as st
import pandas as pd

# Title of the app
st.title("Fitness Data Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type="csv")


if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Show dataframe
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Simple chart
    st.line_chart(df['heart_rate'])
