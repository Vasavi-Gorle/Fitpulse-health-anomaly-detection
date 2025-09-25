import streamlit as st
import pandas as pd
import numpy as np

st.title("Fitness Data Visualization")

num = st.slider("pick number",1,10)

df = pd.DataFrame(np.random.randn(10,2),coloums=["A","B"])
st.line_chart(df)

st.write(f"you picked number:{num}")
