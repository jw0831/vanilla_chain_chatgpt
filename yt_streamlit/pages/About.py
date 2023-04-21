import streamlit as st
import pandas as pd

st.title("DB 정보")
df = pd.read_csv('./db/sample_youtube_db.csv') # 경로는 main.py를 기준으로 함
st.dataframe(df)


st.title("README")
with open("markdown/about.md", "r") as f:
    data = f.read()

st.write(data, unsafe_allow_html=True)
