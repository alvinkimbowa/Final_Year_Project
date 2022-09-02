# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:59:38 2022

@author: ALVIN
"""
#%%
import time  # to simulate a real time data, time loop
import utils as ut
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development

# %%
st.set_page_config(
    page_title="Power Output and Load Prediction",
    # page_icon="✅",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="auto"
)

# st.title("Streamlit is lit")


# %%
# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)       


st.markdown("<h1 style='text-align: center; color: white;'>Streamlit is not as lit</h1>", unsafe_allow_html=True)

#%%
# read csv from a github repo
load_data_path = "Data/test.csv"
pv_data_path = "Data/Site_3_01-2018_02-2018.csv"

# read csv from a URL
@st.experimental_memo
def get_load_data() -> pd.DataFrame:
    return ut.read_load_data(load_data_path)

@st.experimental_memo
def get_pv_data() -> pd.DataFrame:
    return ut.read_pv_data(pv_data_path)

load_df = get_load_data()
pv_df = get_pv_data()

st.write("Load data")
load_df

"PV data"
pv_df

"With some coloring"
st.dataframe(load_df.style.highlight_max(color='red',axis=0))
# st.table(load_df)

"Load consumption"
st.line_chart(load_df['energy_kWh'])

"PV Power Ouput"
st.line_chart(pv_df[['Pyranometer_1', 'Active_Power']])


"Sample map"
map_data = pd.DataFrame(
    np.random.randn(20, 2) / [100, 100] + [0.3326, 32.5686],
    columns=['lat', 'lon'])

st.map(map_data)


# Sliders
"Rate me on a scale of 1-10"
s1 = st.slider('Slider 1', 0,10)
if s1 < 5:
    st.write("Damn, you can't give me a ", s1)
elif s1 < 8:
    st.write("At least a ", s1)
else:
    st.write("You're damn right I'm a ", s1)
    

t1 = st.text_input(label="Please enter your name", key="name")


if st.checkbox("Say Hello"):
    st.write("Wassap, ", t1)
    st.write("Same thing, ", st.session_state.name)

if st.checkbox("Plot graph"):
    no_hrs = st.selectbox("Up to how many hours?", load_df.index)
    st.line_chart(load_df['energy_kWh'][:no_hrs])
    

# Use select box to choose from series
option = st.selectbox("Choose a number", load_df['energy_kWh'])

"You chose ", option

start = st.selectbox("Choose starting month", pv_df['Month'].unique())
end = st.selectbox("Choose ending month", pv_df['Month'].unique())

"Starting month: ", start
"Ending month: ", end


# %% Side bar
s2 = st.sidebar.slider("Choose a value", 0,20)
t2 = st.sidebar.selectbox("Choose", load_df['hour'])

#%% Columns
left_col, right_col = st.columns(2)

left_col.button("Press me")
right_col.button("Press me too")

# Or even better, call Streamlit functions inside a "with" block:
with right_col:
    chosen = st.radio(
        'Choose radio',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")
    

#%% Progress bar
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    latest_iteration.text(f'Iteration {i+1}')
    bar.progress(i+1)
    time.sleep(0.1)
    
"We are done "