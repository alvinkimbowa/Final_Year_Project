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

from PIL import Image
from pv_model import *
from load_model import *
#%%
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.markdown("<h1 style='text-align: center;'>Solar PV Minigrid System</h1>", unsafe_allow_html=True)
        st.subheader("Please Login")
        st.text_input(
            "Enter password to continue", type="password", on_change=password_entered, key="password"
        )
        st.markdown("<p style='text-align:center; margin-top:3em;'>Final Year Project by: <i color:#021691>Alvin B. Kimbowa</i> and <i color:#021691>Moreen Tumwekwatse</i></p>", unsafe_allow_html=True)
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True
#%%
if check_password():
    # %%
    st.set_page_config(
        page_title="Power Output and Load Prediction",
        # page_icon="✅",
        page_icon=":shark:",
        layout="wide",
    )
    
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
    
    st.markdown("<h1 style='text-align: center;'>Solar PV Minigrid System</h1>", unsafe_allow_html=True)
    
    
    #%%
    # read csv from a github repo
    # load_data_path = "Data/test.csv"
    # pv_data_path = "Data/Site_3_01-2018_02-2018.csv"
    
    # read csv from a URL
    # @st.experimental_memo
    # def get_load_data() -> pd.DataFrame:
    #     return ut.read_load_data(load_data_path)
    
    # @st.experimental_memo
    # def get_pv_data() -> pd.DataFrame:
    #     return ut.read_pv_data(pv_data_path)
    
    # load_df = get_load_data()
    # pv_df = get_pv_data()
    
    @st.experimental_memo
    def get_pv_pred() -> pd.DataFrame:
        return pv_pred()
    
    @st.experimental_memo
    def get_load_pred() -> pd.DataFrame:
        return load_pred()
    
    
    # Get predictions
    # PV power output
    pv_predictions = get_pv_pred().reset_index(drop=True)
    # pv_plot_data = pv_predictions.copy()
    # pv_plot_data[:] = None
    # pv_plot_data['Prediction'][0] = pv_predictions['Prediction'][1]
    
    # Load prediction
    load_predictions = get_load_pred().reset_index(drop=True)
    # load_plot_data = load_predictions.copy()
    # load_plot_data[:] = None
    # load_plot_data['Prediction'][0] = load_predictions['Prediction'][1]
    
    
    # pred_load[:10] = None
    # st.line_chart(pred_load)
    
    #%% Columns
    st.subheader("Upload Weather Data")
    uploaded_file = st.file_uploader('Upload a CSV file containing the required data')
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            #st.write(df)
        except:
            st.warning('Please enter a correct document', icon="⚠️")
            
    st.button("Submit")
    st.subheader("Make Predictions")
    
    left_col, right_col = st.columns(2)
    
    # Or even better, call Streamlit functions inside a "with" block:
    with left_col:
        run_pv_model = st.button("Predict Power Output")
        st.write("PV Active Power Ouput")
        pv_chart = st.line_chart(pv_predictions[:2])
        st.write("PV Power Output Data")
        st.dataframe(pv_predictions, height=240)
            
    with right_col:
        run_load_model = st.button("Predict load")
        st.write("Load consumption")
        load_chart = st.line_chart(load_predictions[:2])
        st.write("Load Data Table")
        st.dataframe(load_predictions, height=240)
    
    
    #%% MATLAB Implementation
    st.write("Simulation of the Utility 2.0 PV Mini-grid")
    image = Image.open('deployment_original/streamlit/fyp/media/utility_20_1.jpg')
    st.image(image, caption='Setup of the Utility 2.0 PV Minigrid in MATLAB')
    
    #%% Site Location
    st.subheader("Site Location - Utility 2.0 PV Minigrid")
    st.map(pd.DataFrame({'lat':[0.5107592951044497], 'lon':[32.68729167113441]}))
    
    #%% Update charts
    # if run_pv_model:
    for i in range(len(pv_predictions)):
        # pv_plot_data['Prediction'][i] = pv_predictions['Prediction'][i]
        # pv_chart.add_rows(pv_plot_data[i:i+1])
        pv_chart.add_rows(pv_predictions[i:i+1])
        # pv_chart.line_chart(pred_pv[i:i+1])
        # pv_chart.line_chart(actual_pv[i:i+1])
        time.sleep(.25)
        
    # if run_load_model:
    for i in range(len(load_predictions)):
        # load_plot_data['Prediction'][i] = load_predictions['Prediction'][i]
        # load_chart.add_rows(load_plot_data[i:i+1])
        load_chart.add_rows(load_predictions[i:i+1])
        time.sleep(0.25)