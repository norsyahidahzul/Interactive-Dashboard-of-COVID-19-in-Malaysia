import streamlit as st
import pandas as pd
import numpy as np

def Overview_of_Cases():
    selected_data = st.sidebar.selectbox('Choose data to display', ['General','Kuala Lumpur','Selangor','Perak','Negeri Sembilan','Johor','Sabah', 'Sarawak','Terengganu',
    'Pahang','Kelantan','Perlis', 'Kedah', 'Melaka','Pulau Pinang'])
    
    url= 'https://github.com/norsyahidahzul/Interactive-Dashboard-of-COVID-19-in-Malaysia/blob/f8ea7a4c347b003511537439c8c0f2ed6f1415b6/Data_Covid19_MalaysiaGeneral.xlsx'
    df = pd.read_excel (url) 
    st.write(df)
    
    maps = pd.DataFrame(np.random.randn(800,2)
    / [50,50] + [46.34, -108.7], columns =['latitude','longitude'])
    st.map (maps)

    
    
    
st.set_page_config(page_title="Overview of Cases 📊", page_icon="📊")
st.markdown("# Overview of Cases")
st.sidebar.header("Overview of Cases")   
st.write(
    """This page gives you an overview of COVID-19 cases in Malaysia during first, second and early of third waves based on general cases or state cases. 
    """
    ) 
st.write(
    """**This page is under development. Please come again later.**
    """
    )
    
Overview_of_Cases()

