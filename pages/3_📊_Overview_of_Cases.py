import streamlit as st
import pandas as pd
import numpy as np
import xlrd

def Overview_of_Cases():
    selected_data = st.sidebar.selectbox('Choose data to display', ['General','Kuala Lumpur','Selangor','Perak','Negeri Sembilan','Johor','Sabah', 'Sarawak','Terengganu',
    'Pahang','Kelantan','Perlis', 'Kedah', 'Melaka','Pulau Pinang'])
    
    
    
    #url= "https://github.com/norsyahidahzul/Interactive-Dashboard-of-COVID-19-in-Malaysia/blob/f133ca7db8ab6514c075233e9730a55730bc83a1/Data_Covid19_MalaysiaGeneral.csv"
    #df = pd.read_csv(url)
    
    url= "https://github.com/norsyahidahzul/Interactive-Dashboard-of-COVID-19-in-Malaysia/blob/fabb4ae7fe7347d398ea896b14db89324c93ecdb/Data_Covid19_MalaysiaGeneral.xlsx"
    
    #url= r'C:\Users\USER\OneDrive - International Islamic University Malaysia\Desktop\SyahidZul\Master2021\Data Collection\Data_Covid19_MalaysiaGeneral.xlsx'
    #df = pd.read_excel(url)
   
    #sheet_url = "https://docs.google.com/spreadsheets/d/1Fx7f6rM5Ce331F9ipsEMn-xRjUKYiR3R_v9IDBusUUY/edit#gid=182521220"
    #url_1 = sheet_url.replace('/edit#gid=', '/export?format=xlsx&gid=')
    
    #sheet_url = "https://docs.google.com/spreadsheets/d/1-v1HV1nzqkNgmjJJxB3JMS10jEyZM-5f/edit#gid=23292093"
    #url= sheet_url.replace('/edit#gid=', '/export?format=xlsx&gid=')
    df=pd.read_excel(url,engine='xlrd')
    st.write(df)
    
    maps_general = pd.DataFrame(np.random.randn(800,2)/[2,3]+[4.2105, 101.9758], columns =['latitude','longitude'])

    st.map (maps_general, use_container_width=True)

    
    
    
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


