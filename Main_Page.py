import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

def run():
  st.set_page_config(
    page_title="Main Page",
    page_icon="🦠")
	
  st.write("# Welcome to Interactive Dashboard of COVID-19 in Malaysia! 👋")

  st.sidebar.success("select a page above.")

  st.markdown(
    """
    This interactive dashboard of COVID-19 in Malaysia is made for research purpose of Master's degree. This dashboard
    contains COVID-19 simulator, COVID-19 predictor and Overview of COVID-19 cases in Malaysia. It is mainly
    builded for researcher, analyst and government agency to help making a proper decision upon the spike or subside of COVID-19 cases.
	
    👈 **Click the pages on the sidebar** to explore this interactive dashboard!
    Thank you.
    """
     )

if __name__ == "__main__":
    run()
