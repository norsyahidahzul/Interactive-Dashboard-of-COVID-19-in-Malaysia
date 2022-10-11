import streamlit as st


def Overview_of_Cases():
    


    st.set_page_config(page_title="Overview of Cases", page_icon="📊")
    st.markdown("# Overview of Cases")
    st.sidebar.header("Overview of Cases")
    st.write(
    """This demo shows how to use `st.write` to visualize Pandas DataFrames.
    (Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)"""
    )

    st.write(
    """This page gives you an overview of COVID-19 cases in Malaysia during first,second and early of third waves based on general cases or state cases. 
    """
    )

    st.write(
    """**This page is under development. Please come again later.**
    """
    )
Overview_of_Cases()
