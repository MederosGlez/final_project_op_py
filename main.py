import streamlit as st
from display import display
from settings import settings
from constants import *
import sys

def main():
    def Display():
        display()
    def Settings():
        settings()
    st.sidebar.title('Menu')
    page=st.sidebar.selectbox('selecciona',['Display','Settings'])

    if page=='Display':
        Display()
    else:
        Settings()

if __name__ == "__main__":
    main()
