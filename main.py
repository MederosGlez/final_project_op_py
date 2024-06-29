import streamlit as st
from display import display
from settings import settings
from const import *

def main():
     # Configurar el tema de la página en light
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
