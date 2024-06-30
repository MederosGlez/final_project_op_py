import streamlit as st
from tools import handle_define, handle_evaluate, handle_gen, handle_load
from constants import *


buildin_functions = {
    "gen": handle_gen,
    "load": handle_load,
    "evaluate": handle_evaluate,
    "save": handle_define
}


def handle_buildin_function(tokens):
    buildin_function = tokens[0]
    assert buildin_function in buildin_functions, "no conozco esta funcion"
    fun = buildin_functions[buildin_function]
    result = fun(*tokens[1:])
    return result

coso = [
    ["gen","Generate"],
    ["load","Load"],
    ["evaluate","Evaluate"],
    ["save","Save"]
]

def display():
    st.title("Welcome user")

    # Text boxes for user input
    mensajito=""
    for _fun , _name in coso:
        col1, col2 = st.columns(2)
        with col1:
            tmp = st.text_input(f"{_name}")
        with col2:
            st.write("\n")
            st.write("\n")
            if st.button(_name):
                mensajito = buildin_functions[_fun](tmp)
                

    if mensajito:
        st.success(mensajito)
    else:
        st.warning("Failed to load data. Please check your data loading logic.")

    if os.path.exists('figure.png'):
        # Si el archivo existe, eliminarlo
        st.image("figure.png", use_column_width=True)
    if os.path.exists('figure.gif'):
        # Si el archivo existe, eliminarlo
        st.image("figure.gif", use_column_width=True)
    
