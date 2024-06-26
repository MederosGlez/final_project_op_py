import streamlit as st
from tools import handle_define, handle_evaluate, handle_gen, handle_load, handle_save_all
from constants import *


buildin_functions = {
    "gen": handle_gen,
    "load": handle_load,
    "evaluate": handle_evaluate,
    "save": handle_define,
    "save_all": handle_save_all
}


def handle_buildin_function(tokens):
    buildin_function = tokens[0]
    assert buildin_function in buildin_functions, "no conozco esta funcion"
    fun = buildin_functions[buildin_function]
    result = fun(*tokens[1:])
    return result

options = [
    ["load","Load"],
    ["evaluate","Evaluate"],
    ["gen","Generate"],
    ["save","Save"]
]

def display():
    st.title("Welcome user")

    # Text boxes for user input
    mensajito=""
    ok = False
    for _fun , _name in options:
        col1, col2 = st.columns([3,1])
        with col1:
            tmp = st.text_input(f"{_name}")
        with col2:
            st.write("\n")
            st.write("\n")
            if st.button(_name):
                try:
                    mensajito = buildin_functions[_fun](tmp)
                    ok = True
                except AssertionError as err:
                    ok = False
                    mensajito = err


    if st.button("Save all"):
        handle_save_all()

    if mensajito:
        mensajito=str(mensajito)
        if ok:
            for i in mensajito.split('\n'):
                st.success(i)
        else:
            st.warning(mensajito)
    else:
        st.warning("Escriba algo! no sea timido")

    if os.path.exists('figure.png'):
        # Si el archivo existe, eliminarlo
        st.image("figure.png", use_column_width=True)
    if os.path.exists('figure.gif'):
        # Si el archivo existe, eliminarlo
        st.image("figure.gif", use_column_width=True)
    
