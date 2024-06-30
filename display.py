import streamlit as st
import sys

import multiprocessing
import re
from tools import handle_define, handle_evaluate, handle_gen
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import os
from constants import *

def handle_load(path, *args):
    print(path)
    with open(file=path) as file:
        for ix, line in enumerate(file.readlines()):
            if "=" in line:
                handle_define(line)
            else:
                result = handle_evaluate(line)
                print(f'line{ix}:{result}')


buildin_functions = {
    "gen": handle_gen,
    "load": handle_load
}


def handle_buildin_function(tokens):
    buildin_function = tokens[0]
    assert buildin_function in buildin_functions, "no conozco esta funcion"
    fun = buildin_functions[buildin_function]
    result = fun(*tokens[1:])
    return result


def process_line(line):
    if line.startswith("/"):
        tokens = line[1:].split(" ")
        handle_buildin_function(tokens)
        print("voy a ejecutar uno de los comandos reservados")

    elif "=" in line:
        handle_define(line)
    else:
        return handle_evaluate(line)

def display():
    st.title("Welcome user")
    def new_line():
        st.session_state.submitted_texts.append(text_input)
    # Inicializar la lista de textos enviados si no existe en session_state
    if 'submitted_texts' not in st.session_state:
        st.session_state.submitted_texts = []

    # Mostrar todas las líneas enviadas
    if st.session_state.submitted_texts:
        st.subheader("Submitted Lines")
        all_texts = "\n".join(st.session_state.submitted_texts)
        st.code(all_texts, language="")

    # Área de texto de una sola línea para escribir la línea
    text_input = st.text_input("Write your command here")

    # Botón para enviar el texto
    if st.button("Submit", on_click=new_line,):
        if text_input:
            try:
                tmp=process_line(text_input)
                if tmp != None:
                    st.success(tmp)
                else:
                    st.success("Text submitted successfully!")
            except AssertionError as err:
                st.warning(err)
        else:
            st.warning("Please write something before submitting.")
    if os.path.exists('figure.png'):
        # Si el archivo existe, eliminarlo
        st.image("figure.png", use_column_width=True)
    if os.path.exists('figure.gif'):
        # Si el archivo existe, eliminarlo
        st.image("figure.gif", use_column_width=True)
    