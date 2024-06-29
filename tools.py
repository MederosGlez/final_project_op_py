from constants import *
from rpn_notation import evaluate_function, evaluate_postfix, shunting_yard


import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

import re
import operator
import math
from collections import deque
import os

from collections import OrderedDict
from multiprocessing import Pool
import pickle


rango_x = [0, np.pi, 0.05]
rango_y = [0, np.pi, 0.05]
rango_z = [0,1,0.1]
core = 5
condimento = "spatial"
kind = "plot"
polar = False


function_name_defs = {}


def parse_function_definition(function_name_def):
    match = re.match(
        r'^\s*([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*=\s*(.+)\s*$', function_name_def)
    if not match:
        raise ValueError("Definicion de function_nameion invalida")
    name, params, body = match.groups()
    params = [p.strip() for p in params.split(',')] if params else []

    tokens = pre_process_expression(body)
    idents = {text for t_name, text in tokens if t_name == 'ident'}

    undefined = idents - set(params) - set(OPERATORS) - \
        set(BUILD_IN_FUNCTIONS) - set(CONSTANTS)
    if undefined:
        raise ValueError(f"Parametros no definidos: {', '.join(undefined)}")

    _shunting_yard = shunting_yard(tokens).tokens

    return name, params, _shunting_yard



def pre_process_expression(expression):
    matches = re.finditer(
        PATTERN,
        expression
    )

    tokens = [
        (TOKENS_TYPES(token.lastgroup), token.group())
        for token in matches
        if token.lastgroup is not None
    ]

    tokens = [
        (t_type, t_value[:-1])
        if t_type == TOKENS_TYPES.FUNCTION_CALL
        else (t_type, t_value)
        for (t_type, t_value)
        in tokens
    ]
    return tokens


def evaluate_expression(expression, vars):

    tokens = pre_process_expression(expression)
    _shunting_yard = shunting_yard(tokens).tokens

    return evaluate_postfix(_shunting_yard, vars)


def log_interaction(log_file, interaction):
    with open(log_file, 'a') as f:
        f.write(interaction + '\n')


def handle_define(function_name_def):
    function_name_name, function_name_params, function_name_body = parse_function_definition(function_name_def)

    try:
        USERS_FUNCTIONS[function_name_name] = dict(
            function_name_params=function_name_params,
            function_name_body=function_name_body,
            function_name_def=function_name_def
        )
        print(f"function_nameión '{function_name_name}' definida exitosamente.")
    except ValueError as e:
        print(f"Error al definir la function_nameión '{function_name_name}': {e}")


def handle_evaluate(expression):
    try:
        result = evaluate_expression(expression, {})
        print(result)
        return result
    except (ValueError, ZeroDivisionError) as e:
        return f"Error: {e}"


def handle_list(function_name_defs):
    if function_name_defs:
        return "function_nameiones definidas:\n" + "\n".join(f"{name}({', '.join(params)}) = {body}" for name, (params, body) in function_name_defs.items())
    else:
        return "Aun no hay function_nameiones definidas."


def save_database():
    print("Guardando base de datos")
    print(f"Funciones definidas: {[*USERS_FUNCTIONS.keys()]}")

    with open(f'.save_functions.txt', mode='wb') as f:
        pickle.dump(USERS_FUNCTIONS, f)


def handle_config():

    with open(f'user_settings.pkl', mode='rb') as f:
        settings = pickle.load(f)

    global rango_x, rango_y, rango_z, kind, core, condimento, polar

    rango_x = [settings["x_range"]["begin"],settings["x_range"]["finish"],settings["x_range"]["step"]]    
    rango_y = [settings["y_range"]["begin"],settings["y_range"]["finish"],settings["y_range"]["step"]]
    rango_z = [settings["z_range"]["begin"],settings["z_range"]["finish"],settings["z_range"]["step"]]

    core = settings["core"]
    condimento = settings["consal"]
    kind = settings["kind"]

    if settings["polar"] == "yes":
        polar = True
    else:
        polar = False


def load_database():
    global function_name_defs
    with open(f'.save_functions.txt', mode='rb') as f:
        function_name_defs = pickle.load(f)


def _handle_gen(func_name, args):
    print(args)
    #config_args = args[0]
    #args = [*args[1:]]
    func_description = USERS_FUNCTIONS[func_name]
    return [*args, evaluate_function(func_description, args)]


def handle_gen(function_name, *args):


    if os.path.exists('figure.png'):
        # Si el archivo existe, eliminarlo
        os.remove('figure.png')


    if os.path.exists('figure.gif'):
        # Si el archivo existe, eliminarlo
        os.remove('figure.gif')


    save_database()


    function_description = USERS_FUNCTIONS[function_name]
    params_definition = function_description['func_params']
    body = function_description['func_body']
    function_name_def = function_description['func_def']

    print(function_description)
    print("lol" , params_definition)
    print("body",body)
    print("run",function_name_def)
    print(_handle_gen(function_name,params_definition))

    params = []
    rangos = [rango_x, rango_y, rango_z]


    for i in range(len(params_definition)):
        params.append(
            list(
                np.arange(
                    rangos[i][0],
                    rangos[i][1],
                    rangos[i][2]
                )
            )
        )
    
    print(params)

    if len(params_definition) == 1:
        params_1 = params[0]

        result = []

        with Pool(core) as p:
            result = p.map(
                _handle_gen,
                product([(function_name, args)], *params)
            )

        print(result)

        x,y=np.array(params_1), np.array(result)

        if(polar):
            x,y=y*np.cos(x),y*np.sin(x)
        result = list(zip(params_1, result))
        fig, ax = plt.subplots()
        # Plot some data on the Axes.
        if kind == "plot":
            ax.plot(x, y)
        elif kind == "scatter":
            ax.scatter(x, y)
        else:
            print("Introduce una opcion correcta")
        fig.savefig("figure.png")
        print("guarde el grafico correctamente")
        plt.show()


    elif len(args) == 2:
        print('LLegue')
        x , y = params
        params_1=[]
        for i in x:
            for j in y:
                params_1.append([i,j])
        result = []
        with Pool(core) as p:
            result = p.map(
                _handle_gen,
                [
                    (function_name, param_1)
                    for param_1 in params_1
                ]
            )
        x = [u[0] for u in params_1]
        y = [u[1] for u in params_1]
        z = result
        if condimento == "spatial":
            print("spatial")
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if kind == "plot":
                xi = np.linspace(min(x), max(x), 100)
                yi = np.linspace(min(y), max(y), 100)
                xi, yi = np.meshgrid(xi, yi)
                zi = griddata((x, y), z, (xi, yi), method='cubic')
                ax.plot_surface(xi, yi, zi, cmap='viridis')
            elif kind == "scatter":
                ax.scatter(x, y,z)
            else:
                print("Introduce una opcion correcta")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            fig.savefig("figure.png")
            print("guarde el grafico correctamente")
            plt.show()
        elif condimento == "temporal":
            print("temporal")
            y,x,t=z,x,y 
            x,y=np.array(x), np.array(y)
            if(polar):
                x,y=y*np.cos(x),y*np.sin(x)
            mapa = OrderedDict()
            # Iterar sobre los datos x, y, t
            for _x, _y, _t in zip(x, y, t):
                if _t not in mapa:
                    mapa[_t] = []  # Inicializar la lista para este tiempo si no existe
                mapa[_t].append([_x, _y])  # Agregar el par de coordenadas

            datos = []
            for i in mapa.values():
                datos.append(i)
            # Mostrar el resultado
            
            def update(frame):
                ax.clear()
                datox=[i[0] for i in datos[frame]]
                datoy=[i[1] for i in datos[frame]]
                if kind == "plot":
                    ax.plot(datox, datoy)
                elif kind == "scatter":
                    ax.scatter(datox, datoy)
                else:
                    print("Introduce una opcion correcta")
                ax.set_xlim(min(x), max(x))
                ax.set_ylim(min(y), max(y))
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(f'Time: {frame:.2f}')
            fig, ax = plt.subplots()
            ani = function_nameAnimation(fig, update, frames=len(datos), repeat=True)
            ani.save('figure.gif', writer=PillowWriter(fps=20))
            print('guarde la animacion correctamente')
            plt.show()
    elif len(args)==3:
        x , y, z = params
        params_1=[]
        for i in x:
            for j in y:
                for k in z:
                    params_1.append([i,j,k])
        result = []
        with Pool(core) as p:
            result = p.map(
                _handle_gen,
                [
                    (function_name, param_1)
                    for param_1 in params_1
                ]
            )
        x = [u[0] for u in params_1]
        y = [u[1] for u in params_1]
        z = [u[2] for u in params_1]
        _z = result
        z,y,x,t=_z,y,x,z 
        
        mapa = OrderedDict()
        # Iterar sobre los datos x, y, t
        for _x, _y, _z , _t in zip(x, y, z, t):
            if _t not in mapa:
                mapa[_t] = []  # Inicializar la lista para este tiempo si no existe
            mapa[_t].append([_x, _y, _z])  # Agregar el par de coordenadas

        datos = []
        for i in mapa.values():
            datos.append(i)
        
        def update(frame):
            ax.clear()
            datox=[i[0] for i in datos[frame]]
            datoy=[i[1] for i in datos[frame]]
            datoz=[i[2] for i in datos[frame]]
            if kind == "plot":
                xi = np.linspace(min(x), max(x), 100)
                yi = np.linspace(min(y), max(y), 100)
                xi, yi = np.meshgrid(xi, yi)
                zi = griddata((datox, datoy), datoz, (xi, yi), method='cubic')
                ax.plot_surface(xi, yi, zi, cmap='viridis')
            elif kind == "scatter":
                ax.scatter(datox, datoy,datoz)
            else:
                print("Introduce una opcion correcta")
            ax.set_xlim(min(x), max(x))
            ax.set_ylim(min(y), max(y))
            ax.set_zlim(min(z), max(z))
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Time: {frame:.2f}')
            plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ani = function_nameAnimation(fig, update, frames=len(datos), repeat=True)
        ani.save('figure.gif', writer=PillowWriter(fps=20))
        print('guarde la animacion correctamente')
        plt.show()
    else:
        print('Demasiados argumentos para graficar D:')

def test__handle_gen():
    _handle_gen('f', 1)

handle_gen("f")
