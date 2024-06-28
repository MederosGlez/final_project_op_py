import pickle
from multiprocessing import Pool
import re
import operator
import math
from collections import deque
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation, PillowWriter
from collections import OrderedDict


rango_x = [0, np.pi, 0.05]
rango_y = [0, np.pi, 0.05]
rango_z = [0,1,0.1]
core = 5
condimento = "spatial"
kind = "plot"
polar = False

# Diccionario de operadores con su precedencia, asociatividad y funcion correspondiente
OPERATORS = {
    '+': (1, 'L', operator.add),
    '-': (1, 'L', operator.sub),
    '*': (2, 'L', operator.mul),
    '/': (2, 'L', operator.truediv),
    '^': (3, 'R', operator.pow),
    'neg': (4, 'R', operator.neg)
}

# Diccionario de funciones matematicas
FUNCTIONS = {
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'cot': lambda x: 1 / math.tan(x),
    'log': math.log,
    'log10': math.log10,
    'sqrt': math.sqrt
}

# Diccionario de constantes matematicas
CONSTANTS = {
    'pi': math.pi,
    'e': math.e
}

func_defs = {}


def parse_function_definition(func_def):
    match = re.match(
        r'^\s*([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*=\s*(.+)\s*$', func_def)
    if not match:
        raise ValueError("Definicion de funcion invalida")
    name, params, body = match.groups()
    params = [p.strip() for p in params.split(',')] if params else []

    tokens = pre_process_expression(body)
    idents = {text for t_name, text in tokens if t_name == 'ident'}

    undefined = idents - set(params) - set(OPERATORS) - \
        set(FUNCTIONS) - set(CONSTANTS)
    if undefined:
        raise ValueError(f"Parametros no definidos: {', '.join(undefined)}")

    _shunting_yard = shunting_yard(tokens)

    return name, params, _shunting_yard


def shunting_yard(tokens):
    output, ops = deque(), []
    arg_count_stack = []

    for t_name, token in tokens:
        if t_name in {'float', 'int'}:
            output.append(float(token))
        elif t_name == 'ident':
            output.append(token)
        elif t_name == 'function_call':
            func_name = token[:-1]
            if func_name in FUNCTIONS or func_name in func_defs:
                ops.append(func_name)
                arg_count_stack.append(0)
                ops.append('(')
            else:
                print([*func_defs.keys()])
                raise ValueError(f"Token desconocido: {func_name}")
        elif token == ',':
            while ops and ops[-1] != '(':
                output.append(ops.pop())
            if arg_count_stack:
                arg_count_stack[-1] += 1
        elif token in OPERATORS:
            if (not output or output[-1] in OPERATORS or output[-1] in {'(', ')'}) and token == '-':
                token = 'neg'
            while (ops and ops[-1] in OPERATORS and
                    ((OPERATORS[token][1] == 'L' and OPERATORS[token][0] <= OPERATORS[ops[-1]][0]) or
                     (OPERATORS[token][1] == 'R' and OPERATORS[token][0] < OPERATORS[ops[-1]][0]))):
                output.append(ops.pop())
            ops.append(token)
        elif token == '(':
            ops.append(token)
        elif token == ')':
            while ops and ops[-1] != '(':
                output.append(ops.pop())
            ops.pop()
            if ops and ops[-1] not in OPERATORS and ops[-1] not in {'(', ')'}:
                func = ops.pop()
                arg_count = arg_count_stack.pop() + 1
                output.append((func, arg_count))

    while ops:
        output.append(ops.pop())

    return output


def evaluate_postfix(postfix, vars):
    stack = []
    while postfix:
        token = postfix.popleft()
        if isinstance(token, float):
            stack.append(token)
        elif isinstance(token, tuple):
            func_name, arg_count = token
            args = [stack.pop() for _ in range(arg_count)]
            if func_name in func_defs:
                stack.append(evaluate_function(
                    func_defs[func_name], args[::-1]))
            else:
                stack.append(FUNCTIONS[func_name](*args[::-1]))
        elif token in vars:
            stack.append(vars[token])
        elif token in CONSTANTS:
            stack.append(CONSTANTS[token])
        elif token in OPERATORS:
            if len(stack) < 2 and token != 'neg':
                raise ValueError(
                    f"Error: no hay suficientes operandos para el operador '{token}'")
            b = stack.pop()
            a = stack.pop() if token != 'neg' else 0
            stack.append(OPERATORS[token][2](a, b))
        else:
            raise ValueError(f"Token desconocido: {token}")

    return stack[0]


def pre_process_expression(expression):
    matches = re.finditer(
        r'(?P<float>\d+\.\d+)|(?P<int>\d+)|(?P<function_call>\w+\()|(?P<LPAR>\()|(?P<RPAR>\))|(?P<ident>\w+)|(?P<symbol>[+\-*/^])|(?P<args_separator>,)|(?:\s*)',
        expression
    )

    tokens = [
        (token.lastgroup, token.group())
        for token in matches
        if token.lastgroup is not None
    ]
    return tokens


def full_evaluate_expression(expression, vars):

    tokens = pre_process_expression(expression)
    _shunting_yard = shunting_yard(tokens)

    return evaluate_postfix(_shunting_yard, vars)


def evaluate_expression(expression, vars):

    return evaluate_postfix(expression, vars)


def evaluate_function(func, args):
    params, body = func
    if len(params) != len(args):
        raise ValueError("Numero incorrecto de argumentos")
    return evaluate_expression(body, dict(zip(params, args)))


def log_interaction(log_file, interaction):
    with open(log_file, 'a') as f:
        f.write(interaction + '\n')


func_defs = {}


def handle_define(func_def):
    try:
        func_name, func_params, func_body = parse_function_definition(func_def)
        func_defs[func_name] = (func_params, func_body)
        print(f"Funcion '{func_name}' definida exitosamente.")
        return f"Funcion '{func_name}' definida exitosamente."
    except ValueError as e:
        return f"Error: {e}"


def handle_evaluate(expression):
    try:
        result = full_evaluate_expression(expression, {})
        print(result)
        return result
    except (ValueError, ZeroDivisionError) as e:
        return f"Error: {e}"


def handle_list(func_defs):
    if func_defs:
        return "Funciones definidas:\n" + "\n".join(f"{name}({', '.join(params)}) = {body}" for name, (params, body) in func_defs.items())
    else:
        return "Aun no hay funciones definidas."


def _handle_gen(args):
    func_name = args[0]
    args = args[1]
    #print(f"Generando grafico de {func_name} con parametros {args}")
    load_database()

    func_desciption = func_defs[func_name]

    return evaluate_function(func_desciption, args)


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



def save_database():
    with open(f'.save_functions.txt', mode='wb') as f:
        pickle.dump(func_defs, f)


def load_database():
    global func_defs
    with open(f'.save_functions.txt', mode='rb') as f:
        func_defs = pickle.load(f)

import os
def handle_gen(func, *args):
    if os.path.exists('figure.png'):
        # Si el archivo existe, eliminarlo
        os.remove('figure.png')
    if os.path.exists('figure.gif'):
        # Si el archivo existe, eliminarlo
        os.remove('figure.gif')
    save_database()

    (args, _) = func_defs[func]
    params = []
    rangos = [rango_x,rango_y, rango_z]
    for i in range(len(args)):
        params.append(
            list(
                np.arange(
                    rangos[i][0],
                    rangos[i][1],
                    rangos[i][2]
                )
            )
        )
    if len(args) == 1:
        params_1 = params[0]

        result = []
        with Pool(core) as p:
            result = p.map(
                _handle_gen,
                [
                    (func, [param_1])
                    for param_1 in params_1
                ]
            )
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
                    (func, param_1)
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
            ani = FuncAnimation(fig, update, frames=len(datos), repeat=True)
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
                    (func, param_1)
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
        ani = FuncAnimation(fig, update, frames=len(datos), repeat=True)
        ani.save('figure.gif', writer=PillowWriter(fps=20))
        print('guarde la animacion correctamente')
        plt.show()
    else:
        print('Demasiados argumentos para graficar D:')

def test__handle_gen():
    _handle_gen('f', 1)
