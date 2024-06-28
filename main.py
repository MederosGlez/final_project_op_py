
import multiprocessing
import re
from tools import handle_define, handle_evaluate, handle_gen, handle_config
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool



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
    "load": handle_load,
    "config": handle_config
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


if __name__ == "__main__":
    lines = [
        "/load input.txt",
        "5+5"
        "/exit"
        "/gen h",
    ]
    print('aguacate')
    handle_config("5")
    for line in lines:
        if re.match("/exit",line):
            exit()
        else:
            try:
                process_line(line)
            except AssertionError as err:
                print(err)

    print("el programa finalizo correctamente")
