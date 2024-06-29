from constants import *
import numpy as np
import pickle

rango_x = [0, np.pi, 0.05]
rango_y = [0, np.pi, 0.05]
rango_z = [0,1,0.1]
core = 5
condimento = "spatial"
kind = "plot"
polar = False


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

    print(settings["dimension"])
    print(polar)
    print(rango_x)

#handle_config()
