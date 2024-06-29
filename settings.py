import streamlit as st
import pickle
from tools import handle_config

# Function to load saved settings
def load_settings():
    try:
        with open('user_settings.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def get_settings(saved_settings):
    # Set default values from saved settings or defaults if no saved settings
    default_core = saved_settings['core'] if saved_settings else 1
    default_type = saved_settings['consal'] if saved_settings else "spatial"
    default_kind = saved_settings['kind'] if saved_settings else "plot"
    default_polar = saved_settings['polar'] if saved_settings else "no"
    
    default_x_range_begin = saved_settings['x_range']['begin'] if saved_settings else 0.0
    default_x_range_finish = saved_settings['x_range']['finish'] if saved_settings else 10.0
    default_x_range_step = saved_settings['x_range']['step'] if saved_settings else 1.0
    
    default_y_range_begin = saved_settings['y_range']['begin'] if saved_settings else 0.0
    default_y_range_finish = saved_settings['y_range']['finish'] if saved_settings else 10.0
    default_y_range_step = saved_settings['y_range']['step'] if saved_settings else 1.0
    
    default_z_range_begin = saved_settings['z_range']['begin'] if saved_settings else 0.0
    default_z_range_finish = saved_settings['z_range']['finish'] if saved_settings else 10.0
    default_z_range_step = saved_settings['z_range']['step'] if saved_settings else 1.0

    # Get core (integer only)
    core = st.number_input("Enter an integer value for core", value=default_core, step=1)
    col1, col2= st.columns(2)
    with col1:
        # Get type (spatial or temporal)
        type_choice = st.selectbox("Select type", ["spatial", "temporal"], index=["spatial", "temporal"].index(default_type))
    with col2:
        # Get kind (plot or scatter)
        kind_choice = st.selectbox("Select kind", ["plot", "scatter"], index=["plot", "scatter"].index(default_kind))
    
    # Get polar (yes or no)
    polar_choice = st.radio("Select polar", ["yes", "no"], index=["yes", "no"].index(default_polar))

    col1, col2, col3 = st.columns(3)
    with col1:
        x_range_begin = st.number_input("X Range Begin", value=default_x_range_begin)
    with col2:
        x_range_finish = st.number_input("X Range Finish", value=default_x_range_finish)
    with col3:
        x_range_step = st.number_input("X Range Step", value=default_x_range_step)

    col1, col2, col3 = st.columns(3)
    with col1:
        y_range_begin = st.number_input("Y Range Begin (optional)", value=default_y_range_begin)
    with col2:
        y_range_finish = st.number_input("Y Range Finish (optional)", value=default_y_range_finish)
    with col3:
        y_range_step = st.number_input("Y Range Step (optional)", value=default_y_range_step)

    col1, col2, col3 = st.columns(3)
    with col1:
        z_range_begin = st.number_input("Z Range Begin (optional)", value=default_z_range_begin)
    with col2:
        z_range_finish = st.number_input("Z Range Finish (optional)", value=default_z_range_finish)
    with col3:
        z_range_step = st.number_input("Z Range Step (optional)", value=default_z_range_step)

    settings = {
        'x_range': {
            'begin': x_range_begin,
            'finish': x_range_finish,
            'step': x_range_step
        },
        'y_range': {
            'begin': y_range_begin,
            'finish': y_range_finish,
            'step': y_range_step
        },
        'z_range': {
            'begin': z_range_begin,
            'finish': z_range_finish,
            'step': z_range_step
        },
        'core': int(core),
        'consal': type_choice,
        'kind': kind_choice,
        'polar': polar_choice,
    }

    return settings

def settings():
    st.title("User Settings")
    
    # Load saved settings
    saved_settings = load_settings()
    
    user_settings = get_settings(saved_settings)
    
    st.write("User settings:")
    for key, value in user_settings.items():
        st.write(f"{key}: {value}")

    # Save settings when the user clicks the button
    if st.button("Save Settings"):
        with open('user_settings.pkl', 'wb') as f:
            pickle.dump(user_settings, f)
        handle_config()
        st.success("Settings saved successfully!")

# Run the settings function to display the UI
settings()
