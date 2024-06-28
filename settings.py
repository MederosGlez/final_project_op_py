import streamlit as st
import pickle
from tools import handle_config


def get_settings():
    # Get core (integer only)
    core = st.number_input("Enter an integer value for core", value=1, step=1)
    col1, col2= st.columns(2)
    with col1:
        # Get type (spatial or temporal)
        type_choice = st.selectbox("Select type", ["spatial", "temporal"])
    with col2:
        # Get kind (plot or scatter)
        kind_choice = st.selectbox("Select kind", ["plot", "scatter"])
    
    # Get polar (yes or no)
    polar_choice = st.radio("Select polar", ["yes", "no"])

    col1, col2, col3 = st.columns(3)
    with col1:
        x_range_begin = st.number_input("X Range Begin", value=0.0)
    with col2:
        x_range_finish = st.number_input("X Range Finish", value=10.0)
    with col3:
        x_range_step = st.number_input("X Range Step", value=1.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        y_range_begin = st.number_input("Y Range Begin (optional)", value=0.0)
    with col2:
        y_range_finish = st.number_input("Y Range Finish (optional)", value=10.0)
    with col3:
        y_range_step = st.number_input("Y Range Step (optional)", value=1.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        z_range_begin = st.number_input("Z Range Begin (optional)", value=0.0)
    with col2:
        z_range_finish = st.number_input("Z Range Finish (optional)", value=10.0)
    with col3:
        z_range_step = st.number_input("Z Range Step (optional)", value=1.0)

    # Get t_range (optional)
    # t_range_begin = st.number_input("Enter the beginning value for t_range (optional)", value=0.0)
    # t_range_finish = st.number_input("Enter the ending value for t_range (optional)", value=10.0)
    # t_range_step = st.number_input("Enter the step value for t_range (optional)", value=1.0)

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
        # 't_range': {
        #     'begin': t_range_begin,
        #     'finish': t_range_finish,
        #     'step': t_range_step
        # },
        'core': int(core),
        'consal': type_choice,
        'kind': kind_choice,
        'polar': polar_choice,
    }

    return settings

def settings():
    st.title("User Settings")
    user_settings = get_settings()
    st.write("User settings:")
    for key, value in user_settings.items():
        st.write(f"{key}: {value}")

    # Save settings when the user clicks the button
    if st.button("Save Settings"):
        with open('user_settings.pkl', 'wb') as f:
            pickle.dump(user_settings, f)
        handle_config()
        st.success("Settings saved successfully!")

