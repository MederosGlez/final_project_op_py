import streamlit as st
import pickle

def get_settings():

    # Get core (integer only)
    core = st.number_input("Enter an integer value for core", value=1, step=1)

    # Get type (spatial or temporal)
    type_choice = st.selectbox("Select type", ["spatial", "temporal"])

    # Get kind (plot or scatter)
    kind_choice = st.selectbox("Select kind", ["plot", "scatter"])

    # Get polar (yes or no)
    polar_choice = st.radio("Select polar", ["yes", "no"])

    # Get dimension (2D, 2D+T, 3D, 3D+T)
    dimension_choice = st.selectbox("Select dimension", ["2D", "2D+T", "3D", "3D+T"])

    x_range_begin = st.number_input("Enter the beginning value for x_range", value=0.0)
    x_range_finish = st.number_input("Enter the ending value for x_range", value=10.0)
    x_range_step = st.number_input("Enter the step value for x_range", value=1.0)

    # Get y_range (optional)
    y_range_begin = st.number_input("Enter the beginning value for y_range (optional)", value=0.0)
    y_range_finish = st.number_input("Enter the ending value for y_range (optional)", value=10.0)
    y_range_step = st.number_input("Enter the step value for y_range (optional)", value=1.0)

    # Get z_range (optional)
    z_range_begin = st.number_input("Enter the beginning value for z_range (optional)", value=0.0)
    z_range_finish = st.number_input("Enter the ending value for z_range (optional)", value=10.0)
    z_range_step = st.number_input("Enter the step value for z_range (optional)", value=1.0)

    # Get t_range (optional)
    #t_range_begin = st.number_input("Enter the beginning value for t_range (optional)", value=0.0)
    #t_range_finish = st.number_input("Enter the ending value for t_range (optional)", value=10.0)
    #t_range_step = st.number_input("Enter the step value for t_range (optional)", value=1.0)

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

        #'t_range': {
            #'begin': t_range_begin,
            #'finish': t_range_finish,
            #'step': t_range_step
        #},

        'core': int(core),
        'consal': type_choice,
        'kind': kind_choice,
        'polar': polar_choice,
        'dimension': dimension_choice
    }

    return settings

def main():
    st.title("User Settings")
    user_settings = get_settings()
    st.write("User settings:")
    for key, value in user_settings.items():
        st.write(f"{key}: {value}")

    # Save settings when the user clicks the button
    if st.button("Save Settings"):
        with open('user_settings.pkl', 'wb') as f:
            pickle.dump(user_settings, f)
        st.success("Settings saved successfully!")

if __name__ == "__main__":
    main()