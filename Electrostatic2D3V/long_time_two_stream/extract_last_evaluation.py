import sys
import numpy as np
import h5py

def get_step_keys(data):
    """
    Returns sorted list of step names.
    """
    return sorted(data.keys(), key=lambda x: int(x.split("#")[-1]))

def get_last_evaluations(data):
    """
    Gets the last set of potential evaluations from the input file.
    """
    
    # Get the last data entry key
    last_step_name = get_step_keys(data)[-1]
    
    # grid indices in x
    index_values_0 = data[last_step_name]["INDEX_0"]

    # grid indices in y
    index_values_1 = data[last_step_name]["INDEX_1"]
    for ix in index_values_1:
        assert ix == 0, "Script only written for 1 line in the y direction."
    
    # raw potential values in the same order as 
    phi_values_raw = data[last_step_name]["FIELD_EVALUATION_0"]
    x_values_raw = data[last_step_name]["x"]
    
    # It is not given that the points are in the correct order so we reorder.
    num_points = len(x_values_raw)
    x_values = np.empty(num_points)
    phi_values = np.empty(num_points)
    for index, index_value in enumerate(index_values_0):
        phi_values[index_value] = phi_values_raw[index]
        x_values[index_value] = x_values_raw[index]

    return x_values, phi_values


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
"""
Run like:
    python3 plot_phi_fft.py Electrostatic2D3V_line_field_evaluations.h5part
""")
        quit()

    data_h5 = h5py.File(sys.argv[1], "r")
    
    x_values, phi_values = get_last_evaluations(data_h5)

    print(x_values)
    print(phi_values)



