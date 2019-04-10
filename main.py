
import numpy as np
import torch
import matplotlib.pyplot as plt

from catmullrom import CatmullRomSpline

def initialize_cp_tanh(range_min, range_max, cp_num):
    """Initializes the control points by sampling them uniformly from hyperbolic tangent tanh.
    Ghost points are trivial, the first one is to the left of the left-most control point.
    Similarly, the second one is to the right of the right-most control point."""

    init_cp = np.tanh(np.linspace(range_min, range_max, num=cp_num))

    return init_cp


if __name__ == "__main__":

    range_min = -2.
    range_max = 2.
    num_control_points = 22

    input_range_min = -10.
    input_range_max = 10.
    num_inputs = 100

    # 3 neurons, 22 control points for each one
    control_points = np.array([initialize_cp_tanh(range_min, range_max, num_control_points),
        initialize_cp_tanh(range_min, range_max, num_control_points), initialize_cp_tanh(range_min, range_max, num_control_points)])

    cm = CatmullRomSpline(range_min, range_max, control_points)

    inputs = np.linspace(input_range_min, input_range_max, num=num_inputs)
    output = cm.interpolate_CR(inputs)
