
import numpy as np
from matplotlib import pyplot as plt

def initialize_cp_tanh(range_min, range_max, cp_num):
    """Initializes the control points by sampling them uniformly from hyperbolic tangent tanh.
    Ghost points are trivial, the first one is to the left of the left-most control point.
    Similarly, the second one is to the right of the right-most control point."""

    init_cp = np.tanh(np.linspace(range_min, range_max, num=cp_num))

    return init_cp

def plot_control_points(range_min, range_max, control_points_arr):
    """Takes a vector of control points and plots corresponding Catmull-Rom interpolation.
    range_min and range_max specify range of the control points."""

    inputs = np.linspace(2 * range_min, 2 * range_max, num = 15)

    cp_num = control_points_arr.shape[0] - 2
    delta_x = (range_max - range_min) / cp_num
    basis_mat = 0.5 * np.array(
        [[-1., 3., -3., 1.],
        [2., -5., 4., -1.],
        [-1., 0., 1., 0.],
        [0., 2., 0., 0.]])

    outputs = []

    p_0_ind = np.floor((inputs - range_min) * (cp_num - 2) / (range_max - range_min) + 1)
    p_0_ind[inputs <= range_min] = 1
    p_0_ind[inputs >= range_max] = cp_num - 1
    p_0_ind = p_0_ind.astype(int)

    p__1_ind = p_0_ind - 1
    p_1_ind = p_0_ind + 1
    p_2_ind = p_0_ind + 2

    temp_u_vec = inputs / delta_x
    u = temp_u_vec - np.floor(temp_u_vec)

    U = np.array([np.power(u, 3), np.power(u, 2), u, np.ones((u.shape[0]))]).T

    for input_ind in range(inputs.shape[0]):
        Q = np.array([control_points_arr[p__1_ind[input_ind]], control_points_arr[p_0_ind[input_ind]], control_points_arr[p_1_ind[input_ind]], control_points_arr[p_2_ind[input_ind]]])
        outputs.append(np.dot(np.dot(U[input_ind], basis_mat), Q))

    plt.figure()
    plt.plot(inputs, outputs)
    plt.show()
