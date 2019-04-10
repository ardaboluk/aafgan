
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from catmullrom import CatmullRomActivation

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

    num_inputs = 100
    input_dim = 1000

    # 3 neurons, 22 control points for each one
    initial_control_points = torch.tensor([initialize_cp_tanh(range_min, range_max, num_control_points),
        initialize_cp_tanh(range_min, range_max, num_control_points), initialize_cp_tanh(range_min, range_max, num_control_points)])

    cm = CatmullRomActivation(range_min, range_max, input_dim, initial_control_points.clone())

    inputs = torch.tensor(np.random.rand(100,input_dim))
    output = cm(inputs)

    print(output)

    #plt.plot(inputs.numpy(), output.data.numpy()[:, 0])
    #plt.show()
