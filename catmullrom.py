
# catmullrom.py
# S. Arda BÖLÜK, 2019
# This code interpolates a given point using Catmull-Rom spline.

import numpy as np
import torch
from torch.autograd import Variable

class CatmullRomSpline:

    def __init__(self, range_min, range_max, control_points_mat):
        """range_min and range_max specify the range in which the control points are defined.
        control_points_mat hyperparameter spcifies control points for each neuron in the layer.
        It's shape is (nxk) where n is the number of neurons and k is the number of control points.
        The first entry in control_points_arr is assumed to be the first ghost point.
        Similarly, the last entry in control_points_arr is assumed to be the second ghost point."""

        self.range_min = range_min
        self.range_max = range_max

        # control_points_mat is assumed to be a tensor wrapped with Variable, because its gradient will be calculated
        self.control_points_mat = control_points_mat

        # number of control points, excluding the ghost points
        self.cp_num = self.control_points_mat.size()[1] - 2

        self.delta_x = (range_max - range_min) / self.cp_num

        # basis matrix for Catmull-Rom spline
        self.basis_mat = 0.5 * torch.tensor(
        [[-1., 3., -3., 1.],
        [2., -5., 4., -1.],
        [-1., 0., 1., 0.],
        [0., 2., 0., 0.]]).double()

    def interpolate_CR(self, input_s_vec):
        """Returns the Catmull-Rom spline interpolation for a given set of points.
        input_s_vec is assumed to be a numpy array"""

        # input_s_vec is assumed to be a tensor wrapped with Variable, but its gradient won't be calculated

        # output is a mxn matrix where m is the number of inputs and n is the number of neurons in the layer
        output_mat = torch.empty(input_s_vec.size()[0], self.control_points_mat.size()[0])

        # inputs should be between range_min and range_max
        # if it is, map each input to index of corresponding P0
        # if not, each one to one of the corresponding edge interval, depending on the magnitude
        p_0_ind = torch.floor((input_s_vec - self.range_min) * (self.cp_num - 2) / (self.range_max - self.range_min) + 1)
        p_0_ind[input_s_vec <= self.range_min] = 1
        p_0_ind[input_s_vec >= self.range_max] = self.cp_num - 1
        p_0_ind = p_0_ind.int()

        # find the indices of the 3 remaining control points
        p__1_ind = p_0_ind - 1
        p_1_ind = p_0_ind + 1
        p_2_ind = p_0_ind + 2

        # normalize the input
        temp_u_vec = input_s_vec / self.delta_x
        u = temp_u_vec - torch.floor(temp_u_vec)

        # vectorize the normalized inputs to obtain matrix U of size (mx4) where m is the number of inputs
        U = torch.stack((torch.pow(u, 3), torch.pow(u, 2), u, torch.ones(u.size()[0]).double()), 1).unsqueeze(1)

        for input_ind in range(input_s_vec.size()[0]):
            Q = torch.stack((self.control_points_mat[:, p__1_ind[input_ind]], self.control_points_mat[:, p_0_ind[input_ind]],
                self.control_points_mat[:, p_1_ind[input_ind]], self.control_points_mat[:, p_2_ind[input_ind]]), 0)

            output_mat[input_ind] = torch.mm(torch.mm(U[input_ind], self.basis_mat), Q)

        return output_mat
