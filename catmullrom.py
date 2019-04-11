
# catmullrom.py
# S. Arda BÖLÜK, 2019
# This code interpolates a given point using Catmull-Rom spline.

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class CatmullRomActivation(nn.Module):

    def __init__(self, device, range_min, range_max, input_dim, initial_control_points):
        """range_min and range_max specify the range in which the control points are defined.
        initial_control_points_mat hyperparameter spcifies control points for each neuron in the layer.
        It's shape is assumed to be (1xk) where k is the number of control points.
        This will be replicated as many times as number of neurons so that the control_points_mat will be of size (num_neurons x k).
        The first entry in a row of control_points_mat is assumed to be the first ghost point.
        Similarly, the last entry in a row of control_points_mat is assumed to be the second ghost point."""

        super(CatmullRomActivation, self).__init__()
        
        self.device = device

        self.range_min = range_min
        self.range_max = range_max
        # number of neurons is the same as the input dimension
        self.num_neurons = input_dim

        # initial control points aren't used, instead, a copy of these are used as a result of the repear operation.
        # control_points_mat should be wrapped with Parameter as it's a parameter of the model.
        self.control_points_mat = Parameter(initial_control_points.repeat(self.num_neurons, 1), requires_grad = True)

        # number of control points, excluding the ghost points
        self.cp_num = self.control_points_mat.size()[1] - 2

        self.delta_x = (range_max - range_min) / self.cp_num

        # basis matrix for Catmull-Rom spline
        self.basis_mat = 0.5 * torch.tensor(
        [[-1., 3., -3., 1.],
        [2., -5., 4., -1.],
        [-1., 0., 1., 0.],
        [0., 2., 0., 0.]]).to(self.device)

    def forward(self, input_s_vec):
        """Returns the Catmull-Rom spline interpolation for a given set of points.
        input_s_vec is assumed to be a tensor of size (mxn) where m is the number of inputs and n is the input dimension. """

        # output is of size (mxn) where m is the number of inputs and n is the number of neurons in the layer
        output_mat = torch.empty(input_s_vec.size()[0], self.num_neurons).to(self.device)

        # indices of each control point are also of size (mxn), because we need different indices for each neuron for each sample
        # inputs should be between range_min and range_max
        # if it is, map each input to index of corresponding P0
        # if not, each one to one of the corresponding edge interval, depending on the magnitude
        p_0_ind = torch.floor((input_s_vec - self.range_min) * (self.cp_num - 2) / (self.range_max - self.range_min) + 1)
        p_0_ind[input_s_vec <= self.range_min] = 1
        p_0_ind[input_s_vec >= self.range_max] = self.cp_num - 1
        p_0_ind = p_0_ind.int()
        #p_0_ind = p_0_ind.unsqueeze(1)

        # find the indices of the 3 remaining control points
        p__1_ind = p_0_ind - 1
        p_1_ind = p_0_ind + 1
        p_2_ind = p_0_ind + 2

        # normalize the input
        # u is also of size (mxn)
        temp_u_vec = input_s_vec / self.delta_x
        u = temp_u_vec - torch.floor(temp_u_vec)

        # vectorize the normalized inputs to obtain matrix U of size ((m*n) x 4)
        U = torch.stack((torch.pow(u, 3), torch.pow(u, 2), u, torch.ones(u.size()).to(self.device)), 2).view(input_s_vec.size()[0] * self.num_neurons, 4)
        
        # vectorize the Q matrix so that it's of size (4 x (n*m))
        Q = torch.cat((torch.gather(self.control_points_mat, 1, p__1_ind.t().long()).view(1, self.num_neurons * input_s_vec.size()[0]), 
            torch.gather(self.control_points_mat, 1, p_0_ind.t().long()).view(1, self.num_neurons * input_s_vec.size()[0]), 
            torch.gather(self.control_points_mat, 1, p_1_ind.t().long()).view(1, self.num_neurons * input_s_vec.size()[0]),
            torch.gather(self.control_points_mat, 1, p_2_ind.t().long()).view(1, self.num_neurons * input_s_vec.size()[0])), 0)
        # Q = torch.cat((torch.gather(self.control_points_mat, 1, p__1_ind.t().long()), torch.gather(self.control_points_mat, 1, p_0_ind.t().long()), 
            # torch.gather(self.control_points_mat, 1, p_1_ind.t().long()), torch.gather(self.control_points_mat, 1, p_2_ind.t().long())), 1).view(4, self.num_neurons * input_s_vec.size()[0])
            
        mat_1 = U.mm(self.basis_mat).to(self.device)
        mat_1 = torch.index_select(mat_1, 1, torch.tensor([3,2,1,0]).to(self.device))
        output_mat = torch.sum(mat_1.t() * Q.float(), 0).view(input_s_vec.size()[0], self.num_neurons)
        
        return output_mat
