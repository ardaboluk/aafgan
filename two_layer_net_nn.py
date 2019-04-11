import numpy as np
import torch
import itertools

from catmullrom import CatmullRomActivation
import util

"""
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.
This implementation uses the nn package from PyTorch to build the network.
PyTorch autograd makes it easy to define computational graphs and take gradients,
but raw autograd can be a bit too low-level for defining complex neural networks;
this is where the nn package can help. The nn package defines a set of Modules,
which you can think of as a neural network layer that has produces output from
input and may have some trainable weights or other state.
"""

device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

range_min = -2.
range_max = 2.
num_control_points = 22

# 3 neurons, 22 control points for each one
initial_control_points = torch.tensor(util.initialize_cp_tanh(range_min, range_max, num_control_points))
util.plot_control_points(range_min, range_max, initial_control_points)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# After constructing the model we use the .to() method to move it to the
# desired device.
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          #torch.nn.ReLU(),
          CatmullRomActivation(range_min, range_max, H, initial_control_points),
          torch.nn.Linear(H, D_out),
        ).to(device)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function. Setting
# reduction='sum' means that we are computing the *sum* of squared errors rather
# than the mean; this is for consistency with the examples above where we
# manually compute the loss, but in practice it is more common to use mean
# squared error as a loss by setting reduction='elementwise_mean'.
loss_fn = torch.nn.MSELoss(reduction='sum')

#crlayer = next(itertools.islice(model.modules(), 2, 3))
#loss_fn = loss_fn + torch.sum((crlayer.control_points_mat - initial_control_points)**2)

learning_rate = 1e-4
for t in range(500):
  # Forward pass: compute predicted y by passing x to the model. Module objects
  # override the __call__ operator so you can call them like functions. When
  # doing so you pass a Tensor of input data to the Module and it produces
  # a Tensor of output data.
  y_pred = model(x)

  # Compute and print loss. We pass Tensors containing the predicted and true
  # values of y, and the loss function returns a Tensor containing the loss.
  loss = loss_fn(y_pred, y)
  print(t, loss.item())

  # Zero the gradients before running the backward pass.
  model.zero_grad()

  # Backward pass: compute gradient of the loss with respect to all the learnable
  # parameters of the model. Internally, the parameters of each Module are stored
  # in Tensors with requires_grad=True, so this call will compute gradients for
  # all learnable parameters in the model.
  loss.backward()

  # Update the weights using gradient descent. Each parameter is a Tensor, so
  # we can access its data and gradients like we did before.
  with torch.no_grad():
    for name, param in model.named_parameters():
        #print(name)
        if name == '1.control_points_mat':
            param.data -= learning_rate * 1000 * param.grad
        else:
            param.data -= learning_rate * param.grad

util.plot_control_points(range_min, range_max, next(itertools.islice(model.modules(), 2, 3)).control_points_mat.data.numpy()[1])

# print the final control points of Catmull-Rom activation
print(torch.sum(torch.abs(next(itertools.islice(model.modules(), 2, 3)).control_points_mat - initial_control_points)).data.numpy())
