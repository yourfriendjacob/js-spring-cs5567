from tsensor import explain as exp
import numpy as np

# activation function
def relu(x):
    return np.maximum(0, x)

# gradient descent function
def gradient_mse(actual, predicted):
    return predicted - actual

def calculate_mse(actual, predicted):
    return (actual - predicted) ** 2

x_input = np.array([[10], [20], [-20], [-40], [-3]])

# 1x2 weight matrix
l1_weights = np.array([[.73, .2]])

# 1x2 bias matrix
l1_bias = np.array([[4, 2]])

# layer 2 weight and bias arrays
l2_weights = np.array([[1], [2]])
l2_bias = np.array([[4]])

# output of layer 1
with exp() as c:
    l1_output = x_input @ l1_weights + l1_bias
    l1_activated = relu(l1_output)

# output layer
with exp() as c:
    l2_output = l1_activated @ l2_weights + l2_bias

actual = np.array([[9], [13], [5], [-2], [-1]])

print(calculate_mse(actual,l2_output))

print(gradient_mse(actual,l2_output))

# get the output gradient
output_gradient = gradient_mse(actual, l2_output)

# get weight gradient
with exp():
    l2_w_gradient =  l1_activated.T @ output_gradient
l2_w_gradient

# get bias gradient
with exp():
    l2_b_gradient =  np.mean(output_gradient, axis=0)

l2_b_gradient


# Set a learning rate
lr = 1e-4

with exp():
    # Update the bias values
    l2_bias = l2_bias - l2_b_gradient * lr
    # Update the weight values
    l2_weights = l2_weights - l2_w_gradient * lr

l2_weights

with exp():
    # Calculate the gradient on the output of layer 1
    l1_activated_gradient = output_gradient @ l2_weights.T

l1_activated_gradient

with exp():
    l1_output_gradient = l1_activated_gradient * np.heaviside(l1_output, 0) 
    # heaviside is acting as a piecewise function on each element of l1_output
    # if x < 0 return 0
    # if x == 0 return x2
    # if x > 0 return 1
    # if this case heaviside is acting as the deritive of the relu activation function

l1_output_gradient

# back propagation
l1_w_gradient =  input.T @ l1_output_gradient
l1_b_gradient = np.mean(l1_output_gradient, axis=0)
# get the mean of the elements of the output gradient accross each row
# V
#|2 3|
#|3 4|

# (3+4) / 2
# (3+4) / 2

# results in a new matrix that is smaller but contains the mean of each row


# gradient descent
l1_weights -= l1_w_gradient * lr
l1_bias -= l1_b_gradient * lr

