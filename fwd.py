from tsensor import explain as exp
import numpy as np

def relu(x):
    return np.maximum(0, x)

x_input = np.array([[10], [20], [-20], [-40], [-3]])

# 1x2 weight matrix
l1_weights = np.array([[.73, .2]])

# 1x2 bias matrix
l1_bias = np.array([[4, 2]])

l2_weights = np.array([[1], [2]])
l2_bias = np.array([[4]])

def gradient_mse(actual, predicted):
    return predicted - actual

# output
with exp() as c:
    l1_output = x_input @ l1_weights + l1_bias
    l1_activated = relu(l1_output)

with exp() as c:
    l2_output = l1_activated @ l2_weights + l2_bias

def calculate_mse(actual, predicted):
    return (actual - predicted) ** 2

actual = np.array([[9], [13], [5], [-2], [-1]])

print(calculate_mse(actual,l2_output))

print(gradient_mse(actual,l2_output))

output_gradient = gradient_mse(actual, l2_output)

with exp():
    l2_w_gradient =  l1_activated.T @ output_gradient
l2_w_gradient

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

l1_output_gradient

# back propagation
l1_w_gradient =  input.T @ l1_output_gradient
l1_b_gradient = np.mean(l1_output_gradient, axis=0)

# gradient descent
l1_weights -= l1_w_gradient * lr
l1_bias -= l1_b_gradient * lr

