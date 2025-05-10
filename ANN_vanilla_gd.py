import pandas as pd
import math

# Hyperparameters
n_epochs = 6100
learning_rate = 0.7

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(sig_x):
    return sig_x * (1 - sig_x)

# Matrix operations
def dot(mat, vec):
    return [sum(m * v for m, v in zip(row, vec)) for row in mat]

def transpose(matrix):
    return list(map(list, zip(*matrix)))

def apply(vec, func):
    return [func(v) for v in vec]

def mse_loss(y_true, y_pred):
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

# Input and target
x = [0.3, 0.1, 0.2]
y = [1, 0, 0]

# Initial weights
v = [[-1.0, -0.5, 0.5], [1.0, 0.0, -0.5], [0.5, -0.5, 0.5]]
w = [[-1.0, -0.5, 0.5], [1.0, 0.0, 0.5], [0.5, -0.5, 0.5]]
t = [[0.0, 0.0, 0.0], [0.5, 0.0, -0.5], [0.0, 0.5, -0.5]]

for epoch in range(n_epochs):
    # FORWARD
    z1 = dot(transpose(v), x)
    a1 = apply(z1, sigmoid)

    z2 = dot(transpose(w), a1)
    a2 = apply(z2, sigmoid)

    z3 = dot(transpose(t), a2)
    a3 = apply(z3, sigmoid)

    loss = mse_loss(y, a3)

    # LMS OUTPUT-ERROR RULE
    delta3 = [(yt - at) * at * (1 - at) for yt, at in zip(y, a3)]

    # BACKPROPAGATE to hidden2
    t_T = transpose(t)
    delta2 = []
    for j in range(3):
        error = sum(delta3[k] * t_T[j][k] for k in range(3))
        delta2.append(error * sigmoid_derivative(a2[j]))

    # BACKPROPAGATE to hidden1
    w_T = transpose(w)
    delta1 = []
    for j in range(3):
        error = sum(delta2[k] * w_T[j][k] for k in range(3))
        delta1.append(error * sigmoid_derivative(a1[j]))

    # GRADIENTS
    grad_t = [[delta3[j] * a2[i] for j in range(3)] for i in range(3)]
    grad_w = [[delta2[j] * a1[i] for j in range(3)] for i in range(3)]
    grad_v = [[delta1[j] * x[i] for j in range(3)] for i in range(3)]

    # WEIGHT UPDATE (gradient descent)
    for i in range(3):
        for j in range(3):
            v[i][j] += learning_rate * grad_v[i][j]
            w[i][j] += learning_rate * grad_w[i][j]
            t[i][j] += learning_rate * grad_t[i][j]

# FINAL OUTPUT
final_output = a3
desired = y
actual = final_output
percent_error = [
    abs((d - c) / d * 100) if d != 0 else abs(c * 100)
    for d, c in zip(desired, actual)
]

# Result
print("\n=== Final Report ===")
print(f"Desired Output: {{ {desired[0]:.4f}, {desired[1]:.4f}, {desired[2]:.4f} }}")
print(f"Actual Output : {{ {actual[0]:.4f}, {actual[1]:.4f}, {actual[2]:.4f} }}")
print(f"Percent Error : {{ {percent_error[0]:.2f}%, {percent_error[1]:.2f}%, {percent_error[2]:.2f}% }}")
