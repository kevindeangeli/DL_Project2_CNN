'''
Created by: Kevin De Angeli
Email: kevindeangeli@utk.edu
Date: 2020-01-16

'''

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    x = z * (1 - z)
    return z * (1 - z)

def linear(x):
    return x