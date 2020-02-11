'''
Created by: Kevin De Angeli
Email: kevindeangeli@utk.edu
Date: 2020-01-16

'''

import numpy as np

def crossEntropy(prediction, output):
    if output == 1:
      return -log(prediction)
    else:
      return -log(1 - prediction)

def crossEntropy_prime(prediction, output):
    return -((output/prediction) - ((1-output)/(1-prediction)))

def mse(prediction, output):
    if len(prediction) == 1:
        return (prediction[0]-output[0])**2
    else:
        return ((prediction - output)**2).mean(axis=0)

def mse_prime(output, target):
    return output-target




