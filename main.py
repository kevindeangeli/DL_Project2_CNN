'''
Created by: Kevin De Angeli
Email: kevindeangeli@utk.edu
Date: 2020-02-06

Note: The activation and loss function are in two separate files
which will be included in the zip file.

General description:


'''


from errorFunctions import * #I put the error functions and their derivatives in a different file.
from activation import *     #I put the activation functions and their derivatives in a different file.

import numpy as np
import sys
import matplotlib.pyplot as plt


class ConvolutionalLayer():
    def __init__(self, input_dimension, kernels_number = 1, kernel_size = 2, activation_function = "sigmoid", learningRate=0.1, weight = None, bias= None, lossfunc = "mse"):
        self.kernels = []
        self.numberOfNeurons = (input_dimension-kernel_size+1)**2 #### may not need this stored actually.
        self.kernel_size = kernel_size
        self.outputSize = int(self.numberOfNeurons **.5)


        if weight is None:
            for i in range(kernels_number):
                newKernel = []
                #generate weights for this kernel.
                weight = [np.random.random_sample() for i in range(kernel_size**2)]
                #bias = np.random.random_sample()
                bias = 0
                newKernel = [Neuron(inputLen=input_dimension, activationFun=activation_function,lossFunction=lossfunc ,learningRate=learningRate, weights=weight, bias = bias) for i in range(self.numberOfNeurons)]
                self.kernels.append(newKernel)
        else:
            for i in range(kernels_number):
                newKernel = [Neuron(inputLen=input_dimension, activationFun=activation_function,lossFunction=lossfunc ,learningRate=learningRate, weights=weight, bias = bias) for i in range(self.numberOfNeurons)]
                self.kernels.append(newKernel)

    def calculate(self, input):
        layerOutput = []
        total_output = []
        for kernel in self.kernels:
            #kernels are a list of list containing neurons. Each list represents a filter
            i1, j1 = 0,0
            i2, j2 = self.kernel_size, self.kernel_size
            matrix_limit_index = input.shape[0]+1
            for neuron in kernel:
                if j2 == matrix_limit_index:
                    j2 = self.kernel_size
                    j1 = 0
                    i1 += 1
                    i2 += 1
                layerOutput.append(neuron.calculate(input[i1:i2,j1:j2]))
                j2 += 1
                j1 += 1
            matrix_out_dim = int(len(layerOutput) ** .5) #Given the assumption that matrix is square
            self.outputSize = matrix_out_dim
            total_output.append(np.reshape(layerOutput, (matrix_out_dim, matrix_out_dim)))
        return total_output



class MaxPoolingLayer():
    def __init__(self, kernel_size, input_dimension):
        self.kernel_size = kernel_size
        self.input_dimension=input_dimension
        self.numberOfNeurons = (input_dimension-kernel_size+1)**2 #### may not need this stored actually.
        self.outputSize = int(self.numberOfNeurons**.5)

    def calculate(self, matrix_list):
    #input should be a list of matrices:
        matrixes_output = []
        individual_outputs = []
        for matrix in matrix_list:
            individual_outputs = []
            i1, j1 = 0, 0
            i2, j2 = self.kernel_size, self.kernel_size
            matrix_limit_index = matrix.shape[0] + 1
            for i in range(self.numberOfNeurons):
                if j2 == matrix_limit_index:
                    j2 = self.kernel_size
                    j1 = 0
                    i1 += 1
                    i2 += 1
                individual_outputs.append(np.max(matrix[i1:i2, j1:j2]))
                j2 += 1
                j1 += 1
            matrixes_output.append(np.reshape(individual_outputs,(self.outputSize,self.outputSize)))
        return matrixes_output


class FlattenLayer():
    def __init__(self):
        x=0







class Neuron():
    #If bias == weights == None: Then random initial
    def __init__(self,inputLen, activationFun = "sigmoid", lossFunction="mse" , learningRate = .5, weights = None, bias = None):
        self.inputLen = inputLen
        self.learnR = learningRate
        self.activationFunction = activationFun
        self.lossFunction = lossFunction

        self.output = None #Saving the output of the neuron (after feedforward) makes things easy for backprop
        self.input = None   #Saves the input to the neuron for backprop.
        self.newWeights = [] #Saves new weight but it doesn't update until the end of backprop. (method: updateWeight)
        self.newBias = None
        self.delta = None #individual deltas required for backprop.

        self.weights  = weights
        #self.weights = np.reshape(weights, (weights.shape[0]**2))
        self.bias = bias

        #this series of if statement define the activation and loss functions, and their derivatives.
        if activationFun is "sigmoid":
            self.activate = sigmoid
            self.activation_prime = sigmoid_prime
        else:
            self.activate = linear

        if lossFunction is "mse":
            self.loss = mse
            self.loss_prime= mse_prime
        else:
            self.loss = crossEntropy
            self.loss_prime = crossEntropy_prime

    #The following pictures will be defined based on the parameters
    #that is passed to the object.
    def activate(self):
        pass
    def loss(self):
        pass
    def activation_prime(self):
        pass
    def loss_prime(self):
        pass

    #This function is called after backpropagation.
    def updateWeight(self):
        self.weights = self.newWeights
        self.newWeights = []
        self.bias = self.newBias
        self.newBias = None

    def calculate(self, input):
        '''
        Given an input, it will calculate the output
        :return:
        '''
        #newInput = np.reshape(input, (self.weights.shape[0]))
        self.input = input
        a = input
        b = self.weights
        c = np.multiply(a,b)
        d = np.sum(c)

        self.output = np.sum(np.multiply(self.input,self.weights)) + self.bias
        #self.output = np.dot(newInput,self.weights) + self.bias #Is this correct ? Yes, right? because max is not affected by bias, since bias is applied to all of them.
        #self.output = self.activate(np.dot(input,self.weights) + self.bias)
        return self.output

    #The delta of the last layer is computed a little different, so it has its own function.
    def backpropagationLastLayer(self, target):
        self.delta = self.loss_prime(self.output, target) * self.activation_prime(self.output)
        x1=  self.loss_prime(self.output, target)
        x2= self.activation_prime(self.output)
        self.newBias = self.bias - self.learnR*self.delta
        for index, PreviousNeuronOutput in enumerate(self.input):
            self.newWeights.append(self.weights[index] - self.learnR * self.delta * PreviousNeuronOutput)

    def backpropagation(self, sumDelta):
        #sumDelta will be computed at the layer level. Since it requires weights from multiple neurons.
        self.delta = sumDelta * self.activation_prime(self.output)
        self.newBias = self.bias - self.learnR * self.delta
        for index, PreviousNeuronOutput in enumerate(self.input):
            self.newWeights.append(self.weights[index] - self.learnR * self.delta * self.input[index])

    #Used to compute the sumation of the Deltas for backprop.
    def mini_Delta(self, index):
        return self.delta * self.weights[index]



class FullyConnectedLayer():
    def __init__(self, inputLen, numOfNeurons = 5, activationFun = "sigmoid", lossFunction= "mse", learningRate = .1, weights = None, bias = None):
        self.inputLen = inputLen
        self.neuronsNum = numOfNeurons
        self.activationFun = activationFun
        self.learningRate = learningRate
        self.weights = weights
        self.bias = bias
        self.layerOutput = []
        self.lossFunction = lossFunction

        #Random weights or user defined weights:
        if weights is None:
            self.neurons = [Neuron(inputLen=self.inputLen, activationFun=activationFun,lossFunction=self.lossFunction ,learningRate=self.learningRate, weights=self.weights) for i in range(numOfNeurons)]
        else:
            self.neurons = [Neuron(inputLen=self.inputLen, activationFun=activationFun,lossFunction=self.lossFunction, learningRate=self.learningRate, weights=self.weights[i], bias= self.bias[i]) for i in range(numOfNeurons)]


    def calculate(self, input):
        '''
        Will calculate the output of all the neurons in the layer.
        :return:
        '''
        self.layerOutput = []
        for neuron in self.neurons:
            self.layerOutput.append(neuron.calculate(input))

        return self.layerOutput

    def backPropagateLast(self, target):
        for targetIndex, neuron in enumerate(self.neurons):
            neuron.backpropagationLastLayer(target=target[targetIndex])

    def updateWeights(self):
        for neuron in self.neurons:
            neuron.updateWeight()

    #Computes the sum of the deltas times their weights based on the number of neurons in the previous layer.
    def deltaSum(self):
        delta_sumArr  = []
        x=len(self.neurons[0].weights)
        for i in range(len(self.neurons[0].weights)): #Number of Weights in the RightLayer = Number of neurons in the LeftLayer
            delta_sum = 0
            for index, neuron in enumerate(self.neurons):
                delta_sum += neuron.mini_Delta(i)
            delta_sumArr.append(delta_sum)
        return delta_sumArr

    def backpropagation(self, deltaArr):
        #Each neuron needs a delta to update their weights:
        for index, neuron in enumerate(self.neurons):
            neuron.backpropagation(deltaArr[index])


class NeuralNetwork():
    def __init__(self, inputSize, lossFunction = "mse", learningRate = .1):
        self.inputSize   = inputSize
        self.lossFunction = lossFunction
        self.learningRate = learningRate
        self.layers = []
        self.inputList = []

        self.inputList.append(inputSize)

    def addLayer(self, layer_type, weights = None, bias = None,kernels_number=1, kernel_size= 2, activation_function="sigmoid", learning_rate =0.1, lossfunc = "mse" ):
        if layer_type == "ConvolutionalLayer":
            layer = ConvolutionalLayer(input_dimension=self.inputList[-1], kernels_number = kernels_number, kernel_size = kernel_size, activation_function = activation_function, learningRate = learning_rate, weight = weights, bias = bias, lossfunc = lossfunc)
            self.layers.append(layer)
            self.inputList.append(layer.outputSize)
        elif layer_type == "MaxPoolingLayer":
            layer = MaxPoolingLayer(kernel_size = kernel_size, input_dimension = self.inputList[-1])
            self.layers.append(layer)
            self.inputList.append(layer.outputSize)
        elif layer_type == "FlattenLayer":
            x=0
        else:
            print("addLayer only accepts three types: ConvolutionalLayer,MaxPoolingLayer, or FlattenLayer ")
            return 0




    def showWeights(self):
        #Function which just goes through each neuron in each layer and displays the weights.
        inputLenLayer = self.inputLen
        for i in range(self.layersNum):
            print(" ")
            for k in range(self.neuronsNum[i]):
                print(self.layers[i].neurons[k].weights)

            inputLenLayer = self.neuronsNum[i]

    def showBias(self):
        #Function which just goes through each neuron in each layer and displays the bias.
        inputLenLayer = self.inputLen
        for i in range(self.layersNum):
            #print(" ")
            for k in range(self.neuronsNum[i]):
                print(self.layers[i].neurons[k].bias)

            inputLenLayer = self.neuronsNum[i]

    def calculate(self, input):
        '''
        given an input calculates the output of the network.
        input should be a list.
        :return:
        '''
        output = input
        for layer in self.layers:
            output = layer.calculate(output)

        return output

    def backPropagate(self, target):
        self.layers[-1].backPropagateLast(target)
        layersCounter = self.layersNum+1

        for i in range(2,layersCounter):
            #Calculate the sum delta for the following layer to update the previous layer.
            deltaArr = self.layers[-i + 1].deltaSum()
            self.layers[-i].backpropagation(deltaArr)

        for layer in self.layers:
            layer.updateWeights()



    def calculateLoss(self,input,target, function = "mse"):
        '''
        Given an input and desired output, calculate the loss.
        Can be implemented with MSE and binary cross.
        '''
        N = len(input)
        output = self.calculate(input)
        if function == "mse":
            error = mse(output, target)
        else:
            crossEntropy(output, target)

        return error


    def train(self, input, target, showLoss = False):
        '''
        Basically, do forward and backpropagation all together here.
        Given a single input and desired output, it will take one step of gradient descent.
        :return:
        '''
        self.calculate(input)
        if showLoss is True:
            print("mse: ", self.calculateLoss(input=input, target=target))
        self.backPropagate(target)


def doExample():
    '''
    This function does the "Example" forward and backpop pass required for the assignemnt.
    '''
    print( "--- Example ---")

    #Let's try the class example by setting the bias and weights:
    Newweights = [[[.15,.20], [.25, .30]], [[.40, .45], [.5, .55]]]
    newBias = [[.35,.35],[.6,.6]]
    model = NeuralNetwork(neuronsNum=[2, 2, 2], activationVector=['sigmoid', 'sigmoid'], lossFunction="mse",
                          learningRate=.5, weights=Newweights, bias = newBias)


    print("Original weights and biases of the network: ")
    print("Model's Weights:")
    model.showWeights()
    print("\nModel's Bias:")
    model.showBias()


    print("\nForward pass: ")
    print(model.calculate([.05,.1]))

    #model.train(input= [.05,.1], target=[.01, .99]) #you could use just this function to do all at once.
    model.backPropagate(target= [.01, .99])
    print("\nAfter BackProp, the updated weights are:")
    print("Model's Weights:")
    model.showWeights()
    print("\nModel's Bias:")
    model.showBias()



def main():
    input = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
    weights = np.array([[1,0,1], [0,1,0], [1,0,1]])
    weights2 = np.array([[1,1],[0,1]])
    bias = [0]
    print("Input: ")
    print(input)
    print("Weights: ")
    print(weights)
    print(" ")

    layer_options  = ["ConvolutionalLayer", "MaxPoolingLayer", "FlattenLayer"]

    #Problem X: kernel_size not required if weights is given
    model = NeuralNetwork(inputSize=5, learningRate=0.1, lossFunction="mse")
    model.addLayer(layer_type=layer_options[0],kernels_number=1, kernel_size= 2,learning_rate =0.1, weights = weights2, bias = bias, activation_function="sigmoid", lossfunc = "mse" )

    print("Output 1: ")
    print(model.layers[0].calculate(input))
    out=model.layers[0].calculate(input)

    print("Output 2: ")
    model.addLayer(layer_type=layer_options[1],kernel_size= 2)
    print(model.layers[1].calculate(out))



    model.addLayer(layer_type=layer_options[1],kernel_size= 2)




    #model.addLayer(layer_type=layer_options[1], kernel_size=)

    # program_name = sys.argv[0]
    # input = sys.argv[1:] #Get input from the console.
    # # Input validation:
    # if len(input) != 1:
    #     print("Input only one of these: example, and, or xor")
    #     return 0

    # This is just to run it from the editor instead of the console.
    #input = ["example", "and", "xor","lossLearning", "lossEpoch"]
    #input = [input[3]]

    # if input[0] == "example":
    #     doExample()
    # elif input[0] == "and":
    #     doAnd()
    # elif input[0] == "xor":
    #     doXor()
    # elif input[0] == "lossLearning":
    #     learningRateArr = np.linspace(0.1, 12, num=50)
    #     showLoss(learningRateArr, data="and")
    # elif input[0] == "lossEpoch":
    #     lossVSEpoch(data="and")
    # else:
    #     # Input validation
    #     print("Input Options: example, and, or xor")
    #     return 0



if __name__ == "__main__":
    main()
