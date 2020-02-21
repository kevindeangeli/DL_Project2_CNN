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

layer_options = ["ConvolutionalLayer", "MaxPoolingLayer", "FlattenLayer", "Dense"]


class ConvolutionalLayer():
    def __init__(self, input_dimension, kernels_number = 1, kernel_size = 2, activation_function = "sigmoid", learningRate=0.1, weight = None, bias= None, lossfunc = "mse"):
        self.kernels = []
        self.numberOfNeurons = (input_dimension[1]-kernel_size+1)**2 #### may not need this stored actually.
        self.kernel_size = kernel_size
        self.outputSize = (kernels_number,int(self.numberOfNeurons**.5),int(self.numberOfNeurons**.5) )
        #(numb_kernel, XdimensionOfKernel, YdimensionOfKernel)
        self.input = []
        self.kernels_number = kernels_number
        self.learningRate= learningRate
        self.deltas_received = []

        if weight is None:
            for i in range(kernels_number):
                newKernel = []
                #generate weights for this kernel.
                weight = np.array([np.random.normal(size=kernel_size) for i in range(kernel_size)])
                bias = np.random.normal()
                newKernel = [Neuron(inputLen=input_dimension, activationFun=activation_function,lossFunction=lossfunc ,learningRate=learningRate, weights=weight, bias = bias) for i in range(self.numberOfNeurons)]
                self.kernels.append(newKernel)
        else:
            for index in range(kernels_number):
                newKernel = [Neuron(inputLen=input_dimension, activationFun=activation_function,lossFunction=lossfunc ,learningRate=learningRate, weights=weight[index], bias = bias[index]) for i in range(self.numberOfNeurons)]
                self.kernels.append(newKernel)

    def calculate(self, input):
        self.input = input
        input = np.array(input)
        layerOutput = []
        total_output = []
        for kernel in self.kernels:
            layerOutput=[]
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
            total_output.append(np.reshape(layerOutput, (self.outputSize[1], self.outputSize[2])))
        return total_output

    def convolve(self,kernel, input):
        layerOutput=[]
        kernel_size = kernel.shape[0]
        outputSize = (input.shape[0] - kernel_size + 1)

        #kernels are a list of list containing neurons. Each list represents a filter
        i1, j1 = 0,0
        i2, j2 = kernel_size, kernel_size
        matrix_limit_index = input.shape[0]+1
        for i in range(outputSize**2):
            if j2 == matrix_limit_index:
                j2 = kernel_size
                j1 = 0
                i1 += 1
                i2 += 1
            #layerOutput.append(neuron.calculate(input[i1:i2,j1:j2]))
            layerOutput.append(np.sum(np.multiply(input[i1:i2,j1:j2], kernel)))
            j2 += 1
            j1 += 1
        total_output = np.reshape(layerOutput, (outputSize, outputSize))
        return total_output



    def deltaSum(self):
        filters = []
        deltas_out = []

        delta_receivedShape = self.deltas_received[0].shape[0]
        padding = self.kernel_size- 1

        newMatrix_Dim = delta_receivedShape+(padding)*2 #add the filter -1 *2 <-- That's be
        newMatrixes = []
        for index, kernel in enumerate(self.kernels):
            filters.append(np.rot90(np.rot90(kernel[0].weights)))
            mat = np.zeros((newMatrix_Dim,newMatrix_Dim))
            mat[padding:padding+delta_receivedShape,padding:padding+delta_receivedShape] = self.deltas_received[index]
            newMatrixes.append(mat)

        for index, kernel in enumerate(filters):
            deltas_out.append(self.convolve(kernel=kernel, input = newMatrixes[index]))

        return deltas_out
        #for filter in filters:




    def backpropagation(self, deltaMatrix):
        self.deltas_received = deltaMatrix
        outputs = []
        for index in range(self.kernels_number):
            outputs.append(self.convolve(kernel = np.array(deltaMatrix[index]), input=self.input))
        #You can't update the weights until after deltasum()

        for index2, kernel in enumerate(self.kernels):
            for neuron in kernel:
                neuron.newWeights = neuron.weights - self.learningRate * outputs[index2]

        self.deltaSum()


    def updateWeights(self):
        for kernel in self.kernels:
            for neuron in kernel:
                neuron.updateWeight()

class MaxPoolingLayer():
    def __init__(self, kernel_size, input_dimension):
        self.kernel_size = kernel_size
        self.input_dimension=input_dimension
        self.numberOfNeurons = int((((input_dimension[1]-kernel_size)/2)+1)**2) #### may not need this stored actually.
        self.outputSize = (input_dimension[0],int(self.numberOfNeurons**.5),int(self.numberOfNeurons**.5))
        self.maxIndexList = []
        self.output = [] #Stores the output for backpropagation

    def calculate(self, matrix_list):
    #input should be a list of matrices:
        matrixes_output = []
        max_index_matrix = []
        individual_outputs = []
        for matrix in matrix_list:
            individual_outputs = []
            max_index_matrix = []
            i1, j1 = 0, 0
            i2, j2 = self.kernel_size, self.kernel_size
            matrix_limit_index = matrix.shape[0] + 1
            for i in range(self.numberOfNeurons):
                if j2 >= matrix_limit_index:
                    j2 = self.kernel_size
                    j1 = 0
                    i1 += 2
                    i2 += 2
                max_val = np.max(matrix[i1:i2, j1:j2])
                individual_outputs.append(max_val)
                max_index= np.array(np.unravel_index(matrix[i1:i2, j1:j2].argmax(), (self.kernel_size,self.kernel_size)))
                max_index[0] += i1
                max_index[1] += j1
                max_index_matrix.append(max_index)
                j2 += 2
                j1 += 2
            self.maxIndexList.append(max_index_matrix)
            matrixes_output.append(np.reshape(individual_outputs,(self.outputSize[1],self.outputSize[2])))
        self.output = matrixes_output
        return matrixes_output

    def deltaSum(self, PreviousdeltaSum):
        newMatrices = [np.zeros((self.input_dimension[1], self.input_dimension[2])) for i in range(self.input_dimension[0])]
        numOfMatrices = len(self.output)
        counter_index_deltas = 0
        for i in range(numOfMatrices): #For each matrix
            for k in range(int(len(PreviousdeltaSum[0])/numOfMatrices)): #There are a certain amount of deltas
                mat_indx1= self.maxIndexList[i][k][0]
                mat_indx2= self.maxIndexList[i][k][1]
                newMatrices[i][mat_indx1][mat_indx2]=PreviousdeltaSum[0][counter_index_deltas]
                counter_index_deltas +=1

        return newMatrices #Matrix of Deltas in Max position with 0 felling

    def backpropagation(self, deltasArray):
        return

    def updateWeights(self):
        return


class FlattenLayer():
    def __init__(self,input_dimension):
        self.input_dimension = input_dimension
        self.outputSize = (1,int(input_dimension[0]*input_dimension[1]*input_dimension[2])) #(1,N) vector
        self.deltas = 0

    def calculate(self, matrix_list):
        output = []
        for matrix in matrix_list:
            output.append(matrix.flatten())
        output = np.array(output)
        return output.flatten()

    def deltaSum(self, PreviousdeltaSum):
        #self.deltas = PreviousdeltaSum
        return PreviousdeltaSum

    def backpropagation(self, deltas):
        return

    def updateWeights(self):
        return



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
        #self.activation = False #Variables tells you if the neuron contains an activation function

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
        self.output = sigmoid(np.sum(np.multiply(self.input,self.weights)) + self.bias)
        return self.output


    #The delta of the last layer is computed a little different, so it has its own function.
    def backpropagationLastLayer(self, target):
        self.delta = self.loss_prime(self.output, target) * self.activation_prime(self.output)
        self.newBias = self.bias - self.learnR*self.delta
        for index, PreviousNeuronOutput in enumerate(self.input):
            self.newWeights.append(self.weights[0][index] - self.learnR * self.delta * PreviousNeuronOutput)

    def backpropagation(self, sumDelta):
        #sumDelta will be computed at the layer level. Since it requires weights from multiple neurons.
        self.delta = sumDelta * self.activation_prime(self.output)
        self.newBias = self.bias - self.learnR * self.delta
        for index, PreviousNeuronOutput in enumerate(self.input):
            self.newWeights.append(self.weights[index] - self.learnR * self.delta * self.input[index])

    #Used to compute the summation of the Deltas for backprop.
    def mini_Delta(self, index):
        return self.delta * self.weights[index]



class FullyConnectedLayer():
    def __init__(self, inputLen, numOfNeurons = 5, activationFun = "sigmoid", lossFunction= "mse", learningRate = .1, weights = None, bias = None, activation = False):
        self.inputLen = inputLen
        self.neuronsNum = numOfNeurons
        self.activationFun = activationFun
        self.learningRate = learningRate
        self.weights = weights
        self.bias = bias
        self.layerOutput = []
        self.lossFunction = lossFunction
        self.activation = activation
        self.outputSize = (1,numOfNeurons)

        #Random weights or user defined weights:
        if weights is None:
            random_weights = np.array([np.random.normal(size=inputLen) for i in range(numOfNeurons)])
            random_bias = np.random.normal()
            self.neurons = [Neuron(inputLen=self.inputLen, activationFun=activationFun,lossFunction=self.lossFunction ,learningRate=self.learningRate, weights=random_weights, bias = random_bias) for i in range(numOfNeurons)]
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

        return self.layerOutput[0]

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

    def addLayer(self, layer_type, weights = None, bias = None,kernels_number=1, kernel_size= 2, activation_function="sigmoid", learning_rate =0.1, lossfunc = "mse", numOfNeurons= None ):
        if layer_type == "ConvolutionalLayer":
            layer = ConvolutionalLayer(input_dimension=self.inputList[-1], kernels_number = kernels_number, kernel_size = kernel_size, activation_function = activation_function, learningRate = self.learningRate, weight = weights, bias = bias, lossfunc = lossfunc)
            self.layers.append(layer)
            self.inputList.append(layer.outputSize)
        elif layer_type == "MaxPoolingLayer":
            layer = MaxPoolingLayer(kernel_size = kernel_size, input_dimension = self.inputList[-1])
            self.layers.append(layer)
            self.inputList.append(layer.outputSize)
        elif layer_type == "FlattenLayer":
            layer = FlattenLayer(self.inputList[-1])
            self.layers.append(layer)
            self.inputList.append(layer.outputSize)
            #output = (1, K) <-- index 1 is the number of neuron that goes into the dense layer.
        elif layer_type == "Dense":
            layer = FullyConnectedLayer(inputLen=self.inputList[-1][1], numOfNeurons = numOfNeurons, activationFun = "sigmoid", lossFunction= "mse", learningRate = self.learningRate, weights = weights, bias = bias,activation=True)
            self.layers.append(layer)
            self.inputList.append(layer.outputSize[1])
            #outputSize  = (1,K)
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
        self.layersNum = len(self.layers)
        self.layers[-1].backPropagateLast(target)
        layersCounter = self.layersNum+1

        for i in range(2,layersCounter):
            #Calculate the sum delta for the following layer to update the previous layer.

            if type(self.layers[-i + 1]) == FlattenLayer or type(self.layers[-i + 1]) == MaxPoolingLayer:
                deltaArr=self.layers[-i + 1].deltaSum(deltaArr)
                #continue
            else:
                deltaArr = self.layers[-i + 1].deltaSum()
            self.layers[-i].backpropagation(deltaArr)

        for layer in self.layers:
            layer.updateWeights()



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


def doExample1():
    '''
    This function does the "Example" forward and backpop pass required for the assignemnt.
    '''
    print( "--- Example 1---")
    # layer_options  = ["ConvolutionalLayer", "MaxPoolingLayer", "FlattenLayer", "Dense"]

    input = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
    weights3 = np.array([[[.1,.8,.2], [1,.1,0.5], [.4,.5,.7]]])
    bias3 = [.9]
    model = NeuralNetwork(inputSize=input.shape, learningRate=0.5, lossFunction="mse")
    model.addLayer(layer_type=layer_options[0],kernels_number=1, kernel_size= 3, weights = weights3, bias = bias3, activation_function="sigmoid", lossfunc = "mse" )
    model.addLayer(layer_type=layer_options[2])
    Newweights = [[[0.89286015,0.33197981,0.82122912,0.04169663,0.10765668,0.59505206,0.52981736,0.41880743,0.33540785]]]
    newBias = [[0.62251943]]
    model.addLayer(layer_type=layer_options[3], numOfNeurons=1, weights = Newweights, bias = newBias)
    print("Network Output: ", model.calculate(input))


def doExample2():
    '''
    This function does the "Example" forward and backpop pass required for the assignemnt.
    '''
    print( "--- Example 2---")
    # layer_options  = ["ConvolutionalLayer", "MaxPoolingLayer", "FlattenLayer", "Dense"]

    input = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
    weights3 = np.array([[[.1,.8,.2], [1,.1,0.5], [.4,.5,.7]]])
    bias3 = [.9]
    model = NeuralNetwork(inputSize=input.shape, learningRate=0.5, lossFunction="mse")
    model.addLayer(layer_type=layer_options[0],kernels_number=1, kernel_size= 3, weights = weights3, bias = bias3, activation_function="sigmoid", lossfunc = "mse" )
    print("Output 1 (Cov): ")
    weights3 = np.array([[[.1,.8,.2], [1,.1,0.5], [.4,.5,.7]]])
    model.addLayer(layer_type=layer_options[0],kernels_number=1, kernel_size= 3, weights = weights3, bias = bias3, activation_function="sigmoid", lossfunc = "mse" )


    model.addLayer(layer_type=layer_options[2])

    Newweights = [[[0.89286015]]]
    newBias = [[0.33197981]]
    model.addLayer(layer_type=layer_options[3], numOfNeurons=1, weights = Newweights, bias = newBias)
    print("Network Output: ", model.calculate(input))


def doExample3():
    input = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24],
                      [25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36]])

    weights3 = np.array([[[.5,.5,.5], [.5,.5,.5], [.5,.5,.5]],[[-.5,-.5,-.5], [-.5,-.5,-.5], [-.5,-.5,-.5]]])
    bias3 = [1,-1]

    print("Input: ")
    print(input)
    print("Weights: ")
    print(weights3)
    print(" ")

    #layer_options  = ["ConvolutionalLayer", "MaxPoolingLayer", "FlattenLayer", "Dense"]

    model = NeuralNetwork(inputSize=input.shape, learningRate=0.5, lossFunction="mse")

    #Pre-Defined weights:
    model.addLayer(layer_type=layer_options[0],kernels_number=2, kernel_size= 3, weights = weights3, bias = bias3, activation_function="sigmoid", lossfunc = "mse")

    print("Output 1 (Cov): ")
    out=model.layers[0].calculate(input)
    print(out)

    model.addLayer(layer_type=layer_options[1],kernel_size= 2)
    print("Output 2 (MaxP): ")
    out = model.layers[1].calculate(out)
    print(out)

    model.addLayer(layer_type=layer_options[2])
    print("Output 3 (Flatt): ")
    out = model.layers[2].calculate(out)
    print(out)


    Newweights = [[[0.89286015,0.33197981,0.82122912,0.04169663,0.10765668,0.59505206,0.52981736,0.41880743]]]
    newBias = [[0.33540785]]

    model.addLayer(layer_type=layer_options[3], numOfNeurons=1, weights = Newweights, bias = newBias, learning_rate=.5)
    print("Output 4: (Dense)")
    out = model.layers[3].calculate(out)
    print(out) #This output doesn't include the activation function

    model.backPropagate(target=[.5])





def DoEntireThing():
    input = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
    weights = np.array([[1,0,1], [0,1,0], [1,0,1]])
    weights2 = np.array([[1,1],[0,1]])

    weights3 = np.array([[.1,.8,.2], [1,.001,0.005], [.4,.5,.7]])
    bias3 = [.9]


    bias = [0]
    print("Input: ")
    print(input)
    print("Weights: ")
    print(weights)
    print(" ")

    layer_options  = ["ConvolutionalLayer", "MaxPoolingLayer", "FlattenLayer", "Dense"]

    model = NeuralNetwork(inputSize=input.shape, learningRate=0.1, lossFunction="mse")

    #Pre-Defined weights:
    model.addLayer(layer_type=layer_options[0],kernels_number=1, kernel_size= 3, weights = weights3, bias = bias3, activation_function="sigmoid", lossfunc = "mse" )
    #Random:
    #model.addLayer(layer_type=layer_options[0],kernels_number=1, kernel_size= 4,learning_rate =0.1, weights = None, bias = None, activation_function="sigmoid", lossfunc = "mse" )

    print("Output 1 (Cov): ")
    out=model.layers[0].calculate(input)
    print(out)

    model.addLayer(layer_type=layer_options[1],kernel_size= 2)
    print("Output 2 (MaxP): ")
    out = model.layers[1].calculate(out)
    print(out)

    model.addLayer(layer_type=layer_options[2])
    print("Output 3 (Flatt): ")
    out = model.layers[2].calculate(out)
    print(out)


    Newweights = [[[0.9670298,0.5472323,0.9726844,0.714816]]]
    newBias = [[0.6977288]]

    model.addLayer(layer_type=layer_options[3], numOfNeurons=1, weights = Newweights, bias = newBias, learning_rate=.5)
    print("Output 4: (Dense)")
    out = model.layers[3].calculate(out)
    print(out)

    #model.train(input=input, target=3)



def main():
    #doExample1()
    #doExample2()
    doExample3()

if __name__ == "__main__":
    main()


