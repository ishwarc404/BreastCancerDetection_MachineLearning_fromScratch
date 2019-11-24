from numpy import exp, array, random, dot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("breast_cancer.csv")
df = df.drop(columns=["Unnamed: 0"]) #dropping the initial indexing column
df.iloc[:,10].replace(2, 0,inplace=True)
df.iloc[:,10].replace(4, 1,inplace=True)

df = df.astype(float)


names = df.columns[0:10]
scaler = MinMaxScaler() 
scaled_df = scaler.fit_transform(df.iloc[:,0:10]) 
scaled_df = pd.DataFrame(scaled_df, columns=names)


xtrain=scaled_df.iloc[0:500,1:10] #.values.transpose()
ytrain=df.iloc[0:500,10:] #.values.transpose()
xval=scaled_df.iloc[501:600,1:10] #.values.transpose()
yval=df.iloc[501:600,10:] #.values.transpose()
xtest=scaled_df.iloc[501:683,1:10] #.values.transpose()
ytest=df.iloc[501:683,10:] #.values.transpose()

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __ReLU(self,x):
        return x * (x > 0)

    def __dReLU(self,x):
        return 1. * (x > 0)

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__dReLU(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__dReLU(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__ReLU(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__ReLU(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print("    Layer 1 (4 neurons, each with 3 inputs):")
        print(self.layer1.synaptic_weights)
        print("    Layer 2 (1 neuron, with 4 inputs):")
        print(self.layer2.synaptic_weights)


if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    # Create layer 1 (4 neurons, each with 3 inputs)
    layer1 = NeuronLayer(8, 9)

    # Create layer 2 (a single neuron with 4 inputs)
    layer2 = NeuronLayer(1,8)


    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = xtrain.to_numpy() #array([[0, 0, 1, 0, 1, 1, 1, 0, 1], [0, 1, 0, 1, 0, 0, 1, 1, 1]])
    training_set_outputs = ytrain.to_numpy()#array([[0, 1]]).T

    print(training_set_inputs)
    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    
    neural_network.train(training_set_inputs, training_set_outputs, 90000)

    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    xtest = xtest.to_numpy()
    ytest = ytest.to_numpy()

    new_xtest = []
    new_ytest = []

    for i in xtest :
        new_xtest.append(list(i))

    for i in ytest :
        new_ytest.append(list(i))

    # Test the neural network with a new situation.

    # Test the neural network with a new situation.
    outputs = []
    print("Stage 3")

    for i in new_xtest:
        hidden_state, output = neural_network.think(array(i))
        print(output)

    
    #for i in range(0,len(new_xtest)):
    #new_xtest[0]
    # hidden_state, output = neural_network.think(array())
    # print(output)
        # if(output[0] < 1):
        #     outputs.append(0)
        # else:
        #     outputs.append(1)


    # count = 0
    # for i in range(len(outputs)):
    #     #print(outputs[i],ytest[i][0])
    #     if(outputs[i]==ytest[i][0]):
    #         count +=1
    # print(count,len(outputs))
    # print("Accuracy Percent:",(count/len(outputs))*100)