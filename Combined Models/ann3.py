from numpy import exp, array, random, dot
##getting the data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("breast_cancer.csv")
df = df.drop(columns=["Unnamed: 0"]) #dropping the initial indexing column
df.iloc[:,10].replace(2, 0,inplace=True)
df.iloc[:,10].replace(4, 1,inplace=True)

df = df.astype(float)


def dataset_minmax(dataset):
    print("comeshere")
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


names = df.columns[0:10]
scaler = MinMaxScaler() 
scaled_df = scaler.fit_transform(df.iloc[:,0:10]) 
scaled_df = pd.DataFrame(scaled_df, columns=names)
print(scaled_df)


# scaled_df = pd.DataFrame(scaled_df, columns=names)


xtrain=scaled_df.iloc[0:512,1:10] #.values.transpose()
ytrain=df.iloc[0:512,10:] #.values.transpose()

# xval=scaled_df.iloc[501:600,1:10] #.values.transpose()
# yval=df.iloc[501:600,10:] #.values.transpose()

xtest=scaled_df.iloc[513:683,1:10] #.values.transpose()
ytest=df.iloc[513:683,10:] #.values.transpose()

###################################################################################


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
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        y = self.__sigmoid(x)
        return y * (1 - y)
    

    def __ReLU(self,x):
        return x * (x > 0)

    def __dReLU(self,x):
        return 1. * (x > 0)

        

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print ("    Layer 1 (8 neurons, each with 9 inputs): ")
        print (self.layer1.synaptic_weights)
        print ("    Layer 2 (1 neuron, with 8 inputs):")
        print (self.layer2.synaptic_weights)

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    # Create layer 1 (8 neurons, each with 9 inputs)
    #this initiliases the numpy weight matrix of dimension 9x8 (the initial weights of all the links)
    #this is the first hidden layer which takes in 9 inputs
    layer1 = NeuronLayer(8, 9)

    # Create layer 2 (a single neuron with 8 inputs)
    #this last layer #again here the cons
    layer2 = NeuronLayer(1, 8)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = xtrain.to_numpy()
    training_set_outputs = ytrain.to_numpy()
    #training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    #training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs,100)

    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print("Stage 3) Considering a new situation: ")
    xtest = xtest.to_numpy()
    ytest = ytest.to_numpy()
    count = 0


    for i in range(0,len(xtest)):
        hidden_state, output = neural_network.think((xtest[i]))
        print("Actual Output:",ytest[i][0])
        print("Model Output:",output[0])

        # if((output[0] ==  0.5)):
        #     count = count + 1
        # # #look into this once.
        if((output[0] ==  0.5)and ytest[i][0]==0 or (output[0] >  0.5 and ytest[i][0]==1) ):
            count = count + 1

    # #hidden_state, output = neural_network.think(array([1, 1, 0]))
    print("Matched Values:",count)
    print("Actual Values:",len(xtest))
    print("Accuracy:",count*100/len(xtest))



