# Importing necessary libraries
import numpy
import scipy.special
import matplotlib.pyplot

# Defining the neural network class
class neuralNetwork:

    # Initializing the neural network
    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes1 = hiddennodes1
        self.hnodes2 = hiddennodes2
        self.onodes = outputnodes
        self.lr = learningrate
        
        # Weight initialization
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes1, self.inodes))
        self.whh = numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes2, self.hnodes1))
        self.who = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.onodes, self.hnodes2))

        # Bias initialization
        self.bih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes1, 1))
        self.bhh = numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes2, 1))
        self.bho = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.onodes, 1))

        # Activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    # Training the neural network
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs1 = numpy.dot(self.wih, inputs) + self.bih
        hidden_outputs1 = self.activation_function(hidden_inputs1)

        hidden_inputs2 = numpy.dot(self.whh, hidden_outputs1) + self.bhh
        hidden_outputs2 = self.activation_function(hidden_inputs2)

        final_inputs = numpy.dot(self.who, hidden_outputs2) + self.bho
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors2 = numpy.dot(self.who.T, output_errors)
        hidden_errors1 = numpy.dot(self.whh.T, hidden_errors2)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs2))
        self.bho += self.lr * numpy.sum(output_errors * final_outputs * (1.0 - final_outputs), axis=1, keepdims=True)
        
        self.whh += self.lr * numpy.dot((hidden_errors2 * hidden_outputs2 * (1.0 - hidden_outputs2)),
                                        numpy.transpose(hidden_outputs1))
        self.bhh += self.lr * numpy.sum(hidden_errors2 * hidden_outputs2 * (1.0 - hidden_outputs2), axis=1, keepdims=True)
        
        self.wih += self.lr * numpy.dot((hidden_errors1 * hidden_outputs1 * (1.0 - hidden_outputs1)),
                                        numpy.transpose(inputs))
        self.bih += self.lr * numpy.sum(hidden_errors1 * hidden_outputs1 * (1.0 - hidden_outputs1), axis=1, keepdims=True)

    # Querying the neural network
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs1 = numpy.dot(self.wih, inputs) + self.bih
        hidden_outputs1 = self.activation_function(hidden_inputs1)

        hidden_inputs2 = numpy.dot(self.whh, hidden_outputs1) + self.bhh
        hidden_outputs2 = self.activation_function(hidden_inputs2)

        final_inputs = numpy.dot(self.who, hidden_outputs2) + self.bho
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # Method to save weights and biases to file
    def save_parameters(self, filename_prefix):
        numpy.savetxt(f"out/{filename_prefix}_wih.dat", self.wih, delimiter=",", newline=",\n\r")
        numpy.savetxt(f"out/{filename_prefix}_whh.dat", self.whh, delimiter=",", newline=",\n\r")
        numpy.savetxt(f"out/{filename_prefix}_who.dat", self.who, delimiter=",", newline=",\n\r")
        numpy.savetxt(f"out/{filename_prefix}_bih.dat", self.bih, delimiter=",", newline=",\n\r")
        numpy.savetxt(f"out/{filename_prefix}_bhh.dat", self.bhh, delimiter=",", newline=",\n\r")
        numpy.savetxt(f"out/{filename_prefix}_bho.dat", self.bho, delimiter=",", newline=",\n\r")


def main():
    input_nodes = 784
    hidden_nodes1 = 64
    hidden_nodes2 = 32
    output_nodes = 10
    learning_rate = 0.16
    n = neuralNetwork(input_nodes, hidden_nodes1, hidden_nodes2, output_nodes, learning_rate)

    training_data_file = open("data/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = 15
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

    # Save the weights and biases after training
    n.save_parameters("mnist")

    test_data_file = open("data/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        scorecard.append(1 if label == correct_label else 0)

    scorecard_array = numpy.asarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)


if __name__ == "__main__":
    main()
