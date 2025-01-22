import random
import numpy as np

class NeuralNetwork:
    """
    Represents a simple neural network for generating patterns.
    """

    def __init__(self, input_size, hidden_layers, output_size):
        """
        Initializes the neural network with random weights and biases.

        Args:
            input_size (int): Number of input nodes.
            hidden_layers (list of int): List specifying the number of neurons in each hidden layer.
            output_size (int): Number of output nodes.
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        # Initialize weights and biases
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)]
        self.biases = [np.random.randn(size) for size in self.layers[1:]]

    def forward(self, input_data):
        """
        Performs a forward pass through the network.
    
        Args:
            input_data (numpy array): The input data for the neural network.
    
        Returns:
            numpy array: The output of the network.
        """
        if input_data is None:
            raise ValueError("Input data cannot be None. Ensure valid input is passed.")
        
        current = input_data
        for weight, bias in zip(self.weights, self.biases):
            current = self._activation(np.dot(current, weight) + bias)
        return current

    def mutate(self, mutation_rate):
        """
        Applies random mutations to the weights and biases.

        Args:
            mutation_rate (float): Probability of changing each weight/bias.

        Returns:
            None
        """
        for i in range(len(self.weights)):
            mutation_mask = np.random.rand(*self.weights[i].shape) < mutation_rate
            self.weights[i] += mutation_mask * np.random.randn(*self.weights[i].shape)

        for i in range(len(self.biases)):
            mutation_mask = np.random.rand(*self.biases[i].shape) < mutation_rate
            self.biases[i] += mutation_mask * np.random.randn(*self.biases[i].shape)

    def _activation(self, x):
        """
        Activation function for the neural network.

        Args:
            x (numpy array): Input to the activation function.

        Returns:
            numpy array: Output after applying the activation function.
        """
        return np.tanh(x)  # Use tanh for simplicity

    def __repr__(self):
        """
        Provides a string representation of the neural network structure.

        Returns:
            str: Description of the network.
        """
        return f"NeuralNetwork(Layers: {self.layers})"
