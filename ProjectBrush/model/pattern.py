import numpy as np

class Pattern:
    def __init__(self, neural_network, metadata=None):
        """
        Initializes a pattern with a NEAT neural network.

        Args:
            neural_network (NeuralNetwork): NEAT-based neural network wrapper
            metadata (dict): Additional evolutionary context
        """
        self.neural_network = neural_network  # Correct attribute name
        self.metadata = metadata if metadata else {}
        self.fitness = 0
        self.canvas = None
        self.generate_pattern()

    def generate_pattern(self):
        try:
            # Get output from network
            output = self.neural_network.forward(None)  # Input handled internally
            
            # Ensure output is a valid array
            if not isinstance(output, np.ndarray):
                output = np.random.rand(10, 10)
                
            # Ensure valid output dimensions
            if output.size < 100:
                output = np.pad(output, (0, 100 - output.size))
                
            side_length = int(np.sqrt(output.size))
            self.canvas = output[:side_length**2].reshape(side_length, side_length)
            
            # Normalize to [0,1]
            self.canvas = (self.canvas - np.min(self.canvas)) / \
                         (np.max(self.canvas) - np.min(self.canvas) + 1e-7)
        except Exception as e:
            print(f"Pattern generation error: {e}")
            self.canvas = np.random.rand(10, 10)

    def evaluate_fitness(self, fitness_evaluator):
        """
        Evaluates pattern fitness using provided evaluator.
        """
        self.fitness = fitness_evaluator.evaluate_objective(self)

    def __repr__(self):
        """
        Provides pattern summary with NEAT-specific information.
        """
        network_info = self.neural_network.__repr__()
        return (f"Pattern(Fitness: {self.fitness:.2f}, "
                f"Canvas: {self.canvas.shape if self.canvas is not None else 'None'}, "
                f"Network: {network_info})")