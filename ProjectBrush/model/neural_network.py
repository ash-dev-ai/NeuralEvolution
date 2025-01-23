import numpy as np
import neat

class NeuralNetwork:
    """
    Represents a NEAT-optimized neural network for generating patterns.
    (Now wraps NEAT genome instead of manual weight management)
    """

    def __init__(self, genome, config):
        """
        Initializes neural network from NEAT genome.

        Args:
            genome (neat.DefaultGenome): NEAT genome object
            config (neat.Config): NEAT configuration
        """
        self.genome = genome
        self.config = config
        self.network = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Track network structure for compatibility
        self.input_size = config.genome_config.num_inputs
        self.output_size = config.genome_config.num_outputs
        self.layers = self._get_network_structure()

    def forward(self, input_data):
        """NEAT-compatible forward pass with spatial awareness"""
        try:
            # Generate coordinate grid
            size = int(np.sqrt(self.config.genome_config.num_outputs))
            x = np.linspace(0, 1, size)
            y = np.linspace(0, 1, size)
            xx, yy = np.meshgrid(x, y)
            
            # Process spatial coordinates through network
            output = []
            for x_coord, y_coord in zip(xx.flatten(), yy.flatten()):
                activation = self.network.activate([x_coord, y_coord])
                output.append(activation[0])
                
            return np.array(output).reshape(size, size)
        except Exception as e:
            print(f"Forward error: {e}")
            return np.zeros((10, 10))

    def _get_network_structure(self):
        """
        Extracts network structure from genome for visualization.
        
        Returns:
            list: Layer sizes [input, hidden..., output]
        """
        layers = [self.input_size]
        node_ids = sorted(self.genome.nodes.keys())
        
        # Count hidden nodes (exclude input/output)
        input_range = range(0, -self.input_size, -1)
        output_range = range(1, self.output_size+1)
        hidden_nodes = [n for n in node_ids if n not in input_range and n not in output_range]
        
        if hidden_nodes:
            layers.append(len(hidden_nodes))
        layers.append(self.output_size)
        
        return layers

    def __repr__(self):
        """
        Provides string representation of evolved network.
        """
        return (f"NEATNetwork(Inputs: {self.input_size}, "
                f"Hidden: {len(self.layers)-2}, "
                f"Outputs: {self.output_size}, "
                f"Connections: {len(self.genome.connections)})")