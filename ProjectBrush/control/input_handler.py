class InputHandler:
    """
    Handles user input and passes it to the ModelController.
    """

    def __init__(self, controller):
        """
        Initializes the InputHandler.

        Args:
            controller (ModelController): The main controller managing the simulation.
        """
        self.controller = controller

    def process_feedback(self, feedback):
        """
        Processes user feedback for pattern fitness.

        Args:
            feedback (dict): A dictionary where keys are pattern IDs and values are user ratings (0-1).

        Returns:
            None
        """
        # Validate feedback
        for pattern_id, rating in feedback.items():
            if not (0 <= rating <= 1):
                raise ValueError(f"Invalid rating {rating} for pattern {pattern_id}. Must be between 0 and 1.")

        # Forward feedback to the controller
        print(f"Processing user feedback: {feedback}")
        for pattern_id, rating in feedback.items():
            pattern = self._get_pattern_by_id(pattern_id)
            if pattern:
                subjective_score = self.controller.fitness_evaluator.evaluate_subjective(rating)
                pattern.fitness = self.controller.fitness_evaluator.combine_scores(
                    pattern.fitness, subjective_score
                )

    def update_parameters(self, mutation_rate=None, crossover_rate=None, population_size=None):
        """
        Updates simulation parameters based on user input.

        Args:
            mutation_rate (float, optional): New mutation rate (0-1).
            crossover_rate (float, optional): New crossover rate (0-1).
            population_size (int, optional): New population size (>0).

        Returns:
            None
        """
        # Validate parameters
        if mutation_rate is not None and not (0 <= mutation_rate <= 1):
            raise ValueError(f"Invalid mutation rate: {mutation_rate}. Must be between 0 and 1.")
        if crossover_rate is not None and not (0 <= crossover_rate <= 1):
            raise ValueError(f"Invalid crossover rate: {crossover_rate}. Must be between 0 and 1.")
        if population_size is not None and population_size <= 0:
            raise ValueError(f"Invalid population size: {population_size}. Must be greater than 0.")

        # Forward parameters to the controller
        self.controller.update_parameters(mutation_rate, crossover_rate, population_size)

    def handle_command(self, command):
        """
        Handles user commands such as 'start', 'stop', or 'reset'.

        Args:
            command (str): The command to execute.

        Returns:
            None
        """
        command = command.lower()
        if command == "start":
            self.controller.start_simulation(self._neural_network_factory)
        elif command == "stop":
            self.controller.stop_simulation()
        elif command == "reset":
            self.controller.stop_simulation()
            self.controller.start_simulation(self._neural_network_factory)
        else:
            raise ValueError(f"Unknown command: {command}")

    def _get_pattern_by_id(self, pattern_id):
        """
        Retrieves a pattern by its ID.

        Args:
            pattern_id (int): The ID of the pattern to retrieve.

        Returns:
            Pattern: The corresponding pattern, or None if not found.
        """
        population = self.controller.get_population()
        if 0 <= pattern_id < len(population):
            return population[pattern_id]
        print(f"Pattern ID {pattern_id} not found.")
        return None

    def _neural_network_factory(self):
        """
        A placeholder method to create random neural networks.

        Returns:
            NeuralNetwork: A randomly initialized neural network.
        """
        from model.neural_network import NeuralNetwork
        return NeuralNetwork(input_size=10, hidden_layers=[16, 16], output_size=100)
