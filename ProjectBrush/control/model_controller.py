from model.evolution import Evolution
from model.fitness_evaluator import FitnessEvaluator
from model.pattern import Pattern
from controller.input_handler import InputHandler
from controller.output_handler import OutputHandler

class ModelController:
    """
    Manages the simulation lifecycle and communication between the Model and View.
    """

    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.7):
        """
        Initializes the ModelController with default or user-specified parameters.

        Args:
            population_size (int): Number of patterns in the population.
            mutation_rate (float): Probability of mutation.
            crossover_rate (float): Probability of crossover.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Initialize the FitnessEvaluator and Evolution classes
        self.fitness_evaluator = FitnessEvaluator()
        self.evolution = Evolution(
            population_size=self.population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            fitness_evaluator=self.fitness_evaluator
        )

        # Handlers for input and output
        self.input_handler = InputHandler(self)
        self.output_handler = OutputHandler()

        # Track the simulation state
        self.current_generation = 0
        self.is_running = False

    def start_simulation(self, neural_network_factory):
        """
        Starts the simulation by initializing the population and setting the running flag.

        Args:
            neural_network_factory (callable): A function to create random neural networks.

        Returns:
            None
        """
        self.evolution.initialize_population(neural_network_factory)
        self.is_running = True
        print(f"Simulation started with {self.population_size} patterns.")

    def stop_simulation(self):
        """
        Stops the simulation.

        Returns:
            None
        """
        self.is_running = False
        print("Simulation stopped.")

    def run_generation_cycle(self):
        """
        Executes a single generation cycle.

        Returns:
            dict: Statistics about the current generation.
        """
        if not self.is_running:
            print("Simulation is not running.")
            return

        # Evaluate the current population
        self.evolution.evaluate_population()

        # Generate the next generation
        self.evolution.generate_next_generation()
        self.current_generation += 1

        # Fetch statistics for the current generation
        stats = self.get_generation_statistics()
        print(f"Generation {self.current_generation} completed: {stats}")
        return stats

    def get_population(self):
        """
        Retrieves the current population.

        Returns:
            list of Pattern: The current population.
        """
        return self.evolution.get_population()

    def get_generation_statistics(self):
        """
        Provides statistics about the current generation.

        Returns:
            dict: A dictionary with stats like average fitness and best fitness.
        """
        fitness_values = [p.fitness for p in self.evolution.get_population()]
        return {
            "generation": self.current_generation,
            "average_fitness": sum(fitness_values) / len(fitness_values),
            "best_fitness": max(fitness_values),
            "population_size": len(fitness_values)
        }

    def update_parameters(self, mutation_rate=None, crossover_rate=None, population_size=None):
        """
        Updates the evolution parameters dynamically.

        Args:
            mutation_rate (float, optional): New mutation rate.
            crossover_rate (float, optional): New crossover rate.
            population_size (int, optional): New population size.

        Returns:
            None
        """
        if mutation_rate is not None:
            self.evolution.mutation_rate = mutation_rate
        if crossover_rate is not None:
            self.evolution.crossover_rate = crossover_rate
        if population_size is not None:
            self.population_size = population_size
            self.evolution.population_size = population_size

        print(f"Parameters updated: mutation_rate={mutation_rate}, crossover_rate={crossover_rate}, population_size={population_size}")

    def process_user_feedback(self, feedback):
        """
        Processes user feedback for fitness evaluation.

        Args:
            feedback (dict): User ratings for specific patterns.

        Returns:
            None
        """
        self.input_handler.process_feedback(feedback)

    def export_evolution_data(self, format="video"):
        """
        Exports evolution data via the OutputHandler.

        Args:
            format (str): The export format, e.g., "video", "images".

        Returns:
            None
        """
        self.output_handler.export(self.evolution, format)
