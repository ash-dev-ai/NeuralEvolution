from model.evolution import Evolution
from model.fitness_evaluator import FitnessEvaluator
from model.pattern import Pattern
from control.input_handler import InputHandler
from control.output_handler import OutputHandler
from model.neural_network import NeuralNetwork

class ModelController:
    """
    Manages the simulation lifecycle and communication between the Model and View.
    """

    def __init__(self, root, population_size=50, mutation_rate=0.1, crossover_rate=0.7):
        self.root = root
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

    def _neural_network_factory(self):
        """
        Factory method to create random NeuralNetwork instances.

        Returns:
            NeuralNetwork: A new randomly initialized neural network.
        """
        return NeuralNetwork(input_size=10, hidden_layers=[16, 16], output_size=100)

    def start_simulation(self):
        """
        Starts the simulation by initializing the population and setting the running flag.
    
        Returns:
            None
        """
        self.evolution.initialize_population(self._neural_network_factory)
        self.is_running = True
        print(f"Simulation started with {self.population_size} patterns.")
        print(f"Initial population: {[p.canvas.shape if p.canvas is not None else None for p in self.evolution.get_population()]}")
        self._run_generations()

    def _run_generations(self):
        """
        Continuously runs generations while the simulation is active.
        """
        if self.is_running:
            stats = self.run_generation_cycle()
            if self.viewer:
                self.root.after(1000, self._run_generations)

    def run_generation_cycle(self):
        """
        Runs a single generation cycle (evaluation, selection, crossover, mutation).
        """
        if not self.is_running:
            return
        
        self.evolution.evaluate_population()
        self.evolution.generate_next_generation()
        self.current_generation += 1
        stats = self.get_generation_statistics()
        
        # Update UI after stats are calculated
        if self.viewer:
            self.root.after(0, lambda: self.viewer.update_ui(stats))  # Force UI refresh
        return stats
        
    def stop_simulation(self):
        """
        Stops the simulation.

        Returns:
            None
        """
        self.is_running = False
        print("Simulation stopped.")

    def reset_simulation(self):
        """
        Resets the simulation state, clearing the population and restarting.
        """
        self.stop_simulation()
        self.output_handler.reset()
        self.evolution.initialize_population(self._neural_network_factory)
        self.current_generation = 0  # Reset generation count
        print("Simulation reset. Current generation: 0")

    def run_generation_cycle(self):
        if not self.is_running:
            print("Simulation is not running.")
            return
        
        # Ensure this gets called periodically
        self.evolution.evaluate_population()
        self.evolution.generate_next_generation()
        self.current_generation += 1

        # Fetch statistics for the current generation
        stats = self.get_generation_statistics()

        # Update the viewer if available
        if hasattr(self, "viewer"):
            self.viewer.render_patterns(self.get_population())
            self.viewer.update_feedback(stats)

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
            "average_fitness": sum(fitness_values) / len(fitness_values) if fitness_values else 0,
            "best_fitness": max(fitness_values) if fitness_values else 0,
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
