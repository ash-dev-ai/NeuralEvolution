from model.evolution import Evolution
from model.fitness_evaluator import FitnessEvaluator
from model.pattern import Pattern
from control.input_handler import InputHandler
from control.output_handler import OutputHandler
from model.neural_network import NeuralNetwork
import neat
import os

class ModelController:
    """
    Manages NEAT-based simulation lifecycle and model-view communication.
    """

    def __init__(self, root, config_path="config/neat-config.ini"):
        """
        Initializes NEAT-based ModelController.

        Args:
            root (tk.Tk): Root window for scheduling updates
            config_path (str): Path to NEAT configuration file
        """
        self.root = root
        self.config_path = config_path
        
        # Initialize core components
        self.fitness_evaluator = FitnessEvaluator()
        
        # Load NEAT configuration
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        # Initialize NEAT population
        self.population = neat.Population(self.config)
        
        # Initialize handlers
        self.input_handler = InputHandler(self)
        self.output_handler = OutputHandler()
        
        # Simulation state
        self.current_generation = 0
        self.is_running = False
        self.viewer = None

    def start_simulation(self):
        """
        Starts the NEAT evolutionary process.
        """
        if not self.is_running:
            self.is_running = True
            print("NEAT simulation started.")
            self._run_generations()

    def _run_generations(self):
        """
        Runs generations continuously while simulation is active.
        """
        if self.is_running:
            self.population.run(self.evaluate_genomes, 1)
            self.current_generation += 1
            
            # Update UI and schedule next generation
            stats = self.get_generation_statistics()
            if self.viewer:
                self.root.after(0, lambda: self.viewer.update_ui(stats))
                self.root.after(1000, self._run_generations)  # 1 second between gens

    def evaluate_genomes(self, genomes, config):
        """
        NEAT-compatible fitness evaluation function.
        """
        for genome_id, genome in genomes:
            try:
                neural_net = NeuralNetwork(genome, self.config)
                pattern = Pattern(neural_net)
                pattern.generate_pattern()
                genome.fitness = self.fitness_evaluator.evaluate_objective(pattern)
            except Exception as e:
                print(f"Genome evaluation failed: {e}")
                genome.fitness = 0.0

    def stop_simulation(self):
        """
        Stops the simulation.
        """
        self.is_running = False
        print("Simulation stopped.")

    def reset_simulation(self):
        """
        Resets the simulation to initial state.
        """
        self.stop_simulation()
        self.output_handler.reset()
        self.population = neat.Population(self.config)
        self.current_generation = 0
        print("Simulation reset. Current generation: 0")

    def get_population(self):
        """
        Retrieves current population as Pattern objects.
        """
        population = []
        for genome in self.population.population.values():
            neural_net = NeuralNetwork(genome, self.config)  # Use wrapper
            pattern = Pattern(neural_net)
            pattern.fitness = genome.fitness if genome.fitness else 0.0
            if pattern.canvas is None:
                pattern.generate_pattern()
            population.append(pattern)
        return population

    def get_generation_statistics(self):
        """
        Provides statistics about the current generation.
        """
        population = self.get_population()
        fitness_values = [p.fitness if p.fitness is not None else 0.0 for p in population]
        return {
            "generation": self.current_generation,
            "average_fitness": sum(fitness_values)/len(fitness_values),
            "best_fitness": max(fitness_values) if fitness_values else 0,
            "population_size": len(population)
        }

    def process_user_feedback(self, feedback):
        """
        Processes user ratings for interactive evolution.
        """
        population = self.population.population
        for idx, rating in feedback.items():
            if 0 <= idx < len(population):
                genome_id = list(population.keys())[idx]
                population[genome_id].fitness = rating
        print("Updated fitness based on user feedback")

    def export_evolution_data(self, format="video"):
        """
        Handles data export through OutputHandler.
        """
        self.output_handler.export(self.population, format)

    def __repr__(self):
        """
        Provides NEAT simulation status.
        """
        return (f"ModelController(NEAT)\n"
                f"Generations: {self.current_generation}\n"
                f"Population: {len(self.get_population())} patterns\n"
                f"Best Fitness: {self.get_generation_statistics()['best_fitness']:.2f}")