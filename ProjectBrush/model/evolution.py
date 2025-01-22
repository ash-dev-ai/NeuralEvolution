import random
from model.pattern import Pattern
from model.fitness_evaluator import FitnessEvaluator

class Evolution:
    """
    Manages the evolutionary process for generating and evolving patterns.
    """

    def __init__(self, population_size, mutation_rate, crossover_rate, fitness_evaluator):
        """
        Initializes the Evolution manager.

        Args:
            population_size (int): Number of patterns in each generation.
            mutation_rate (float): Probability of mutation.
            crossover_rate (float): Probability of crossover.
            fitness_evaluator (FitnessEvaluator): The fitness evaluation object.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.fitness_evaluator = fitness_evaluator
        self.population = []
        self.generation_count = 0

    def initialize_population(self, neural_network_factory):
        """
        Creates the initial random population of patterns.
        """
        self.population = [
            Pattern(neural_network_factory()) for _ in range(self.population_size)
        ]
        for pattern in self.population:
            pattern.generate_pattern()

    def evaluate_population(self):
        """
        Evaluates the fitness of all patterns in the population.

        Returns:
            None
        """
        for pattern in self.population:
            objective_score = self.fitness_evaluator.evaluate_objective(pattern)
            subjective_score = 0  # Placeholder for user feedback
            pattern.fitness = self.fitness_evaluator.combine_scores(objective_score, subjective_score)

    def select_parents(self):
        """
        Selects parent patterns based on fitness using roulette wheel selection.

        Returns:
            tuple: Two parent patterns.
        """
        total_fitness = sum(p.fitness for p in self.population)
        if total_fitness == 0:
            # If all fitness scores are zero, select randomly
            return random.sample(self.population, 2)
        
        probabilities = [p.fitness / total_fitness for p in self.population]
        parent1 = random.choices(self.population, weights=probabilities, k=1)[0]
        parent2 = random.choices(self.population, weights=probabilities, k=1)[0]
        return parent1, parent2

    def crossover(self, parent1, parent2):
        """
        Combines the neural networks of two parents to produce an offspring.

        Args:
            parent1 (Pattern): The first parent pattern.
            parent2 (Pattern): The second parent pattern.

        Returns:
            Pattern: The offspring pattern.
        """
        offspring_network = parent1.neural_network  # Start with a copy of parent1's network
        if random.random() < self.crossover_rate:
            for i in range(len(offspring_network.weights)):
                # Blend weights from both parents
                offspring_network.weights[i] = (parent1.neural_network.weights[i] + parent2.neural_network.weights[i]) / 2
            for i in range(len(offspring_network.biases)):
                # Blend biases from both parents
                offspring_network.biases[i] = (parent1.neural_network.biases[i] + parent2.neural_network.biases[i]) / 2
        return Pattern(offspring_network)

    def mutate(self, pattern):
        """
        Applies mutation to a pattern's neural network.

        Args:
            pattern (Pattern): The pattern to mutate.

        Returns:
            None
        """
        pattern.mutate(self.mutation_rate)

    def generate_next_generation(self):
        """
        Creates the next generation of patterns using selection, crossover, and mutation.

        Returns:
            None
        """
        new_population = []

        # Elitism: Keep the top-performing pattern
        self.population.sort(key=lambda p: p.fitness, reverse=True)
        new_population.append(self.population[0])  # Add the best pattern

        # Generate the rest of the new population
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            offspring = self.crossover(parent1, parent2)
            self.mutate(offspring)
            new_population.append(offspring)

        self.population = new_population
        self.generation_count += 1

    def get_population(self):
        """
        Returns the current population.

        Returns:
            list of Pattern: The current population.
        """
        return self.population

    def __repr__(self):
        """
        Provides a string representation of the evolution state.

        Returns:
            str: Description of the current generation.
        """
        return f"Evolution(Generation: {self.generation_count}, Population Size: {self.population_size})"
