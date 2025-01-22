import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np


class Viewer:
    """
    Handles visualization and user interaction for the AI Art Evolution project.
    """

    def __init__(self, controller):
        """
        Initializes the Viewer.

        Args:
            controller (ModelController): The main controller managing the simulation.
        """
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("AI Art Evolution")
        self.root.geometry("1000x700")

        # UI Components
        self.canvas_frame = ttk.Frame(self.root)
        self.controls_frame = ttk.Frame(self.root)
        self.feedback_frame = ttk.Frame(self.root)

        self._setup_ui()

    def _setup_ui(self):
        """
        Sets up the user interface layout and components.
        """
        # Canvas for displaying patterns
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas_grid = []

        # Controls
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y)
        self._setup_controls()

        # Feedback
        self.feedback_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self._setup_feedback()

    def _setup_controls(self):
        """
        Sets up user controls (start/stop/reset buttons, sliders, etc.).
        """
        ttk.Button(self.controls_frame, text="Start", command=self._start_simulation).pack(pady=10)
        ttk.Button(self.controls_frame, text="Stop", command=self._stop_simulation).pack(pady=10)
        ttk.Button(self.controls_frame, text="Reset", command=self._reset_simulation).pack(pady=10)

        ttk.Label(self.controls_frame, text="Mutation Rate:").pack(pady=5)
        self.mutation_slider = ttk.Scale(
            self.controls_frame, from_=0, to=1, orient=tk.HORIZONTAL, command=self._update_mutation_rate
        )
        self.mutation_slider.set(self.controller.mutation_rate)
        self.mutation_slider.pack(pady=5)

        ttk.Label(self.controls_frame, text="Crossover Rate:").pack(pady=5)
        self.crossover_slider = ttk.Scale(
            self.controls_frame, from_=0, to=1, orient=tk.HORIZONTAL, command=self._update_crossover_rate
        )
        self.crossover_slider.set(self.controller.crossover_rate)
        self.crossover_slider.pack(pady=5)

        ttk.Button(self.controls_frame, text="Export Images", command=self._export_images).pack(pady=10)
        ttk.Button(self.controls_frame, text="Export Video", command=self._export_video).pack(pady=10)

    def _setup_feedback(self):
        """
        Sets up feedback display (generation stats, fitness scores, etc.).
        """
        self.generation_label = ttk.Label(self.feedback_frame, text="Generation: 0")
        self.generation_label.pack(pady=10)

        self.average_fitness_label = ttk.Label(self.feedback_frame, text="Average Fitness: 0")
        self.average_fitness_label.pack(pady=10)

        self.best_fitness_label = ttk.Label(self.feedback_frame, text="Best Fitness: 0")
        self.best_fitness_label.pack(pady=10)

    def render_patterns(self, population):
        """
        Renders the current population as a grid of thumbnails.

        Args:
            population (list of Pattern): The current population.

        Returns:
            None
        """
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        grid_size = int(np.ceil(np.sqrt(len(population))))
        for i, pattern in enumerate(population):
            img = Image.fromarray((pattern.canvas * 255).astype(np.uint8))
            img = img.resize((100, 100))
            photo = ImageTk.PhotoImage(img)

            label = tk.Label(self.canvas_frame, image=photo)
            label.image = photo
            label.grid(row=i // grid_size, column=i % grid_size, padx=5, pady=5)

    def update_feedback(self, stats):
        """
        Updates feedback labels with the latest stats.

        Args:
            stats (dict): Simulation statistics (generation, fitness scores, etc.).

        Returns:
            None
        """
        self.generation_label.config(text=f"Generation: {stats['generation']}")
        self.average_fitness_label.config(text=f"Average Fitness: {stats['average_fitness']:.2f}")
        self.best_fitness_label.config(text=f"Best Fitness: {stats['best_fitness']:.2f}")

    def _start_simulation(self):
        self.controller.start_simulation(self.controller.evolution._neural_network_factory)

    def _stop_simulation(self):
        self.controller.stop_simulation()

    def _reset_simulation(self):
        self.controller.stop_simulation()
        self.controller.start_simulation(self.controller.evolution._neural_network_factory)

    def _update_mutation_rate(self, value):
        self.controller.update_parameters(mutation_rate=float(value))

    def _update_crossover_rate(self, value):
        self.controller.update_parameters(crossover_rate=float(value))

    def _export_images(self):
        population = self.controller.get_population()
        self.controller.export_evolution_data(format="images")

    def _export_video(self):
        self.controller.export_evolution_data(format="video")

    def run(self):
        """
        Starts the Tkinter main loop.

        Returns:
            None
        """
        self.root.mainloop()
