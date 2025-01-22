import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
# Add this import at the top of viewer.py
import matplotlib.cm as cm

class Viewer:
    """
    Handles visualization and user interaction for the AI Art Evolution project.
    """

    def __init__(self, controller, root):
        """
        Initializes the Viewer.

        Args:
            controller (ModelController): The main controller managing the simulation.
        """
        self.controller = controller
        self.root = root
        controller.viewer = self
        self.root.title("Brush")
        self.root.geometry("1000x700")

        # UI Components
        self.canvas_frame = ttk.Frame(self.root)
        self.controls_frame = ttk.Frame(self.root)
        self.feedback_frame = ttk.Frame(self.root)
        self.resize_debounce_timer = None
        self.last_window_size = (0, 0)

        self._setup_ui()

    def _setup_ui(self):
        """
        Sets up the user interface layout and components.
        """
        # Scrollable canvas for displaying patterns
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.canvas_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas_widget = tk.Canvas(self.canvas_frame, yscrollcommand=self.canvas_scrollbar.set)
        self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_scrollbar.config(command=self.canvas_widget.yview)

        self.canvas_grid_frame = ttk.Frame(self.canvas_widget)
        self.canvas_widget.create_window((0, 0), window=self.canvas_grid_frame, anchor="nw")
        self.canvas_grid_frame.bind(
            "<Configure>",
            lambda e: self.canvas_widget.configure(scrollregion=self.canvas_widget.bbox("all"))
        )

        # Bind resize event
        self.root.bind("<Configure>", lambda e: self._resize_canvas(e))

        # Controls and Feedback
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y)
        self._setup_controls()

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
        # Safer widget cleanup
        try:
            for widget in self.canvas_grid_frame.winfo_children():
                widget.destroy()
        except RuntimeError:  # Handle concurrent modification
            pass
    
        # Determine grid size and thumbnail size
        grid_size = int(np.ceil(np.sqrt(len(population))))
        thumbnail_size = max(100, self.canvas_frame.winfo_width() // (grid_size + 2))
    
        for i, pattern in enumerate(population):
            if pattern.canvas is not None:
                try:
                    # Normalize and apply colormap
                    normalized_canvas = (pattern.canvas - np.min(pattern.canvas)) / \
                                       (np.max(pattern.canvas) - np.min(pattern.canvas) + 1e-7)
                    
                    # Convert to RGB using matplotlib's colormap
                    colored_canvas = (cm.viridis(normalized_canvas)[:, :, :3] * 255).astype(np.uint8)
                    
                    # Resize and convert to PhotoImage
                    img = Image.fromarray(colored_canvas)
                    img = img.resize((thumbnail_size, thumbnail_size))
                    photo = ImageTk.PhotoImage(img)
                    
                    # Display image
                    label = tk.Label(self.canvas_grid_frame, image=photo)
                    label.image = photo  # Prevent garbage collection
                    label.grid(row=i // grid_size, column=i % grid_size, padx=5, pady=5)
                    
                except Exception as e:
                    print(f"Error rendering pattern {i}: {e}")
            else:
                # Placeholder for patterns without a canvas
                placeholder = tk.Label(self.canvas_grid_frame, text="No Image", bg="gray", width=15, height=7)
                placeholder.grid(row=i // grid_size, column=i % grid_size, padx=5, pady=5)

        self.canvas_grid_frame.update_idletasks()
        
    def _resize_canvas(self, event):
        """Handle window resize with debouncing"""
        # Get current window size
        current_size = (self.root.winfo_width(), self.root.winfo_height())
        
        # Only update if size actually changed
        if current_size == self.last_window_size:
            return
            
        self.last_window_size = current_size
        
        # Cancel previous debounce timer
        if self.resize_debounce_timer:
            self.root.after_cancel(self.resize_debounce_timer)
            
        # Schedule render after 200ms delay
        self.resize_debounce_timer = self.root.after(200, self._safe_render)

    def _safe_render(self):
        """Safely re-render patterns without recursion"""
        # Temporarily unbind the resize event
        self.root.unbind("<Configure>")
        
        try:
            self.render_patterns(self.controller.get_population())
        finally:
            # Re-bind the event after rendering
            self.root.bind("<Configure>", self._resize_canvas)

    def update_ui(self, stats):
        """
        Updates the UI with the latest generation statistics and re-renders patterns.
    
        Args:
            stats (dict): A dictionary containing generation statistics (e.g., generation, average_fitness, best_fitness).
        """
        # Update generation and fitness labels
        self.generation_label.config(text=f"Generation: {stats['generation']}")
        self.average_fitness_label.config(text=f"Average Fitness: {max(stats['average_fitness'], 0):.2f}")
        self.best_fitness_label.config(text=f"Best Fitness: {max(stats['best_fitness'], 0):.2f}")
    
        # Re-render patterns
        self.render_patterns(self.controller.get_population())
    
        # Force immediate GUI refresh
        self.root.update_idletasks()
    
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
        """
        Starts the simulation and renders the initial population.
        """
        self.controller.start_simulation()
        self.render_patterns(self.controller.get_population())

    def _stop_simulation(self):
        """
        Stops the simulation.
        """
        self.controller.stop_simulation()

    def _reset_simulation(self):
        """
        Resets the simulation and clears the GUI.
        """
        self.controller.reset_simulation()
        self.render_patterns(self.controller.get_population())
        self.update_feedback({
            "generation": 0,
            "average_fitness": 0,
            "best_fitness": 0,
        })

    def _rate_pattern(self, index, rating):
        """
        Rates a specific pattern and updates fitness.

        Args:
            index (int): The index of the pattern to rate.
            rating (float): The rating value.

        Returns:
            None
        """
        feedback = {index: rating}
        self.controller.process_user_feedback(feedback)

    def _update_mutation_rate(self, value):
        """
        Updates the mutation rate based on slider input.

        Args:
            value (str): The slider value.
        """
        self.controller.update_parameters(mutation_rate=float(value))

    def _update_crossover_rate(self, value):
        """
        Updates the crossover rate based on slider input.

        Args:
            value (str): The slider value.
        """
        self.controller.update_parameters(crossover_rate=float(value))

    def _export_images(self):
        """
        Exports the current generation as images.
        """
        self.controller.export_evolution_data(format="images")

    def _export_video(self):
        """
        Exports the evolution process as a video.
        """
        self.controller.export_evolution_data(format="video")

    def run(self):
        """
        Starts the Tkinter main loop.

        Returns:
            None
        """
        self.root.mainloop()
