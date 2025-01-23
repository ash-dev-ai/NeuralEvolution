import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.cm as cm

class Viewer:
    """
    Handles visualization and user interaction for the AI Art Evolution project.
    """

    def __init__(self, controller, root):
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
        """Sets up the user interface layout"""
        # Scrollable canvas setup
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

        # Event bindings
        self.root.bind("<Configure>", self._resize_canvas)

        # Initialize components
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y)
        self._setup_controls()
        self.feedback_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self._setup_feedback()

    def _setup_controls(self):
        """Sets up control buttons"""
        ttk.Button(self.controls_frame, text="Start", command=self._start_simulation).pack(pady=10)
        ttk.Button(self.controls_frame, text="Stop", command=self._stop_simulation).pack(pady=10)
        ttk.Button(self.controls_frame, text="Reset", command=self._reset_simulation).pack(pady=10)
        ttk.Button(self.controls_frame, text="Export Images", command=self._export_images).pack(pady=10)
        ttk.Button(self.controls_frame, text="Export Video", command=self._export_video).pack(pady=10)

    def _setup_feedback(self):
        """Sets up statistics display"""
        self.generation_label = ttk.Label(self.feedback_frame, text="Generation: 0")
        self.generation_label.pack(pady=10)
        self.average_fitness_label = ttk.Label(self.feedback_frame, text="Average Fitness: 0")
        self.average_fitness_label.pack(pady=10)
        self.best_fitness_label = ttk.Label(self.feedback_frame, text="Best Fitness: 0")
        self.best_fitness_label.pack(pady=10)

    def render_patterns(self, population):
        """Renders population patterns"""
        try:
            for widget in self.canvas_grid_frame.winfo_children():
                widget.destroy()
        except RuntimeError:
            pass

        grid_size = int(np.ceil(np.sqrt(len(population))))
        thumbnail_size = max(100, self.canvas_frame.winfo_width() // (grid_size + 3))

        for i, pattern in enumerate(population):
            if pattern.canvas is not None:
                try:
                    normalized = (pattern.canvas - np.min(pattern.canvas)) / (np.max(pattern.canvas) - np.min(pattern.canvas) + 1e-7)
                    colored = (cm.viridis(normalized)[:, :, :3] * 255).astype(np.uint8)
                    img = Image.fromarray(colored).resize((thumbnail_size, thumbnail_size))
                    photo = ImageTk.PhotoImage(img)
                    label = tk.Label(self.canvas_grid_frame, image=photo)
                    label.image = photo
                    label.grid(row=i//grid_size, column=i%grid_size, padx=5, pady=5)
                except Exception as e:
                    print(f"Rendering error: {e}")
            else:
                placeholder = tk.Label(self.canvas_grid_frame, text="No Image", bg="gray", width=15, height=10)
                placeholder.grid(row=i//grid_size, column=i%grid_size, padx=5, pady=5)

        self.canvas_grid_frame.update_idletasks()

    def _resize_canvas(self, event):
        """Handles window resizing"""
        current_size = (self.root.winfo_width(), self.root.winfo_height())
        if current_size == self.last_window_size:
            return
            
        self.last_window_size = current_size
        if self.resize_debounce_timer:
            self.root.after_cancel(self.resize_debounce_timer)
        self.resize_debounce_timer = self.root.after(200, self._safe_render)

    def _safe_render(self):
        """Safe pattern re-rendering"""
        self.root.unbind("<Configure>")
        try:
            self.render_patterns(self.controller.get_population())
        finally:
            self.root.bind("<Configure>", self._resize_canvas)

    def update_ui(self, stats):
        """Updates all UI elements"""
        self.generation_label.config(text=f"Generation: {stats['generation']}")
        self.average_fitness_label.config(text=f"Avg Fitness: {max(stats['average_fitness'], 0):.2f}")
        self.best_fitness_label.config(text=f"Best Fitness: {max(stats['best_fitness'], 0):.2f}")
        self.render_patterns(self.controller.get_population())
        self.root.update_idletasks()

    # Controller interaction methods
    def _start_simulation(self):
        self.controller.start_simulation()
        self.render_patterns(self.controller.get_population())

    def _stop_simulation(self):
        self.controller.stop_simulation()

    def _reset_simulation(self):
        self.controller.reset_simulation()
        self.render_patterns(self.controller.get_population())
        self.update_ui({"generation": 0, "average_fitness": 0, "best_fitness": 0})

    def _export_images(self):
        self.controller.export_evolution_data(format="images")

    def _export_video(self):
        self.controller.export_evolution_data(format="video")

    def run(self):
        self.root.mainloop()