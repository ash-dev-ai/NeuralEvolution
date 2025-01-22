import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class OutputHandler:
    """
    Handles exporting and logging data for the evolution process.
    """

    def __init__(self, export_dir="exports"):
        """
        Initializes the OutputHandler.

        Args:
            export_dir (str): Directory to save exported files.
        """
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

    def export_images(self, population, generation):
        """
        Exports the current population as images.

        Args:
            population (list of Pattern): The population to export.
            generation (int): The current generation number.

        Returns:
            None
        """
        generation_dir = os.path.join(self.export_dir, f"generation_{generation}")
        os.makedirs(generation_dir, exist_ok=True)

        for i, pattern in enumerate(population):
            image_path = os.path.join(generation_dir, f"pattern_{i}.png")
            self._save_image(pattern.canvas, image_path)

        print(f"Exported generation {generation} as images to {generation_dir}.")

    def export_video(self, evolution, fps=5):
        """
        Exports the evolution process as a video.

        Args:
            evolution (Evolution): The evolution object containing population history.
            fps (int): Frames per second for the video.

        Returns:
            None
        """
        video_path = os.path.join(self.export_dir, "evolution_video.mp4")
        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            population = evolution.get_population()
            best_pattern = max(population, key=lambda p: p.fitness)
            ax.imshow(best_pattern.canvas, cmap="viridis")
            ax.set_title(f"Generation: {frame}, Fitness: {best_pattern.fitness:.2f}")

        ani = FuncAnimation(fig, update, frames=evolution.generation_count, repeat=False)
        ani.save(video_path, fps=fps, writer="ffmpeg")
        print(f"Exported evolution video to {video_path}.")

    def log_statistics(self, stats, generation):
        """
        Logs statistics for the current generation.

        Args:
            stats (dict): A dictionary containing statistics (e.g., average fitness, best fitness).
            generation (int): The current generation number.

        Returns:
            None
        """
        log_file = os.path.join(self.export_dir, "evolution_log.txt")
        with open(log_file, "a") as f:
            log_entry = f"Generation {generation}: {stats}\n"
            f.write(log_entry)

        print(f"Logged statistics for generation {generation}.")

    def _save_image(self, canvas, path):
        """
        Saves a single pattern canvas as an image.

        Args:
            canvas (numpy array): The canvas to save.
            path (str): The path to save the image.

        Returns:
            None
        """
        plt.imsave(path, canvas, cmap="viridis")

    def _ensure_directory(self, path):
        """
        Ensures a directory exists.

        Args:
            path (str): The directory path.

        Returns:
            None
        """
        if not os.path.exists(path):
            os.makedirs(path)
