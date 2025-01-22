import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


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
        self._ensure_directory(self.export_dir)

    def reset(self):
        """
        Resets the output handler state for a new simulation run.

        Returns:
            None
        """
        if os.path.exists(self.export_dir):
            for file in os.listdir(self.export_dir):
                file_path = os.path.join(self.export_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    for sub_file in os.listdir(file_path):
                        os.remove(os.path.join(file_path, sub_file))
                    os.rmdir(file_path)
        self._ensure_directory(self.export_dir)
        print("OutputHandler reset.")

    def export(self, evolution, format):
        """
        Exports evolution data.

        Args:
            evolution (Evolution): The evolution object.
            format (str): The export format ("images" or "video").

        Returns:
            None
        """
        if format == "images":
            self.export_images(evolution.get_population(), evolution.generation_count)
        elif format == "video":
            self.export_video(evolution)
        else:
            raise ValueError(f"Unsupported export format: {format}")

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
        self._ensure_directory(generation_dir)

        for i, pattern in enumerate(population):
            if pattern.canvas is not None:
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
            if population:
                best_pattern = max(population, key=lambda p: p.fitness)
                if best_pattern.canvas is not None:
                    ax.imshow(best_pattern.canvas, cmap="viridis")
                    ax.set_title(f"Generation: {frame}, Fitness: {best_pattern.fitness:.2f}")
                else:
                    ax.text(0.5, 0.5, "No valid canvas", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "No population", ha="center", va="center")

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
        if canvas is not None:
            plt.imsave(path, canvas, cmap="viridis")
        else:
            print(f"Skipped saving image to {path}: canvas is None.")

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
