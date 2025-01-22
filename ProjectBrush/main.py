from control.model_controller import ModelController
from view.viewer import Viewer
import tkinter as tk

def main():
    root = tk.Tk()
    controller = ModelController(
        root=root,
        population_size=10,
        mutation_rate=0.1,
        crossover_rate=0.7
    )
    viewer = Viewer(controller, root)
    root.mainloop()

if __name__ == "__main__":
    main()