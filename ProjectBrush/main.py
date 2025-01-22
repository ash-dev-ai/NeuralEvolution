from control.model_controller import ModelController
from view.viewer import Viewer


def main():
    """
    Entry point for the AI Art Evolution application.
    Initializes and connects the components, then starts the application.
    """
    # Initialize the ModelController with default parameters
    population_size = 50
    mutation_rate = 0.1
    crossover_rate = 0.7

    controller = ModelController(
        population_size=population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate
    )

    # Initialize the Viewer and pass the controller to it
    viewer = Viewer(controller)

    # Start the Viewer (this will start the Tkinter event loop)
    viewer.run()


if __name__ == "__main__":
    main()
