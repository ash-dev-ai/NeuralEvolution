from control.model_controller import ModelController
from view.viewer import Viewer
import tkinter as tk
import os

def main():
    # Create root window first
    root = tk.Tk()
    
    # Initialize controller with NEAT configuration
    config_path = os.path.join("config", "neat-config.ini")
    
    try:
        controller = ModelController(
            root=root,
            config_path=config_path
        )
    except FileNotFoundError:
        print(f"NEAT configuration file not found at: {config_path}")
        print("Please create a 'config' directory with 'neat-config.ini'")
        return

    # Initialize viewer with controller reference
    viewer = Viewer(controller, root=root)
    
    # Link viewer to controller
    controller.viewer = viewer
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()