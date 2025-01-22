---

# **Project Brush**

## **Overview**
**Project Brush** is an AI-driven art evolution application that generates and evolves patterns or artworks through neural networks and evolutionary algorithms. Users can interact with the application by rating the generated patterns, adjusting evolution parameters, and exporting results as images or videos.

The application follows the **Model-View-Controller (MVC)** architecture, ensuring a clean separation of logic, UI, and control flow. It is built using Python with a focus on object-oriented programming principles.

---

## **Features**
- **Pattern Generation**: Neural networks create initial patterns based on random inputs.
- **Evolutionary Algorithm**: Patterns evolve over generations, guided by user feedback or predefined fitness functions.
- **User Interaction**:
  - Rate individual patterns to influence the evolution process.
  - Adjust mutation rate, crossover rate, and population size in real time.
- **Export Options**:
  - Save patterns as images.
  - Export the entire evolution process as a video.
- **Dynamic UI**:
  - Scrollable, resizable canvas for displaying patterns.
  - Real-time updates during evolution.

---

## **Project Structure**
The project is organized into the following directories and files:

```
ProjectBrush/
│
├── model/
│   ├── evolution.py         # Manages evolutionary processes
│   ├── fitness_evaluator.py # Evaluates the fitness of patterns
│   ├── neural_network.py    # Encapsulates the neural network logic
│   ├── pattern.py           # Represents a single artwork
│
├── control/
│   ├── model_controller.py  # Manages the simulation lifecycle
│   ├── input_handler.py     # Processes user input
│   ├── output_handler.py    # Handles exporting and logging data
│
├── view/
│   ├── viewer.py            # Manages the UI and visualization
│
├── main.py                  # Entry point for running the application
├── requirements.txt         # Lists Python dependencies
└── README.md                # Project overview and instructions
```

---

## **Installation**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/project-brush.git
   cd project-brush
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv brush_env
   source brush_env/bin/activate  # On Windows: brush_env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**
1. **Run the Application**:
   ```bash
   python main.py
   ```

2. **Interact with the Application**:
   - Click "Start" to begin the simulation.
   - Adjust sliders to modify mutation and crossover rates.
   - Use the "Reset" button to restart the simulation.
   - Rate patterns to guide the evolution process.

3. **Export Results**:
   - Click "Export Images" to save the current generation as images.
   - Click "Export Video" to save the evolution process as a video.

---

## **Customization**
- **Fitness Function**:
  - Modify `fitness_evaluator.py` to define new criteria for evaluating patterns.
- **Neural Network**:
  - Update `neural_network.py` to change the network architecture or activation functions.
- **Evolution Parameters**:
  - Adjust default values for mutation rate, crossover rate, and population size in `model_controller.py`.

---

## **Dependencies**
The application requires the following Python libraries:
- `numpy`
- `Pillow`
- `matplotlib`
- `tkinter` (built-in with Python)
- `ffmpeg` (for video export, ensure it’s installed on your system)

---

## **Future Enhancements**
- Implement additional fitness functions (e.g., symmetry, texture complexity).
- Support for saving and loading evolution states.
- Enhance the UI for more intuitive interaction.
- Add more complex neural network architectures.

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for details.

---

## **Contributing**
Contributions are welcome! Feel free to fork the repository, create a new branch, and submit a pull request.

---

## **Contact**
For questions or feedback, please contact:
- **Ash Mal**: [ash.maldonado.ai@gmail.com](mailto:ash.maldonado.ai@gmail.com)

---

Let me know if you’d like to customize this further!