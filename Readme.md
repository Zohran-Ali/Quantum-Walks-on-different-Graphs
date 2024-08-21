
# Quantum and Classical Random Walks on Graphs

This project simulates quantum and classical random walks on various types of graphs (cyclic, 2D grids, and random graphs). It provides visualization tools to compare the evolution of probability distributions over time for both types of walks.

## Features

- **Quantum Random Walk**: Simulates quantum walks using adjacency matrices and unitary evolution.
- **Classical Random Walk**: Simulates classical random walks using transition matrices.
- **Graph Types Supported**:
  - Cyclic graphs
  - 2D grid graphs
  - Random graphs
- **Visualization**: Visualize the probability distribution evolution on the graph over time and compare quantum vs classical walks.

## Installation

To get started with this project, you'll need to clone the repository and install the necessary dependencies.

### Clone the repository

```bash
git clone https://github.com/yourusername/quantum-classical-walks.git
cd quantum-classical-walks


Install dependencies
 You can install the required Python packages using pip:

Ensure you have the following packages installed:

pip install -r requirements.txt

numpy
matplotlib
networkx
scipy
ipywidgets (for interactive visualizations)
tkinter 

Running the Simulation

To run the simulation and visualize the walks, you can use the provided Python scripts or Jupyter Notebooks.

Jupyter Notebook:

Open the provided Jupyter Notebook Basics_of_QWs/Gneralized_QWs.ipynb
Run the cells in sequence to generate the graphs and visualize the walks.

Python Script:

Run the script Basics_of_QWs/Quantum-and-Classical.py to perform a pre-configured simulation.
Here you can choose and alter the parameters inside the script


Parameters
Graph Type: Select the type of graph (Cyclic Graph, 2D Grid, Random).
Time Evolution: Control the time evolution of the walk using a slider or play widget in the Jupyter Notebook.
Initial State: Customize the initial state of the walker.
Visiualisation:This has been carried out using the tkinter library to mak a GUI for the user to dyamically interact and visualize the graphs this can be found in the Quantum-Walks/Tkinter.py. The user can choose from different types of initial states ( which are defined inside a function wit certain fixd parameters).From different graphs with desired number of vertices. Note that for a 2D graph its hould be a perfect square. A normal implementation can be found in Quantum-Walks/function-for-Gui.ipynb where the same code is implememnted but without the GUI. Hee a general function takes in parameters and can run the code. 



### Notes:

1. **Project Name**: "Quantum-Walks-on-different-Graphs" 
2. **GitHub URL**: `https://github.com/Zohran-Ali/Quantum-Walks-on-different-Graphs  

Save this content as `README.md` in the root of your project directory.
