# %%
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.linalg import expm
from ipywidgets import interact, widgets, FloatSlider
from IPython.display import display
import cmath
from Function import create_adjacency_matrices, Gaussian_initial_state,evolve_2d_graph, superposition_initial_state, localized_initial_state,coherent_state




# %%
# Example code to define a 2D graph using netorkx library 
G = nx.grid_2d_graph(4, 4)  # 4x4 grid

# printing the adjacency list
for line in nx.generate_adjlist(G):
    print(line)
# write edgelist to grid.edgelist
nx.write_edgelist(G, path="grid.edgelist", delimiter=":")
# read edgelist from grid.edgelist
H = nx.read_edgelist(path="grid.edgelist", delimiter=":")

nx.draw(H)
plt.show()

# %%

#Global Parameters to change the type of graph and number of vertices, 
#The create adjacency matrices function returns the correspodning adjacency matrix and the graph
N=49
graph_type = '2D Grid'
H, _ = create_adjacency_matrices(N, graph_type)
_,G= create_adjacency_matrices(N, graph_type)

# Initial state: walker starts at vertex 0
initial_state = localized_initial_state(N, 24)
P = nx.to_numpy_array(G)
P = P / np.sum(P, axis=1, keepdims=True)  # Ensure rows sum to 1

time_range = np.linspace(0, 30, 1000)


# Classical random walk function
def classical_random_walk_2d(P, initial_state, t):
    """
    Simulate a classical random walk on a graph and record probability distribution over time.

    Parameters:
    P (numpy.ndarray): Transition matrix for the classical random walk.
    initial_state (numpy.ndarray): Initial state vector.
    t (numpy.ndarray): Array of time steps.

    Returns:
    numpy.ndarray: Matrix of probabilities over time.
    """
    num_vertices = P.shape[0]
    data = np.zeros((num_vertices, len(t)))
    
    for i, ti in enumerate(t):
        state = initial_state.copy()
        for _ in range(int(ti)):  # Perform the random walk for 'ti' steps
            state = P @ state
        data[:, i] = np.abs (state.flatten())  # Store probabilities
    return data


# Transition matrix for the classical random walk
P = nx.to_numpy_array(G)
P = P / np.sum(P, axis=1, keepdims=True)  # Ensure rows sum to 1

time_range = np.linspace(0, 30, 1000)

# Compute probabilities over time for both walks
quantum_probabilities_over_time = evolve_2d_graph(H, initial_state, time_range)
classical_probabilities_over_time = classical_random_walk_2d(P, initial_state, time_range)

def plot_probability_evolution():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    # Quantum walk heatmap
    im1 = axes[0].imshow(quantum_probabilities_over_time, aspect='auto', cmap='viridis', origin='lower',
                        extent=[time_range.min(), 150, N, 0])
    axes[0].set_title('Quantum Walk Probability Evolution')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Vertex')
    fig.colorbar(im1, ax=axes[0], label='Probability')
    
    # Classical walk heatmap
    im2 = axes[1].imshow(classical_probabilities_over_time, aspect='auto', cmap='viridis', origin='lower',
                        extent=[time_range.min(), 150, N, 0])
    axes[1].set_title('Classical Random Walk Probability Evolution')
    axes[1].set_xlabel('Time')
    fig.colorbar(im2, ax=axes[1], label='Probability')
    
    plt.tight_layout()
    plt.show()

plot_probability_evolution()


# %%
def update_plot(t, graph_type):
    """
    Update plot for quantum and classical random walks on various graph types.

    Parameters:
    t (float): Time step for the simulation.
    graph_type (str): Type of the graph ('Cyclic', '2D', 'Random').
    """
    # Ensure t is a scalar and not a list
    t = float(t)
    evolved_state = evolve_2d_graph(H, initial_state, [t])[:, 0]
    quantum_probabilities = np.abs(evolved_state)**2
        
    classical_state = classical_random_walk_2d(H, initial_state, [t])[:, 0]
    classical_probabilities = classical_state

    # Compute the probability distribution for quantum and classical walks
    if graph_type == 'Cyclic':
        
        # Cyclic graph layout
        pos = dict((n, (np.cos(2 * np.pi * n / N), np.sin(2 * np.pi * n / N))) for n in G.nodes())
    
    elif graph_type == '2D':
        # 2D grid graph layout
        pos = dict((n, n) for n in G.nodes())
    elif graph_type == 'Random':

        pos = nx.spring_layout(G)
    else: 
        raise ValueError("Unsupported graph type.")

    # Create a figure with two subplots
    plt.figure(figsize=(12, 6))

    # Quantum walk subplot
    plt.subplot(1, 2, 1)
    nx.draw(G, pos, node_color=quantum_probabilities, cmap=plt.cm.viridis,
            with_labels=True, node_size=600, font_color='white')
    plt.title(f'Quantum Walk at t = {t:.2f}')
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), ax=plt.gca(), orientation='horizontal')

    # Classical walk subplot
    plt.subplot(1, 2, 2)
    nx.draw(G, pos, node_color=classical_probabilities, cmap=plt.cm.plasma,
            with_labels=True, node_size=600, font_color='white')
    plt.title(f'Classical Walk at t = {t:.2f}')
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.plasma), ax=plt.gca(), orientation='horizontal')

    plt.tight_layout()
    plt.show()

# Create a slider for time
#time_slider = widgets.FloatSlider(min=0, max=30, step=0.1, value=0, description='Time')
time_slider = widgets.Play(min=0, max=100, step=1, value=0, description='Time')
# Create a dropdown for selecting graph type
graph_type_dropdown = widgets.Dropdown(
    options=['Cyclic', '2D','Random'],
    value='Cyclic',
    description='Graph Type:',
)

# Use the interact function to link the slider and dropdown to the plot update
interact(update_plot, t=time_slider, graph_type=graph_type_dropdown)
#%%




N=36
graph_type = 'Random'
H, _ = create_adjacency_matrices(N, graph_type)
_,G= create_adjacency_matrices(N, graph_type)

# Initial state: walker starts at vertex 0
initial_state = Gaussian_initial_state(N, N/2, N/10)
P = nx.to_numpy_array(G)
P = P / np.sum(P, axis=1, keepdims=True)  # Ensure rows sum to 1

time_range = np.linspace(0, 30, 1000)
def Generalized_Quantum_Walk(P, initial_state, t, graph_type):

    Classical_probabilities_over_time = classical_random_walk_2d(P, initial_state, t)
    Quantum_probabilities_over_time = evolve_2d_graph(H, initial_state, t)
    plot_probability_evolution()
    for i in t:
        update_plot(i, graph_type)
    

Generalized_Quantum_Walk(P, initial_state, time_range, graph_type)
    
  



#%%
