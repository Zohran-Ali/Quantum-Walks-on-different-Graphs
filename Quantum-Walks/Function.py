#%%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import expm
from numpy.linalg import norm
from scipy.special import factorial
import cmath
#%%


def generate_random_graph(n, p):
    """
    Generate a random graph using the Erdős-Rényi model.

    Parameters:
    - n (int): Number of nodes in the graph.
    - p (float): Probability of edge creation between any pair of nodes.

    Returns:
    - G (networkx.Graph): The generated random graph.
    """
    # Generate the random graph
    G = nx.erdos_renyi_graph(n, p)

    ## Optionally, you can draw the graph to visualize it
    #plt.figure(figsize=(8, 6))
    #nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
    #plt.title(f"Erdős-Rényi Random Graph with {n} Nodes and Probability {p}")
   # plt.show()

    return G

def quantum_walk(steps):
    """
    Function to simulate a Quantum walk on a line

    Parameters:
    steps(int): The number of steps to take in the walk.
    Returns:
    numpy.ndarray: The probability distribution of the position at each step for each position.
    """   
    # Initializing position and coin states
    position_states = np.zeros((2 * steps + 1, 2), dtype=complex)
    # Starting at position 0 with coin state |0⟩
    position_states[steps, 1] = 1

    # Hadamard coin operator
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    for _ in range(steps):
        new_states = np.zeros_like(position_states)
        for i in range(1, len(position_states) - 1):
            new_states[i-1, 0] += hadamard[0, 0] * position_states[i, 0] + hadamard[0, 1] * position_states[i, 1]
            new_states[i+1, 1] += hadamard[1, 0] * position_states[i, 0] + hadamard[1, 1] * position_states[i, 1]
        position_states = new_states
    
    return np.sum(np.abs(position_states)**2, axis=1)

def create_adjacency_matrices(N, graph_type):
    if graph_type == "Random":
        G = generate_random_graph(N, p=0.3)
        H = nx.adjacency_matrix(G).todense()
       # plt.figure(figsize=(6, 6))
      #  nx.draw(G, with_labels=True, node_size=500, node_color="lightblue", font_size=8)
       # plt.title("Random Graph with  Nodes")
      #  plt.show()
    elif graph_type == "2D Grid":
        side = int(np.sqrt(N))
        if side * side != N:
            raise ValueError("Number of vertices must be a perfect square for 2D grid.")
        G = nx.grid_2d_graph(side,side)
        # Defining the adjacency matrix as the Hamiltonian
        H = nx.adjacency_matrix(G).todense()
        # Draw the graph
        #plt.figure(figsize=(6, 6))
        #nx.draw(G, with_labels=True, node_size=500, node_color="lightblue", font_size=8)
        #plt.title("Cyclic Graph with 51 Nodes")
        #plt.show()
    elif graph_type == "Cyclic Graph":
        G = nx.cycle_graph(N)
        H=create_adjacency_matrix_constant_weights(N)
        #pos = nx.circular_layout(G)

        # Draw the graph
       # plt.figure(figsize=(6, 6))
        #nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=8)
        #plt.title("Cyclic Graph with 51 Nodes")
        #plt.show()
    else:
        raise ValueError("Invalid graph type selected.")
    return H,G

def coherent_state(alpha, N):
    """
    Initializing a coherent state |alpha> in the Fock basis.

    Parameters:
    alpha : complex
        The amplitude of the coherent state.
    N : int
        The number of Fock states to include in the expansion.

    Returns:
    np.ndarray
        A vector representing the coherent state in the Fock basis.
    """
    # Initialize the state vector
    state = np.zeros(N, dtype=complex)
    
    # Calculate each component in the Fock basis
    for n in range(N):
        state[n] = (alpha**n / np.sqrt(factorial(n))) * np.exp(-0.5 * np.abs(alpha)**2)
    
    return state
# Classical random walk on a 1D lattice
def classical_random_walk(steps, num_walkers):


    """
    Function to simulate a classical random walk on a line

    Parameters:
    steps(int): The number of steps to take in the walk.
    num_walkers (int): The number of walkers to simulate for averagig the distribution.

    Returns:
    numpy.ndarray: The probability distribution of the position at each step for each position.
    """   

    position_counts = np.zeros(2 * steps + 1)
    for _ in range(num_walkers):
        position = steps  # Start at position 0 (centered at index `steps`)
        for _ in range(steps):
            position += np.random.choice([-1, 1])
        position_counts[position] += 1
    return position_counts / num_walkers  # Normalize to get probabilities



def create_adjacency_matrix(N):
    """
    Function to create an N x N adjacency matrix for a cyclic graph with N vertices
    The adjacency matrix is such that phase angle of the complex edge weights,
    α, increases linearly from 0 to π along consecutive edges
    Parameters:
    N (int): Number of vertices in the graph

    Returns:
    A_prime (np.ndarray): N x N adjacency matrix
    """

    # Initializing an N x N adjacency matrix with zeros
    A_prime = np.zeros((N, N), dtype=complex)
    
    # Populating the adjacency matrix based on the linear phase ramp
    for i in range(N-1):
        phase = np.exp(1j * i * np.pi / (N-1))
        A_prime[i, i+1] = phase
        A_prime[i+1, i] = np.conj(phase)  # Ensure the matrix is Hermitian
    
    # Set the last element to first connection (closing the cycle)
    A_prime[0, N-1] = np.exp(-1j * np.pi)
    A_prime[N-1, 0] = np.exp(1j * np.pi)
    
    return A_prime


z1 = complex(0, np.pi)	




def create_adjacency_matrix_constant_weights(N):
# Hamiltonian Initialization (Optimized)
    H = np.zeros((N, N), dtype=complex)
    H[np.arange(N-1), np.arange(1, N)] = cmath.exp(z1/2)
    H[np.arange(1, N), np.arange(N-1)] = cmath.exp(-z1/2)
    H[0, N-1] = cmath.exp(-z1/2)
    H[N-1, 0] = cmath.exp(z1/2)
    return H

# Time evolution via Unitary operator U(t) = e^(-iHt)
def time_evolution(H, initial_state, t):
    U = expm(-1j * H * t)  
    return U @ initial_state

def evolve_2d_graph(H, ps_0, t):
    imag = complex(0, -1)
    data = np.zeros((H.shape[0], len(t)))
    for i, ti in enumerate(t):
        U = expm(imag * H * ti)  # Unitary evolution operator
        psif = U @ ps_0  # Apply the unitary evolution
        data[:, i] = np.abs(psif.flatten())**2  # Store probabilities
    return data


# Time evolution function for classical random walk
def classical_random_walk(initial_state, P, t):
    """
    Function to simulate a classical random walk on a line

    Parameters:
    initial_state (numpy.ndarray): The number of steps to take in the walk.
    P()

    Returns:
    numpy.ndarray: The probability distribution of the position at each step for each position.
    """   
    state = initial_state.copy()
    for _ in range(int(t)):
        state = P @ state
    return state


