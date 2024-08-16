#%%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import expm
from numpy.linalg import norm
import cmath




#%%

# Create a cyclic graph with 50 nodes
G = nx.cycle_graph(51)

# Generate the positions using a circular layout
pos = nx.circular_layout(G)

# Draw the graph
plt.figure(figsize=(6, 6))
nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=8)
plt.title("Cyclic Graph with 51 Nodes")
plt.show()







#%%
# Reproducing Fig1 DOI 10.1088/1367-2630/ac1551, Complex edge weights with linear phase ramp
#Talks about directed information transfer via tailoring Gaussian initial state
#The graph is cyclic with N = 51 vertices
imag = complex(0, -1)
# Parameters
N = 51  # Number of vertices
center = 25  # Center of the Gaussian initial state
width = N / 10  # Width of the Gaussiant
p1 = np.pi  # Setting p1 to π
z1 = complex(0, p1)  # Complex number z1 = 0 + πi
#define time range and calculate probabilities over this range
t = np.linspace(0, 30, 1000)

# Gaussian Initial State (Vectorized)
positions = np.arange(N)
psi0 = np.exp(-((positions - center)**2) / (2 * width**2)).reshape(N, 1)
#psi0[N//2] = 1  # Start the walker at the middle vertex
psi0 = psi0 / norm(psi0)  # Normalize the state

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

H = create_adjacency_matrix(N)  #
#print(H)

# Calculate the probability distributions over time (Vectorized)

data = np.zeros((N, len(t)))
for i, ti in enumerate(t):
    U = expm(imag * H * ti)  # Unitary evolution operator
    psif = U @ psi0  # Apply the unitary evolution
    data[:, i] = np.abs(psif.flatten())**2  # Storing probabilities

# Plotting the Heatmap
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='gist_gray_r', aspect='auto', extent=[0,151,51,1])
plt.colorbar(label='Probability')
plt.title('Quantum Walk Probability Evolution on a Cyclic Graph with Linear lamp of complex edge weiights')
plt.xlabel('Time')
plt.ylabel('Vertex  j')
plt.show()

#%%


#Previous results with a different method, by assiging constant complex edge weights from pi/2
# to all the vertices 
#

# Hamiltonian Initialization (Optimized)
H = np.zeros((N, N), dtype=complex)
H[np.arange(N-1), np.arange(1, N)] = cmath.exp(z1/2)
H[np.arange(1, N), np.arange(N-1)] = cmath.exp(-z1/2)
H[0, N-1] = cmath.exp(-z1/2)
H[N-1, 0] = cmath.exp(z1/2)

# Time Evolution
t = np.linspace(0, 30, 1000)
imag = complex(0, -1)

# Calculate the probability distributions over time (Vectorized)
data = np.zeros((N, len(t)))
for i, ti in enumerate(t):
    U = expm(imag * H * ti)  # Unitary evolution operator
    psif = U @ psi0  # Apply the unitary evolution
    data[:, i] = np.abs(psif.flatten())**2  # Store probabilities


print(H)
# Plotting the Heatmap
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='gist_gray_r', aspect='auto', extent=[0,151,51,1])
plt.colorbar(label='Probability')
plt.title('Quantum Walk Probability Evolution on a Cyclic Graph with constant edge weights of pi/2')
plt.xlabel('Time')
plt.ylabel('Vertex')
plt.show()

#%%

