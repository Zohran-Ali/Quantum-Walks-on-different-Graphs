#%%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from Function import create_adjacency_matrix_constant_weights,create_adjacency_matrix
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
p_si0= np.zeros(N, dtype=complex)
# Gaussian Initial State (Vectorized)
positions = np.arange(N)
p_si0 = np.exp(-((positions - center)**2) / (2 * width**2)).reshape(N, 1)
#p_si0[N//2] = 1  # Start the walker at the middle vertex
p_si0 = p_si0 / norm(p_si0)  # Normalize the state


H = create_adjacency_matrix(N)  #
#print(H)

# Calculate the probability distributions over time (Vectorized)

data = np.zeros((N, len(t)))
for i, ti in enumerate(t):
    U = expm(imag * H * ti)  # Unitary evolution operator
    psif = U @ p_si0  # Apply the unitary evolution
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
    
H = create_adjacency_matrix_constant_weights(N)
# Calculate the probability distributions over time (Vectorized)
data = np.zeros((N, len(t)))
for i, ti in enumerate(t):
    U = expm(imag * H * ti)  # Unitary evolution operator
    psif = U @ p_si0  # Apply the unitary evolution
    data[:, i] = np.abs(psif.flatten())**2  # Store probabilities

# Plotting the Heatmap
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='gist_gray_r', aspect='auto', extent=[0,151,51,1])
plt.colorbar(label='Probability')
plt.title('Quantum Walk Probability Evolution on a Cyclic Graph with constant edge weights of pi/2')
plt.xlabel('Time')
plt.ylabel('Vertex')
plt.show()

#%%


# Linear superposition of states
ps_0 = np.zeros(N) 
ps_0[22: 26] = 1  # Initial state
ps0 = ps_0 / norm(ps_0)  # Normalize the state


H = create_adjacency_matrix(N)

t = np.linspace(0, 30, 1000)
imag = complex(0, -1)

data = np.zeros((N, len(t)))
for i, ti in enumerate(t):
    U = expm(imag * H * ti)  # Unitary evolution operator
    psif = U @ ps_0  # Apply the unitary evolution
    data[:, i] = np.abs(psif.flatten())**2  # Store probabilities


plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='gist_gray_r', aspect='auto', extent=[0,151,51,1])
plt.colorbar(label='Probability')
plt.title('Quantum Walk Probability Evolution on a Cyclic Graph with linear superposition of states')
plt.xlabel('Time')
plt.ylabel('Vertex')
plt.show()

#print(ps_0)

if __name__ == "__main__":
    print("This will only print when you run my_module.py directly.")
    create_adjacency_matrix_constant_weights(N)


# %%
