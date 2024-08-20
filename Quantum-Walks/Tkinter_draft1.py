
#%%
import tkinter as tk
import numpy as np
from Function import create_adjacency_matrices,evolve_2d_graph,create_adjacency_matrix_constant_weights
from numpy.linalg import norm
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from tkinter import Tk, Entry, Label, Button, StringVar, Radiobutton, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#%%
z1 = complex(0, np.pi)

# Function to run the quantum walk simulation
def run_simulation():
    try:
        N = int(entry_vertices.get())
        if N <= 0:
            raise ValueError("Number of vertices must be a positive integer.")
        
        initial_range = list(map(int, entry_initial_state.get().split(',')))
        if len(initial_range) != 2 or initial_range[0] < 0 or initial_range[1] >= N or initial_range[0] > initial_range[1]:
            raise ValueError("Please provide a valid range for initial state.")
        
        ps_0 = np.zeros(N)
        ps_0[initial_range[0]:initial_range[1] + 1] = 1
        ps_0 = ps_0 / norm(ps_0)  # Normalize the state

        H = create_adjacency_matrix_constant_weights(N)

        t = np.linspace(0, 30, 1000)
        imag = complex(0, -1)

        data = np.zeros((N, len(t)))
        for i, ti in enumerate(t):
            U = expm(imag * H * ti)  # Unitary evolution operator
            psif = U @ ps_0  # Apply the unitary evolution
            data[:, i] = np.abs(psif.flatten())**2  # Store probabilities

        # Plotting the result in a Tkinter window
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.imshow(data, cmap='gist_gray_r', aspect='auto', extent=[0, 30, N, 1])
        fig.colorbar(cax, label='Probability')
        ax.set_title('Quantum Walk Probability Evolution on a Cyclic Graph')
        ax.set_xlabel('Time')
        ax.set_ylabel('Vertex')

        # Embedding the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack()

    except ValueError as e:
        messagebox.showerror("Input Error", str(e))

# Setting up the main Tkinter window
root = tk.Tk()
root.title("Quantum Walk Simulation")

# Input for number of vertices
tk.Label(root, text="Enter the number of vertices (N):").pack(anchor='w')
entry_vertices = tk.Entry(root)
entry_vertices.pack(anchor='w')

# Input for initial superposition state range
tk.Label(root, text="Enter initial state range (e.g., 22,25):").pack(anchor='w')
entry_initial_state = tk.Entry(root)
entry_initial_state.pack(anchor='w')


# Run button
tk.Button(root, text="Run Simulation", command=run_simulation).pack()

# Start the Tkinter event loop
root.mainloop()