
#%%
import tkinter as tk
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from Function import create_adjacency_matrices,evolve_2d_graph,coherent_state, Gaussian_initial_state,localized_initial_state,superposition_initial_state
from numpy.linalg import norm
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from tkinter import Tk, Entry, Label, Button, StringVar, Radiobutton, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


#%%

# Function to run the quantum walk simulation
def run_simulation():
    """
    Runs a quantum walk simulation based on user input for the number of vertices, 
    initial state type, and graph type. The function creates the initial state vector, 
    adjacency matrix, and time array, then evolves the quantum walk and plots the 
    probability distribution in a Tkinter window.

    Parameters:
    None (uses global variables for user input)

    Returns:
    None (plots the result in a Tkinter window)
    """

    for widget in frame_plot.winfo_children(): #destroying the previous widgets and plots
        widget.destroy()
    try:
        N = int(entry_vertices.get())
        if N <= 0:
            raise ValueError("Number of vertices must be a positive integer.")
        
        # Retrieve the initial state type selected by the user
        initial_state_type = initial_state_type_var.get()


        # Create the initial state vector based on the type selected
        if initial_state_type == "Localized":
            ps_0 = localized_initial_state(N,0)

        elif initial_state_type == "Superposition":

            ps_0 = superposition_initial_state(N,0,N//2)
            
        elif initial_state_type == "Gaussian":
        
            ps_0 = Gaussian_initial_state(N,N/2,N/10)
               

        elif initial_state_type == "Coherent":
            alpha = 1.0 + 1.0j  # Coherent state amplitude
            ps_0= coherent_state(alpha,N)

        else:

            raise ValueError("Unsupported initial state type.")
# Retrieving the graph type selected by the user
        graph_type = graph_type_var.get()

        H, _ = create_adjacency_matrices(N, graph_type)
        _,G= create_adjacency_matrices(N, graph_type)
        t = np.linspace(0, 30, 1000)
     # Choosing the evolution function based on the graph type
        if graph_type == "Random":
             data = evolve_2d_graph(H, ps_0, t)
        elif graph_type == "2D Grid":
            data = evolve_2d_graph(H, ps_0, t)
        elif graph_type == "Cyclic Graph":
            data = evolve_2d_graph(H, ps_0, t)
        else:
            raise ValueError("Unsupported graph type.")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Two subplots in a vertical layout
        fig.subplots_adjust(hspace=0.5)
        norms = mcolors.Normalize(vmin=data.min(), vmax=data.max())
        # First plot
        cax1 = ax1.imshow(data, cmap='viridis', norm=norms, aspect='auto', extent=[0, 150, N, 1])
        fig.colorbar(cax1, ax=ax1, label='Probability')
        ax1.set_title(f'QW Probability Evolution on a {graph_type} graph, with {initial_state_type} initial state ')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Vertex')

        cax2 = nx.draw(G, with_labels=True, node_size=500, node_color="lightblue", font_size=8)
        ax2.set_title(f'{graph_type} with N = {N} Vertices')
        
    

        # Embedding the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except ValueError as e:
        messagebox.showerror("Input Error", str(e))
# Setting up the Tkinter window
root = Tk()
root.title("Quantum Walk Simulation")

# Creating a main frame for layout management
main_frame = tk.Frame(root)
main_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

# Entry for number of vertices
Label(main_frame, text="Number of vertices:").grid(row=0, column=0, sticky='w')
entry_vertices = Entry(main_frame)
entry_vertices.grid(row=0, column=1, sticky='ew')

# Dropdown or radio buttons for initial state type selection
initial_state_type_var = StringVar(value="Localized")  # Default to "Localized"
Label(main_frame, text="Initial State Type:").grid(row=1, column=0, sticky='w')
Radiobutton(main_frame, text="Localized", variable=initial_state_type_var, value="Localized").grid(row=1, column=1, sticky='w')
Radiobutton(main_frame, text="Superposition", variable=initial_state_type_var, value="Superposition").grid(row=2, column=1, sticky='w')
Radiobutton(main_frame, text="Gaussian", variable=initial_state_type_var, value="Gaussian").grid(row=3, column=1, sticky='w')
Radiobutton(main_frame, text="Coherent", variable=initial_state_type_var, value="Coherent").grid(row=4, column=1, sticky='w')

# radio buttons for graph type selection
graph_type_var = StringVar(value="Random")  # Default to "Random"
Label(main_frame, text="Graph Type:").grid(row=5, column=0, sticky='w')
Radiobutton(main_frame, text="Random", variable=graph_type_var, value="Random").grid(row=5, column=1, sticky='w')
Radiobutton(main_frame, text="Cyclic Graph", variable=graph_type_var, value="Cyclic Graph").grid(row=6, column=1, sticky='w')
Radiobutton(main_frame, text="2D Grid", variable=graph_type_var, value="2D Grid").grid(row=7, column=1, sticky='w')

# Run simulation button
Button(main_frame, text="Run Simulation", command=run_simulation).grid(row=8, column=0, columnspan=2, pady=10)

# Frame for the plot
frame_plot = tk.Frame(root, width=800, height=600)  # Set width and height
frame_plot.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

# Configure row and column weights for the main frame
main_frame.grid_rowconfigure(8, weight=1)
main_frame.grid_columnconfigure(1, weight=1)

# Configuring row and column weights for the root window
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

# Start the Tkinter event loop
root.mainloop()

