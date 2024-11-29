# you better arXiv:2001.08361v1  or else
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Constants for base values and exponents
COMPUTE_BASE = 2.3e8
DATASET_BASE = 5.4e13
PARAMETERS_BASE = 8.8e13
EXPS = [-0.05, -0.095, -0.076]

# Default input values and labels
DEFAULTS = ["1e6", "1e9", "1e7"]
LABELS = [
    "Enter your compute (petaflop/days)",
    "Enter your dataset size (tokens)",
    "Enter your model size (number of non-embedding model parameters)"
]

# Apply a dark theme to plots
plt.style.use('dark_background')

def update_function(entry, result_label, base, exp, label):
    """Calculate and update the isolated loss for a given input."""
    try:
        value = float(entry.get() or 0)
        loss = (value / base) ** exp
        result_label.config(text=f"{label}: {int(value):,} → Isolated Loss = {loss:.4f}")
        update_all_graphs()
    except ValueError:
        result_label.config(text=f"{label}: Invalid input")

def update_all_graphs():
    """Refresh all graphs and the estimated loss display."""
    update_main_graph()
    update_estimated_loss()
    update_mini_plots()

def update_main_graph():
    """Adjust the layout of the main graph."""
    try:
        plt.tight_layout()
        plt.subplots_adjust(right=0.45)
    except Exception as e:
        print(f"Error updating main graph: {e}")

def get_bottleneck_description(label):
    """Return a short description based on the full label."""
    if label == "Enter your compute (petaflop/days)":
        return "Compute"
    elif label == "Enter your dataset size (tokens)":
        return "Dataset size"
    elif label == "Enter your model size (number of non-embedding model parameters)":
        return "Model size"
    else:
        return "Unknown"

def update_estimated_loss():
    """Calculate and display the estimated loss and identify the bottleneck."""
    try:
        input_values = [float(entry.get() or 0) for entry in entries]
        losses = [(value / base) ** exp for value, base, exp in zip(input_values, [COMPUTE_BASE, DATASET_BASE, PARAMETERS_BASE], EXPS)]
        
        max_loss = max(losses)
        estimated_loss_label.config(text=f"Estimated LM Configuration Loss: {max_loss:.4f}")
        
        # Determine the bottleneck description
        bottleneck_index = losses.index(max_loss)
        bottleneck_label_text = get_bottleneck_description(LABELS[bottleneck_index])
        bottleneck_label.config(text=f"Your model bottleneck is: {bottleneck_label_text}")
    except Exception as e:
        print(f"Error updating estimated loss: {e}")

def update_mini_plots():
    """Update the mini plots for each input parameter."""
    try:
        input_values = [float(entry.get() or 0) for entry in entries]
        default_ranges = [np.logspace(5, 9, 100), np.logspace(8, 15, 100), np.logspace(6, 14, 100)]
        equations = [
            r"$f(x) = \left(\frac{x}{2.3 \times 10^8}\right)^{-0.05}$",
            r"$g(x) = \left(\frac{x}{5.4 \times 10^{13}}\right)^{-0.095}$",
            r"$j(x) = \left(\frac{x}{8.8 \times 10^{13}}\right)^{-0.076}$"
        ]
        
        for i, (fig, value, base, exp, default_range, equation) in enumerate(zip(mini_figs, input_values, [COMPUTE_BASE, DATASET_BASE, PARAMETERS_BASE], EXPS, default_ranges, equations)):
            ax = fig.gca()
            ax.clear()
            
            x_min, x_max = min(default_range[0], value * 0.01), max(default_range[-1], value * 100)
            x_points = np.logspace(np.log10(x_min), np.log10(x_max), 100)
            y_points = (x_points / base) ** exp
            
            ax.semilogx(x_points, y_points, 'b-', alpha=0.5)
            ax.plot(value, (value / base) ** exp, 'ro')
            
            ax.grid(True, alpha=0.3)
            ax.set_xticks([x_min, value, x_max])
            ax.set_xticklabels([f"{tick:.1e}" for tick in [x_min, value, x_max]], rotation=45)
            ax.text(0.95, 0.95, equation, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', color='white')
            
            fig.tight_layout()
            mini_canvases[i].draw()
    except Exception as e:
        print(f"Error updating mini plots: {e}")

# Create the main application window
root = tk.Tk()
root.title("Power Law Functions")
root.geometry("900x600")

# Create a frame for controls with a fixed height
controls_frame = ttk.Frame(root, height=300)
controls_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
controls_frame.pack_propagate(False)

# Initialize frames, figures, and canvases for each input
frames = []
mini_figs = []
mini_canvases = []

for _ in range(3):
    frame = ttk.Frame(controls_frame)
    frame.pack(fill=tk.X, pady=5)
    frames.append(frame)

    left_frame = ttk.Frame(frame)
    left_frame.pack(side=tk.LEFT, padx=(0, 10))

    right_frame = ttk.Frame(frame)
    right_frame.pack(side=tk.RIGHT)
    
    mini_fig = Figure(figsize=(3.33, 1.5))
    mini_figs.append(mini_fig)
    
    mini_canvas = FigureCanvasTkAgg(mini_fig, master=right_frame)
    mini_canvas.get_tk_widget().pack(side=tk.RIGHT)
    mini_canvases.append(mini_canvas)

# Setup input sections with labels and entry fields
entries = []
results = []

for i, (frame, label_text, default) in enumerate(zip(frames, LABELS, DEFAULTS)):
    left_frame = frame.winfo_children()[0]
    
    tk.Label(left_frame, text=label_text).pack(anchor='w')
    entry = ttk.Entry(left_frame, width=20)
    entry.pack(anchor='w')
    entry.insert(0, default)
    entries.append(entry)
    
    result_label = tk.Label(left_frame, text="f(x) = 0")
    result_label.pack(anchor='w')
    results.append(result_label)

# Results section for estimated loss and bottleneck
estimated_loss_label = tk.Label(root, text="Estimated LM Configuration Loss: 0", font=('Arial', 20, 'bold'))
estimated_loss_label.pack(pady=(20,5))

bottleneck_label = tk.Label(root, text="Your model bottleneck is: None", font=('Arial', 20, 'bold'))
bottleneck_label.pack(pady=(5,20))

# Bind entry fields to update functions
entry1, entry2, entry3 = entries
result1, result2, result3 = results

entry1.bind('<Return>', lambda e: update_function(entry1, result1, COMPUTE_BASE, EXPS[0], "Compute"))
entry2.bind('<Return>', lambda e: update_function(entry2, result2, DATASET_BASE, EXPS[1], "Dataset_size"))
entry3.bind('<Return>', lambda e: update_function(entry3, result3, PARAMETERS_BASE, EXPS[2], "Model_size"))

# Initial update for each function
update_function(entry1, result1, COMPUTE_BASE, EXPS[0], "Compute")
update_function(entry2, result2, DATASET_BASE, EXPS[1], "Dataset")
update_function(entry3, result3, PARAMETERS_BASE, EXPS[2], "Parameters")

# Start the Tkinter main loop
root.mainloop()
