# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:35:43 2024

@author: Natalia
"""

import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import Frame, Label, Toplevel, Entry, IntVar, messagebox
from PIL import Image, ImageTk
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import ticker #as mticker
import matplotlib.colors as colors
import sys
from scipy import signal
codepath = r'C:\Users\Usuario\Nextcloud\Doctorado\Códigos varios'
#codepath = r'C:\Users\Natalia\Nextcloud\Doctorado\Códigos varios'
sys.path.append(codepath)
from pcfanalysis_2024 import pCF, pCF_complete_analysis
from ccpcf_final import crosspCF
from czifile import imread
import time  # For simulating time-consuming tasks
import threading
from matplotlib.colors import LinearSegmentedColormap, Normalize
from tkinter import simpledialog



# ██████  ██       ██████  ██████   █████  ██          ██    ██  █████  ██████  ██  █████  ██████  ██      ███████ ███████ 
#██       ██      ██    ██ ██   ██ ██   ██ ██          ██    ██ ██   ██ ██   ██ ██ ██   ██ ██   ██ ██      ██      ██      
#██   ███ ██      ██    ██ ██████  ███████ ██          ██    ██ ███████ ██████  ██ ███████ ██████  ██      █████   ███████ 
#██    ██ ██      ██    ██ ██   ██ ██   ██ ██           ██  ██  ██   ██ ██   ██ ██ ██   ██ ██   ██ ██      ██           ██ 
# ██████  ███████  ██████  ██████  ██   ██ ███████       ████   ██   ██ ██   ██ ██ ██   ██ ██████  ███████ ███████ ███████ 
original_kimograms = []  # Store kimograms for multiple files
fig = None
canvas = None
rect_selector = None
original_xlim = None  # To store original x-axis limits
original_ylim = None  # To store original y-axis limits
global last_fig, G_to_save, T_to_save, current_figures
current_figures = []
last_fig = None
ax = None
canvas = None
current_ax = None
rect_selector = None
current_index = 0  # Track current kimogram index
G_to_save = None
T_to_save = None
dwell_time = None
pixels = None


def my_colors():
    '''
    This function creates several colormaps options for the user to choose for the plotting.
    Be aware that not all of them are linear. For more info check https://matplotlib.org/stable/users/explain/colors/colormaps.html
    Returns
    -------
    TYPE colormap
    '''
    cmap_name = cmap_var.get()
    colores = {
        'smooth_viridis': ["black", "#440154", "#482677", "#3e4a89", "#2a788e", "#22a884", "#7dcd3e", "#fde725"],
        'plasma': ["#0d0887", "#46039f", "#7201a8", "#ab5b88", "#d8d83c", "#f0f921"],
        'cividis': ["#00224c", "#3b5c96", "#66a85f", "#c7c41f", "#f4f824"],
        'jet': ["#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000"]
    }
    cmap_name = cmap_name if cmap_name in colores else 'smooth_viridis'
    return LinearSegmentedColormap.from_list(cmap_name, colores[cmap_name])


#██       ██████   █████  ██████  ██ ███    ██  ██████       █████  ███    ██ ██████      ██████  ██ ███████ ██████  ██       █████  ██    ██ 
#██      ██    ██ ██   ██ ██   ██ ██ ████   ██ ██           ██   ██ ████   ██ ██   ██     ██   ██ ██ ██      ██   ██ ██      ██   ██  ██  ██  
#██      ██    ██ ███████ ██   ██ ██ ██ ██  ██ ██   ███     ███████ ██ ██  ██ ██   ██     ██   ██ ██ ███████ ██████  ██      ███████   ████   
#██      ██    ██ ██   ██ ██   ██ ██ ██  ██ ██ ██    ██     ██   ██ ██  ██ ██ ██   ██     ██   ██ ██      ██ ██      ██      ██   ██    ██    
#███████  ██████  ██   ██ ██████  ██ ██   ████  ██████      ██   ██ ██   ████ ██████      ██████  ██ ███████ ██      ███████ ██   ██    ██    
                                                                                                                                            
                                                       

def create_kimogram(lines):
    '''
    this is a simple function to accomodate the given data to be plotted as a kimogram
    Parameters
    ----------
    lines : matrix
        DESCRIPTION.
        data already transformed into a matrix with lines and pixels as the dimensions 

    Returns
    -------
    TYPE numpy array with the data dimensions

    '''
    return np.vstack(lines).astype(np.uint16)

import czifile
import xml.etree.ElementTree as ET
def extract_metadata(file_path, callback):
    global dwell_time, pixels
    try:
        czi_file = czifile.CziFile(file_path)
        metadata = czi_file.metadata()  # Call the method to get metadata
        root = ET.fromstring(metadata)  # Parse the XML

        # Find the PixelDwellTime or LineTime
        dwell_time_elem = root.find('.//LineTime')
        if dwell_time_elem is not None:
            dwell_time = np.round(float(dwell_time_elem.text),4)
        else:
            dwell_time = None  # Handle case where LineTime is not found

        # Call the callback to proceed
        callback()

    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")
        dwell_time = None  # Reset dwell_time on error
        callback()  # Still call the callback to avoid blocking


def load_czi_file(file_path, callback=None):
    global dwell_time
    # Load the CZI file
    data = imread(file_path)
    
    # Extract channels, flatten if necessary
    channels = data[0, :, :, 0, 0, :, 0]  # Adjust this based on your data shape
    num_channels = data.shape[2]
    
    # Extract metadata (and ensure callback is called after extracting)
    extract_metadata(file_path, lambda: callback() if callback else None)

    return [channels[:, i, :] for i in range(num_channels)]

from lfdfiles import SimfcsB64 as lfd

# def read_B64(Archivo,pixels):
#     Read=lfd(Archivo)
#     matrix=Read.asarray()
#     if type(len(matrix)/pixels) == float:   ## avoid possible uncompleted lines
#         last_line = int(len(matrix)/pixels)*pixels
#         matrix = matrix[0:last_line]                      
        
#     return (matrix).reshape(int(len(matrix)/pixels),pixels)    

def read_B64(Archivo):
    global dwell_time
    Read = lfd(Archivo)
    matrix = Read.asarray()
    metadata = open(Archivo[:-8] + '.jrn', 'r').readlines()
    # Initialize pixels and sampling_freq
    pixels = 128
    sampling_freq = None
    for line in metadata:
        if 'Box size' in line:
            pixels = int(line.split(':')[1].strip().split()[0])#*2  # Get the first part as an int
            
        if 'Sampling freq' in line:
            sampling_freq_str = line.split('Sampling freq')[1].split(':')[1].strip().split()[0]
            sampling_freq = np.round(float(sampling_freq_str))  # Convert to 
            
            dwell_time = pixels/sampling_freq
    # Check for uncompleted lines
    if len(matrix) % pixels != 0:  
        last_line = (len(matrix) // pixels) * pixels
        matrix = matrix[0:last_line]                      
        
    reshaped_matrix = matrix.reshape(-1, pixels)  # Reshape to have 'pixels' colum
    return reshaped_matrix

def load_lines():
    file_paths = filedialog.askopenfilenames(filetypes=[("TIFF files", "*.tiff;*.tif"), ("CZI files", "*.czi"), ("B64 files", "*.b64;*.raw")])
    if file_paths:
        threading.Thread(target=load_and_display, args=(file_paths,), daemon=True).start()
    else:
        show_message('Error', "You haven't uploaded the files correctly. Try again :)")

def show_message(kind, message):
    '''
    this function is for creating error messages when something has gone wrong. 
    The message given will depend on which function is crushing.

    Parameters
    ----------
    message : TYPE string
        DESCRIPTION. Message to be shown

    Returns
    -------
    None.

    '''    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    if kind=='Error':
        messagebox.showerror(kind, message)
    else:
        messagebox.showinfo(kind, message)
    root.destroy()  # Close the window
    
def load_and_display(file_paths):
    global original_kimograms
    original_kimograms = []
    errors = []

    # Load CZI files
    if len(file_paths) == 1 and file_paths[0].endswith('.czi'):
        # Define a callback function to load the CZI file after extracting metadata
        def load_czi_after_metadata():
            lines = load_czi_file(file_paths[0])  # Load the actual data
            original_kimograms.extend(lines)
            update_table_with_dwell_time()  # Call to update table with dwell time
        
        root.after(0, lambda: extract_metadata(file_paths[0], load_czi_after_metadata))  # Extract metadata and load data
        
    # Load TIFF files
    elif all(file_path.endswith(('.tiff', '.tif')) for file_path in file_paths):
        for file_path in file_paths:
            try:
                lines = np.array(Image.open(file_path))
                kimogram = create_kimogram([lines])
                original_kimograms.append(kimogram)
            except Exception as e:
                errors.append(f"Failed to load {file_path}: {e}")

     # Load B64 files
     # Load B64 files
    # Load B64 files
    else:
        def load_b64_files():
            #pixels = get_pixels()  # Get pixel count in the main thread
            #if pixels is None:
             
                #print("Pixel input was canceled. Skipping B64 file loading.")
                #return

            for file_path in file_paths:
                try:
                    global dwell_time
                    print(f"Loading B64 file: {file_path}")
                    lines = read_B64(file_path)
                    original_kimograms.append(lines)
                    update_table_with_dwell_time()  # Call to update table with dwell time
                    root.after(0, lambda: read_B64(file_path)) 
                    print(f"Successfully loaded lines from {file_path}: {lines.shape}")
                except Exception as e:
                    errors.append(f"Can't interpret {file_path} as a kimogram: {e}")

            if errors:
                show_message('Error', "\n".join(errors))
            else:
                display_kimograms(original_kimograms)
                
        # Call load_b64_files after a short delay to ensure UI is ready
        root.after(0, load_b64_files)
        
    if errors:
        root.after(0, lambda: show_message('Error', "\n".join(errors)))
    else:
        root.after(0, display_kimograms, original_kimograms)



from tkinter import simpledialog, messagebox

def get_pixels():
    # Create a dialog window
    pixels = simpledialog.askinteger("Input", "Please enter the value for pixels:")
    if pixels is None:  # Handle cancellation
        raise ValueError("Pixel input was canceled.")
    return pixels


def plot_kimograms(kimograms):
    '''
    function to stablish the way the kimograms are going to be plotted

    Parameters
    ----------
    kimograms : TYPE array or matrix
        DESCRIPTION.

    Returns 
    -------
    fig : TYPE figure
        DESCRIPTION.
        return the kimogram or kimograms of the uploaded line files

    '''
    global current_figures
    num_kimograms = len(kimograms)
    if num_kimograms>1:
        fig, axs = plt.subplots(1, num_kimograms, figsize=(8 * num_kimograms, 20), squeeze=False)
        axs = axs.flatten()  # Flatten the axis array for easier indexing

        for ax, kimogram in zip(axs, kimograms):
            cax = ax.imshow(kimogram, aspect='auto', cmap=my_colors(),
                            norm=plt.Normalize(vmin=0, vmax=np.mean(kimogram) + 2 * np.std(kimogram), clip=False), origin='upper')
            ax.set_xlabel("Pixel")
            ax.set_ylabel("Line Number")
            ax.set_title(f"Kimogram {axs.tolist().index(ax) + 1}")
            plt.colorbar(cax, ax=ax, label='Intensity')
            plt.subplots_adjust(wspace=0.5)  # Increase space between subplots
    else:
        kimograma = kimograms[0]
        fig, ax = plt.subplots(figsize=(8, 20))
        cax = ax.imshow(kimograma, aspect='auto', cmap=my_colors(),norm=plt.Normalize(vmin=0, vmax=np.mean(kimograma)+2*np.std(kimograma),clip=False), origin='upper')
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Line Number")
        ax.set_title("Kimogram")
        # Optional: Add a colorbar if needed
        plt.colorbar(cax, ax=ax, label='Intensity')
    current_figures.append(fig)
    return fig

def display_kimograms(kimograms):
    global fig, canvas
    fig = plot_kimograms(kimograms)
    plt.close('all')    
    if canvas is not None:
        canvas.get_tk_widget().destroy()
        plt.close('all')
    canvas = FigureCanvasTkAgg(fig, master=image_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0)




# Toggle between kimograms
def toggle_kimogram():
    '''
    This function is only relevant if there are two kimograms.
    It is used to change over which kimogram the buttons such as zoom or h-lines will be used.

    Returns
    -------
    None.
    '''
    global current_index
    if current_index==0:
        current_index = 1
    else:
        current_index = 0
    set_current_axis(current_index)
    update_kimogram_label()  # Update the label text

    canvas.draw()

#███████  ██████   ██████  ███    ███ 
#   ███  ██    ██ ██    ██ ████  ████ 
#  ███   ██    ██ ██    ██ ██ ████ ██ 
# ███    ██    ██ ██    ██ ██  ██  ██ 
#███████  ██████   ██████  ██      ██ 
                           

def zoom_in():
    '''
    this function is apply the kimograms to make a zoom in in the plotted lines.

    Returns
    -------
    None.

    '''
    global rect_selector, current_ax
    if current_ax is None:
        show_message('Error','No kimogram has been selected. Try with the Toggle kimogram button')
        return
    if rect_selector:
        rect_selector.set_active(False)
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        current_ax.set_xlim(min(x1, x2), max(x1, x2))
        current_ax.set_ylim(min(y1, y2), max(y1, y2))
        canvas.draw()
        rect_selector.set_active(False)
    rect_selector = RectangleSelector(
        current_ax, onselect,
        drawtype='box',
        useblit=True,
        button=[1],  # Left mouse button
        minspanx=5,
        minspany=5,
        spancoords='pixels',
        interactive=True
    )

def zoom_out():
    '''
    zoom out over the  selected kimogram. It is meant to be used after the zoom in one. If not it will do nothing.

    Returns
    -------
    None.
    '''
    global current_ax, original_limits
    if current_ax is None:
        show_message('Error','No kimogram has been selected. Try with the Toggle kimogram button')
        return
    ax_index = fig.axes.index(current_ax)
    original_xlim, original_ylim = original_limits[ax_index]
    current_ax.set_xlim(original_xlim)
    current_ax.set_ylim(original_ylim)
    canvas.draw()


        
def set_current_axis(index):
    '''
    this function is used to select over which kimogram the action will be done.

    Parameters
    ----------
    index : TYPE int
        DESCRIPTION. either 0 or 1 corresponding to ch1 and ch2.

    Returns
    -------
    None.

    '''
    global current_ax
    if 0 <= index < len(fig.axes):
        current_ax = fig.axes[index]
        print(f"Current axis set to: Kimogram {index + 1}")
    else:
        print("Invalid axis index")



def on_axis_select(index):
    # Sample button to switch axis (replace with your own logic)
    set_current_axis(index)


def open_hlines_window():
    '''
    this function is used to plot the horizontal profile of the kimogram, i.e. averaged lines.
    This is particularly helpful to identify regions of different intensity. 
    The plot is done in a new window where the user selects wich lines to average. 
    If there are 2 kimograms, use the tgogle function to change which one you want for this window.

    Returns
    -------
    None.

    '''
    hlines_window = Toplevel(root)
    hlines_window.title("Select lines range to plot")

    def plot_lines():
        global original_kimograms, current_index
        try:
            original_kimogram=original_kimograms[current_index]
        except:
            print('no current axis')
            original_kimogram=original_kimograms[0]
        start = start_var.get()
        num = num_var.get()

        if start < 0 or start >= original_kimogram.shape[0]:
            print("Error: Start line out of range.")
            return

        if num <= 0:
            print("Error: Number of lines to average must be positive.")
            return

        end = start + num
        if end > original_kimogram.shape[0]:
            print("Error: End index exceeds data range.")
            return

        # Calculate the average profile over the specified range
        lines = original_kimogram[start:end,:]
        average_lines = np.mean(lines, axis=0)

        # Create a new figure with adjusted size
        profile_fig = Figure(figsize=(8, 6), dpi=100)
        profile_ax = profile_fig.add_subplot(111)
        profile_ax.plot([i for i in range(len(original_kimogram[0]))], average_lines, color='black')
        
        # Set labels with larger font size
        profile_ax.set_title(f'Average intensity from line {start} to {end}', fontsize=14)
        profile_ax.set_xlabel('Pixel', fontsize=12)
        profile_ax.set_ylabel('Intensity', fontsize=12)
        
        # Increase padding and font size for ticks
        profile_ax.tick_params(axis='both', which='major', labelsize=10)
        profile_ax.tick_params(axis='both', which='minor', labelsize=8)
        
        # Add grid for better readability
        profile_ax.grid(True)
        
        # Create and add the canvas
        profile_canvas = FigureCanvasTkAgg(profile_fig, master=hlines_window)
        profile_canvas.draw()
        
        # Use grid() instead of pack()
        profile_canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, sticky='nsew')
        def on_click(event):
            if event.inaxes is not None:
                    x_data = event.xdata
                    y_data = event.ydata
                    print(f"Clicked at x={x_data:.2f}, y={y_data:.4f}")
                    # Update or create a label to show these coordinates
                    coord_label.config(text=f"Coordinates: x={x_data:.2f}s, y={y_data:.4f}")
        # Connect the event handler
        profile_canvas.mpl_connect('button_press_event', on_click)
        
        # Create a label to display the coordinates
        coord_label = Label(hlines_window, text="Coordinates: x= , y=")
        coord_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

    # Add entries for starting pixel and number of pixels
    start_var = IntVar(value=0)
    num_var = IntVar(value=1)
    
    start_label = Label(hlines_window, text="Start line:")
    start_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
    
    start_entry = Entry(hlines_window, textvariable=start_var)
    start_entry.grid(row=0, column=1, padx=10, pady=10, sticky='ew')
    
    num_label = Label(hlines_window, text="Number of lines to Average:")
    num_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')
    
    num_entry = Entry(hlines_window, textvariable=num_var)
    num_entry.grid(row=1, column=1, padx=10, pady=10, sticky='ew')
    
    plot_button = tk.Button(hlines_window, text="Plot H-lines", command=plot_lines)
    plot_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

    hlines_window.grid_columnconfigure(1, weight=1)
    hlines_window.grid_rowconfigure(3, weight=1)  # Allow row 3 to expand if needed


#██       ██████   █████  ██████      ██████   ██████ ███████     ██████   █████  ████████  █████  
#██      ██    ██ ██   ██ ██   ██     ██   ██ ██      ██          ██   ██ ██   ██    ██    ██   ██ 
#██      ██    ██ ███████ ██   ██     ██████  ██      █████       ██   ██ ███████    ██    ███████ 
#██      ██    ██ ██   ██ ██   ██     ██      ██      ██          ██   ██ ██   ██    ██    ██   ██ 
#███████  ██████  ██   ██ ██████      ██       ██████ ██          ██████  ██   ██    ██    ██   ██ 

import pandas as pd
def load_correlation():
    global G_to_save, T_to_save
    G_to_save= []
    file_paths = filedialog.askopenfilenames(filetypes=[("Correlation data", "*.txt;""*.csv")])
    if file_paths:
            file_time = filedialog.askopenfilenames(filetypes=[("Correlation time", "*.txt;""*.csv")])
    else:
        show_message('Error', "Can't correlate without the data :(")
        return
    T_to_save=np.loadtxt(file_time[0])
    if all(file_path.endswith('.txt') for file_path in file_paths):
        for file_path in file_paths:
            G_to_save.append(np.loadtxt(file_path, delimiter=','))
    elif all(file_path.endswith('.csv') for file_path in file_paths):
        for file_path in file_paths:
            G_to_save.append(pd.read_csv(file_path).to_numpy)
            
    else:
         show_message('Error',"I couldn't understand the shape of your files. Sorry")
         return
    
    num_kimograms = len(G_to_save)
    if num_kimograms>0:
        fig, axs = plt.subplots(1, num_kimograms, figsize=(8 * num_kimograms, 30), squeeze=False)
        axs = axs.flatten()  # Flatten the axis array for easier indexing

        for ax, G in zip(axs, G_to_save):
            vmin = 0
            vmax = np.max(G)
            y = np.arange(G.shape[0])
            im = ax.pcolor(y.transpose(),T_to_save, G.transpose(), shading="nearest",
                       cmap=my_colors(), vmin=vmin, vmax=vmax)

            ax.set_xlabel('pixels', fontsize=20)  # replace with your xlabel variable
            ax.set_ylabel('Logarithmic Time (s)', fontsize=20)
            ax.set_title('pcf', fontsize=20)
            cbarformat = ticker.ScalarFormatter()
            cbarformat.set_scientific('%.2e')
            cbarformat.set_powerlimits((0, 0))
            cbarformat.set_useMathText(True)

            cbar = fig.colorbar(im, ax=ax, orientation='vertical', format='%.2f')
            cbar.ax.yaxis.get_offset_text().set_fontsize(20)
            cbar.ax.yaxis.set_offset_position('left')
            cbar.ax.tick_params(labelsize=20)
            cbar.set_label(label='Amplitude', size=20)  # replace with your colorbar label

            ax.xaxis.tick_top()
            ax.tick_params(which='minor', length=2.25, width=1.25)
            ax.tick_params(which='major', length=3.5, width=1.75)
            ax.tick_params(axis='both', labelsize=20)
            ax.set_yscale("log")
            ax.invert_yaxis()    
    
    canvas = FigureCanvasTkAgg(fig, master=image_frame)  # Use image_frame as the parent
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Pack the canvas
    return fig

    
    

#██████   ██████ ███████ 
#██   ██ ██      ██      
#██████  ██      █████   
#██      ██      ██      
#██       ██████ ██      
                       

def update_table_with_dwell_time():
    global table_frame, dwell_time, labels, entries, checkbutton_vars
    # Clear existing labels and entries in the table_frame
    for widget in table_frame.winfo_children():
        widget.destroy()

    # Initialize or reset labels, entries, and checkbutton_vars
    labels = ["Line Time (ms)", "First Line", "Last Line", "Distance (px)", "H Smoothing (px)", "V Smoothing (lines)", "Reverse", 'Normalize']
    entries = []  # Reset entries
    checkbutton_vars = {}  # Reset Checkbutton vars

    # Recreate the table with updated values
    for row, label in enumerate(labels):
        tk.Label(table_frame, text=label, borderwidth=2, relief="solid").grid(row=row, column=0, padx=5, pady=5, sticky="e")

        if label == 'Reverse' or label=='Normalize':
            var = tk.BooleanVar()
            checkbox = tk.Checkbutton(table_frame, variable=var)
            entries.append(checkbox)  # Append checkbox, not its variable
            checkbutton_vars[label] = var
            checkbox.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        else:
            entry = tk.Entry(table_frame)
            if label == 'Line Time (ms)' and dwell_time is not None:
                entry.insert(0, f'{dwell_time * 1000}')
            elif label == 'H Smoothing (px)':
                entry.insert(0, '4')
            elif label == 'V Smoothing (lines)':
                entry.insert(0, '10')
            entries.append(entry)
            entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")


def get_table_data():
    '''
    This function collects the data the user puts into the pCF parameters table.

    Returns
    -------
    data : dict
    '''
    data = {}
    for label, widget in zip(labels, entries):
        if isinstance(widget, tk.Checkbutton):
            # Retrieve the BooleanVar associated with the Checkbutton
            var = checkbutton_vars.get(label)
            if var is not None:
                data[label] = var.get()
            else:
                data[label] = None
        elif isinstance(widget, tk.Entry):
            # Retrieve the value from Entry widgets
            value = widget.get().strip()  # Strip any whitespace
            if value:  # Only convert if the entry is not empty
                try:
                    data[label] = float(value)
                except ValueError:
                    data[label] = value  # Keep it as a string if conversion fails
            else:
                data[label] = ""  # Handle empty entry case
    
    print("Collected data:", data)  # Debug print
    return data


def plot_on_ax(ax, G_log, t_log,title):
    '''
    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    G_log : TYPE matrix
        DESCRIPTION. correlation matrix computed from pcf
    t_log : TYPE list
        DESCRIPTION. correlation delay times
    title : TYPE str
        DESCRIPTION. title asociated to the plot
    Returns
    -------
    None.

    '''
    vmin = 0
    vmax = np.max(G_log)
    y = np.arange(G_log.shape[0])
    im = ax.pcolor(y.transpose(), t_log, G_log.transpose(), shading="nearest",
               cmap=my_colors(), vmin=vmin, vmax=vmax)

    ax.set_xlabel('xlabel', fontsize=10)  # replace with your xlabel variable
    ax.set_ylabel('Logarithmic Time (s)', fontsize=16)
    ax.set_title(title, fontsize=10)
    cbarformat = ticker.ScalarFormatter()
    cbarformat.set_scientific('%.2e')
    cbarformat.set_powerlimits((0, 0))
    cbarformat.set_useMathText(True)

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', format='%.2f')
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.ax.yaxis.set_offset_position('left')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label='Correlation amplitude', size=16)  # replace with your colorbar label

    ax.xaxis.tick_top()
    ax.tick_params(which='minor', length=2.25, width=1.25)
    ax.tick_params(which='major', length=3.5, width=1.75)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_yscale("log")
    ax.invert_yaxis()
    return ax

def apply_ccpCF():
    '''
    this function applys the cross-pair correlation to the uploaded files. There should be two kimograms for this function to work.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    '''
    global G_to_save
    global T_to_save, table_frame
    if original_kimograms is None:
        show_message('Error',"No kimograms data to process.")
        pass
    
    # Get the parameters from the table and apply the pCF function
    data = get_table_data()
    #data = get_values()
    first_line = int(data.get("First Line", ""))
    last_line = int(data.get("Last Line", ""))
    line_time = data.get("Line Time (ms)", "")
    print(data)
    dr = int(data.get("Distance (px)", ""))
    #sigma = [int(data.get("H smoothing (px)", "")),int(data.get("V smoothing (lines)", ""))]
    reverse = data.get('Reverse')
    sigma = [
    int(data.get("H Smoothing (px)", 4)),  # Default to 0 if empty
    int(data.get("V Smoothing (lines)", 10))  # Default to 0 if empty
]
    acfnorm = data.get('Normalize')
    G, T = crosspCF(original_kimograms[0][first_line:last_line], original_kimograms[1][first_line:last_line], linetime=line_time/1000, dr=dr, reverse_PCF=reverse)
    G2, T2 = crosspCF(original_kimograms[1][first_line:last_line], original_kimograms[0][first_line:last_line], linetime=line_time/1000, dr=dr, reverse_PCF=reverse)
    x1 = np.geomspace(1, len(G), 256, dtype=int, endpoint = False)    
    t_lineal = T[:,0]
    t_log = np.geomspace(t_lineal[0], t_lineal[-1], 256, endpoint=True)
    G_basura = []
    for i in x1:
        G_basura.append(list(G[i]))
    G = np.asarray(G_basura).transpose()
    t = []
    for i in x1:
        t.append(t_lineal[i])
    t_lineal = np.asarray(t)
    G_log = np.empty_like(G)
    for i, gi in enumerate(G):
        G_log[i] = np.interp(t_log, t_lineal, gi)
    G_log = gaussian_filter(G_log, sigma = sigma)   ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    G_to_save = G_log  # Store G_log for saving
    T_to_save = t_log
    x1 = np.geomspace(1, len(G2), 256, dtype=int, endpoint = False)    
    G2_basura = [list(G2[i]) for i in x1]
    G2 = np.asarray(G2_basura).transpose()
    G2_log = np.empty_like(G2)
    for i, gi in enumerate(G2):
        G2_log[i] = np.interp(t_log, t_lineal, gi)
    G2_log = gaussian_filter(G2_log, sigma = sigma)   ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    
    G2_to_save = G2_log  # Store G_log for saving
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5.5), dpi=150)
    cmap = my_colors()
 # Plot Channel 1 on the first axis
    plot_on_ax(ax1, G_log,t_log, 'Ch1 to Ch2')
 
    try: 
        # Plot Channel 2 on the second axis
        plot_on_ax(ax2, G2_log,t_log, 'Ch2 to Ch1')
        G_to_save = [G_log,G2_log]  # Store G_log for saving
    except:
        pass
    fig.tight_layout(pad=2.0)
    return fig
    


def apply_pCF():
    '''
    This function computes the correlation (using the pCF function) for the given kimograms with the provided
    parameters. 

    Returns
    -------
    fig : TYPE figure

    '''
    global G_to_save
    global T_to_save, table_frame
    if original_kimograms is None:
        show_message('Error',"No kimograms data to process.")
        pass
    
    # Get the parameters from the table and apply the pCF function
    # Get the parameters from the table and apply the pCF function
    data = get_table_data()
    #data = get_values()
    first_line = int(data.get("First Line", ""))
    last_line = int(data.get("Last Line", ""))
    line_time = data.get("Line Time (ms)", "")
    print(data)
    dr = int(data.get("Distance (px)", ""))
    #sigma = [int(data.get("H smoothing (px)", "")),int(data.get("V smoothing (lines)", ""))]
    reverse = data.get('Reverse')
    sigma = [
    int(data.get("H Smoothing (px)", 4)),  # Default to 0 if empty
    int(data.get("V Smoothing (lines)", 10))  # Default to 0 if empty
]
    acfnorm = data.get('Normalize')
    #print(f"Processing pCF({dr})")
    
    G, T = pCF(original_kimograms[0][first_line:last_line], line_time/128/1000, dr=dr, reverse_PCF=reverse)    
    x1 = np.geomspace(1, len(G), 256, dtype=int, endpoint = False)    
    t_lineal = T[:,0]
    t_log = np.geomspace(t_lineal[0], t_lineal[-1], 256, endpoint=True)
    G_basura = []
    for i in x1:
        G_basura.append(list(G[i]))
    G = np.asarray(G_basura).transpose()
    t = []
    for i in x1:
        t.append(t_lineal[i])
    t_lineal = np.asarray(t)
    G_log = np.empty_like(G)
    for i, gi in enumerate(G):
        G_log[i] = np.interp(t_log, t_lineal, gi)
    G_log = gaussian_filter(G_log, sigma = sigma)   ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    G_to_save = G_log  # Store G_log for saving
    T_to_save = t_log
    if acfnorm:
        print('Normalizing correlation by ACF...')
        A, _ = pCF(original_kimograms[0][first_line:last_line], line_time/128/1000, dr=0,reverse_PCF=reverse)
        x1 = np.geomspace(1, len(A), 256, dtype=int, endpoint = False)    
        A_basura = []
        for i in x1:
            A_basura.append(list(A[i]))
        A = np.asarray(A_basura).transpose()
        A_log = np.empty_like(A)
        for i, gi in enumerate(A):
            A_log[i] = np.interp(t_log, t_lineal, gi)
        A_log = gaussian_filter(A_log, sigma = sigma) 
        #maxim = A_log[:,0]
        maxim = np.max(A_log, axis=1)
        G_to_save = [G_to_save[i,j]/maxim[i] for i in range(len(G_to_save)) for j in range(len(G_to_save[0]))]
        for i in range(len(G_to_save)):
            if G_to_save[i]>2:
                G_to_save[i]=2
                
        G_to_save = np.reshape(G_to_save, np.shape(G_log))
    #-----------------
    
    if len(original_kimograms)>1:
        G2, T2 = pCF(original_kimograms[1][first_line:last_line], line_time/128/1000, dr=dr, reverse_PCF=reverse)
        x1 = np.geomspace(1, len(G), 256, dtype=int, endpoint = False)    
        t_lineal = T2[:,0]
        t_log = np.geomspace(t_lineal[0], t_lineal[-1], 256, endpoint=True)
        G2_basura = []
        for i in x1:
            G2_basura.append(list(G2[i]))
        G2 = np.asarray(G2_basura).transpose()
        t = []
        for i in x1:
            t.append(t_lineal[i])
        t_lineal = np.asarray(t)
        G2_log = np.empty_like(G2)
        for i, gi in enumerate(G2):
            G2_log[i] = np.interp(t_log, t_lineal, gi)
        G2_log = gaussian_filter(G2_log, sigma = sigma)   ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
        
        G2_to_save = G2_log  # Store G_log for saving
        if acfnorm:
            print('Normalizing correlation by ACF...')
            A, _ = pCF(original_kimograms[1][first_line:last_line], line_time/128/1000, dr=0,reverse_PCF=reverse)
            x1 = np.geomspace(1, len(A), 256, dtype=int, endpoint = False)    
            A_basura = []
            for i in x1:
                A_basura.append(list(A[i]))
            A = np.asarray(A_basura).transpose()
            A_log = np.empty_like(A)
            for i, gi in enumerate(A):
                A_log[i] = np.interp(t_log, t_lineal, gi)
            A_log = gaussian_filter(A_log, sigma = sigma) 
            #maxim = A_log[:,0]
            maxim = np.max(A_log, axis=1)
            G2_to_save = [G2_to_save[i,j]/maxim[i] for i in range(len(G2_to_save)) for j in range(len(G2_to_save[0]))]
            for i in range(len(G2_to_save)):
                if G2_to_save[i]>2:
                    G2_to_save[i]=2
                    
            G2_to_save = np.reshape(G2_to_save, np.shape(G2_log))
        
    # Create a figure with two subplots side by side
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5.5), dpi=150)
        # Plot Channel 1 on the first axis
        plot_on_ax(ax1, G_log,t_log, 'Channel 1')
        
        try: 
            # Plot Channel 2 on the second axis
            plot_on_ax(ax2, G2_log,t_log, 'Channel 2')
            G_to_save = [G_log,G2_log]  # Store G_log for saving
        except:
            pass
    else:
        fig, ax = plt.subplots(figsize=(6, 8))
        plot_on_ax(ax, G_to_save, t_log,f'pCF({dr})')
    fig.tight_layout(pad=2.0)
    
    return fig
    

def save_plot(fig):
    '''
    function to save as a png the plot.
    Parameters
    ----------
    fig : TYPE figure
        DESCRIPTION. figure to be saved

    Returns
    -------
    None.
    '''
    # Open a file dialog to choose the save location and filename
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png")],
        title="Save Plot As"
    )
    if file_path:
        # Save the plot
        fig.savefig(file_path)
        print(f"Plot saved as {file_path}")

def display_plot():
    '''
    This function creates an interface for the computed pCF to be plotted on. It also includes the buttons useful for manipulating the
    resulted kimogram.
    
    Returns
    -------
    None.
    '''
    # Generate the plot
    fig = apply_pCF()
    plt.close('all')
    # Create a Tkinter window
    plot_window = tk.Toplevel()  # Use Toplevel instead of Tk
    plot_window.title("pCF Plot")
    plot_window.geometry("800x600")  # Set a size for visibility

    # Create a frame for the plot
    frame = Frame(plot_window)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create a canvas to put the figure on
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Create a frame for the buttons
    button_frame = Frame(plot_window)
    button_frame.pack(pady=10)  # Add some padding around the button frame

    # Create buttons and pack them side by side
    save_plot_button = tk.Button(button_frame, text="Save Plot", command=lambda: save_plot(fig), **button_style)
    save_plot_button.pack(side=tk.LEFT, padx=1)

    save_data_button = tk.Button(button_frame, text="Save Data", command=save_data, **button_style)
    save_data_button.pack(side=tk.LEFT, padx=2)

    save_time_button = tk.Button(button_frame, text="Save Time", command=save_time, **button_style)
    save_time_button.pack(side=tk.LEFT, padx=3)

    profiles_button = tk.Button(button_frame, text="Profiles", command=open_profiles_window, **button_style)
    profiles_button.pack(side=tk.LEFT, padx=4)

    # Ensure the window is shown
    plot_window.mainloop()  # Ensure to call this for the window to stay open
    



def display_ccplot():
    '''
    this function creates an interface for the computed cross-pcf to be plotted on. 
    It also includes the buttons more useful for manipulating the resulted kimogram.

    Returns
    -------
    None.

    '''
    # Generate the plot
    fig = apply_ccpCF()
    plt.close('all')
    # Create a Tkinter window
    plot_window = tk.Toplevel()  # Use Toplevel instead of Tk
    plot_window.title("ccpCF Plot")
    plot_window.geometry("800x600")  # Set a size for visibility

    # Create a frame for the plot
    frame = Frame(plot_window)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create a canvas to put the figure on
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Create a frame for the buttons
    button_frame = Frame(plot_window)
    button_frame.pack(pady=10)  # Add some padding around the button frame

    # Create buttons and pack them side by side
    print("Creating buttons...")  # Debug print
    save_plot_button = tk.Button(button_frame, text="Save Plot", command=lambda: save_plot(fig), **button_style)
    save_plot_button.pack(side=tk.LEFT, padx=1)

    save_data_button = tk.Button(button_frame, text="Save Data", command=save_data, **button_style)
    save_data_button.pack(side=tk.LEFT, padx=2)

    save_time_button = tk.Button(button_frame, text="Save Time", command=save_time, **button_style)
    save_time_button.pack(side=tk.LEFT, padx=3)

    profiles_button = tk.Button(button_frame, text="Profiles", command=open_profiles_window, **button_style)
    profiles_button.pack(side=tk.LEFT, padx=4)

    # Ensure the window is shown
    plot_window.mainloop()  # Ensure to call this for the window to stay open

def save_time():
    '''
    function to save the correlation time

    Returns
    ------- saved file
    None.
    '''
    global T_to_save
    if T_to_save is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                   filetypes=[("txt files", "*.txt"), ("All files", "*.*")])
        if file_path:
            np.savetxt(file_path, T_to_save)
            show_message('Good', f"Correlation time saved to {file_path}")
    else:
         show_message('Error',"No data to save")
            
def save_data():
    '''
    Function to save the correlation data as a csv file in a choosen file.
    If there are multiple plots it will create multiple files.
    Returns
    -------
    None.

    '''
    global G_to_save
    if G_to_save is not None:
        if isinstance(G_to_save, list):
            for i, matrix in enumerate(G_to_save):
                file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                           filetypes=[("CSV files", "*.csv"),("txt files", '*.txt') ,("All files", "*.*")],
                                                           title=f"Save G{i+1} Matrix")
                if file_path:
                    np.savetxt(file_path, matrix,  delimiter=',', fmt='%d', comments='')
                    show_message('Good', f"Data saved to {file_path}")
        else:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                       filetypes=[("CSV files", "*.csv"),("txt files", '*.txt') , ("All files", "*.*")])
            if file_path:
                np.savetxt(file_path, G_to_save, delimiter=',', comments='')
                show_message('Good', f"Data saved to {file_path}")
    else:
        show_message('Error',"No data to save")


#███████ ██ ██      ████████ ███████ ██████  
#██      ██ ██         ██    ██      ██   ██ 
#█████   ██ ██         ██    █████   ██████  
#██      ██ ██         ██    ██      ██   ██ 
#██      ██ ███████    ██    ███████ ██   ██ 
                                            
                                           

def find_p(f1, f2):
    '''
    This function finds the parameter p needed for the filtering of the channels measured.

    Parameters
    ----------
    f1 : TYPE images or lines with matrix format
        DESCRIPTION. control with only the protein 1
    f2 : TYPE images or lines with matrix format
        DESCRIPTION.control with only the protein 2

    Raises
    ------
    ValueError
        DESCRIPTION. This funtion has been implemented thinking of .czi as controls. If you use some
        other file the shape of the matrix generated may not be correctly interpreted

    Returns
    -------
    p : TYPE (2x2) matrix
        DESCRIPTION.
        filtering parameters.

    '''
    # Check if the shapes are as expected
    try:
        if f1.shape[0] == 1 and f2.shape[0] == 1:
            p = [
                [np.mean(f1[0, :, 0, 0, :, :, 0]), np.mean(f2[0, :, 0, 0, :, :, 0])],
                [np.mean(f1[0, :, 1, 0, :, :, 0]), np.mean(f2[0, :, 1, 0, :, :, 0])]
                ]
            p = np.reshape(p, (2, 2))
            print('I created p correctly')
            return p
    except:
        try:
            if len(f1)==2:
                p = [[np.mean(f1[0]),np.mean(f2[0])],
                     [np.mean(f1[1]),np.mean(f2[1])]]
                p = np.reshape(p, (2, 2))
                print('I created p correctly')
                return p
        except:
            print('Im not understanding the data for p')
            raise ValueError("Input arrays do not have the expected shape.")



def bleeding_filter(kimograms,first_line, last_line,p):
    '''
    this function takes the files provided with the chosen parameters and the p calculated to actually filter the data.

    Parameters
    ----------
    kimograms : TYPE list of matrices each one corresponding to the data uploaded
        
    first_line : TYPE int
        DESCRIPTION. first line to analyse
    last_line : TYPE int
        DESCRIPTION.last line to analyse
    p : TYPE (2x2) matrix
        DESCRIPTION.filtering matrix

    Returns
    -------
    ch1_filtrado : TYPE matrix
        DESCRIPTION. the data provided by the first kimogram but filtered
    ch2_filtrado : TYPE matrix
        DESCRIPTION.  the data provided by the second kimogram but filtered

    '''
    ch1 = kimograms[0][first_line:last_line]
    ch2 = kimograms[1][first_line:last_line]
    ch1_filtrado, ch2_filtrado = ch1.astype(float, copy=True), ch1.astype(float, copy=True)
    for j in range(ch1.shape[0]):
        for k in range(ch1.shape[1]):
            try:
                a = np.linalg.solve(p, [ch1[j, k], ch2[j, k]])
                ch1_filtrado[j, k] = max(a[0], 0)
                ch2_filtrado[j, k] = max(a[1], 0)
            except np.linalg.LinAlgError as e:
                print(f"Error solving for indices {j}, {k}: {e}")
                ch1_filtrado[j, k] = 0
                ch2_filtrado[j, k] = 0
    print(f'ch1 filtrado shape = {np.shape(ch1_filtrado)}')
    
    return ch1_filtrado, ch2_filtrado




def plot_two_kimograms(t_log, G_log, G2_log, cruzado=True):
    '''
    This function plots the correlation calculated somewhere else.
    '''
    global G_to_save
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5.5), dpi=150)

    if cruzado:
        l1 = 'Cross 1 to 2'
        l2 = 'Cross 2 to 1'
    else:
        l1 = 'Channel 1'
        l2 = 'Channel 2'

    try:
        if G_log is not None and G2_log is not None:
            plot_on_ax(ax1, G_log, t_log, l1)
            plot_on_ax(ax2, G2_log, t_log, l2)
            G_to_save = [G_log, G2_log]  # Store G_log for saving
        else:
            print("G_log or G2_log is None")
            return None
    except Exception as e:
        print(f"Error in plot_two_kimograms: {e}")
        return None

    fig.tight_layout(pad=2.0)
    return fig


    
def plot_filtered_data(ch1_filtrado, ch2_filtrado, line_time, dr, sigma, reverse, cruzado=True):
    '''
    Computes the corresponding pair correlation for the filtered data.
    '''
    global G_to_save, T_to_save
    try:
        if cruzado:
            G, T = crosspCF(ch1_filtrado, ch2_filtrado, linetime=line_time / 1000, dr=dr, reverse_PCF=reverse)
            G2, T2 = crosspCF(ch2_filtrado, ch1_filtrado, linetime=line_time / 1000, dr=dr, reverse_PCF=reverse)
            print('LA CORRELACION ES CRUZADA')
        else:
            print(f'ch1 shape: {ch1_filtrado.shape}, ch2 shape: {ch2_filtrado.shape}')
            print(f'dr: {dr}, tp: {line_time / 100 / ch1_filtrado.shape[1]}')
            print(f'reverse: {reverse}')
            G, T = crosspCF(ch1_filtrado, ch1_filtrado, linetime=line_time / 1000, dr=dr, reverse_PCF=reverse)
            print('LA CORRELACION ES EN CADA CANAL POR SEPARADO')

            G2, T2 = crosspCF(ch2_filtrado, ch2_filtrado, linetime=line_time / 1000, dr=dr, reverse_PCF=reverse)
        print(f"G shape: {G.shape}, G2 shape: {G2.shape}")  # Debug print

        if len(G) == 0 or len(G2) == 0:
            show_message('Error', "G or G2 returned from crosspCF is empty.")
            return None

        x1 = np.geomspace(1, len(G), 256, dtype=int, endpoint=False)
        t_lineal = T[:, 0]
        t_log = np.geomspace(t_lineal[0], t_lineal[-1], 256, endpoint=True)

        G_basura = []
        for i in x1:
            G_basura.append(list(G[i]))
        G = np.asarray(G_basura).transpose()

        t = []
        for i in x1:
            t.append(t_lineal[i])
        t_lineal = np.asarray(t)

        G_log = np.empty_like(G)
        for i, gi in enumerate(G):
            G_log[i] = np.interp(t_log, t_lineal, gi)

        G_log = gaussian_filter(G_log, sigma=sigma)
        x1 = np.geomspace(1, len(G2), 256, dtype=int, endpoint=False)

        G2_basura = []
        for i in x1:
            G2_basura.append(list(G2[i]))
        G2 = np.asarray(G2_basura).transpose()

        G2_log = np.empty_like(G2)
        for i, gi in enumerate(G2):
            G2_log[i] = np.interp(t_log, t_lineal, gi)
        G2_log = gaussian_filter(G2_log, sigma=sigma)

        G_to_save = [G_log, G2_log]
        T_to_save = t_log

        return plot_two_kimograms(t_log, G_log, G2_log, cruzado)

    except Exception as e:
        print(f"Error in plot_filtered_data: {e}")
        show_message('Error', f"An error occurred while plotting filtered data: {e}")
        return None






# Initialize global variables for file paths
global f1, f2, f1_extra, f2_extra
f1, f2,f1_extra, f2_extra = [], [], [], []

def upload_filter_files():
    '''
    Function for uploading files from controls to filter the collected data.
    
    Returns
    -------
    None.
    '''
    global f1, f2, f1_extra, f2_extra
    f1 = filedialog.askopenfilenames(filetypes=[("Control files for protein 1", "*.tiff"), 
                                             ("Control files for protein 1", "*.tif"),
                                             ("Control files for protein 1", "*.czi"),
                                             ("Control files for protein 1", "*.b64")])
    
    if f1:
        # Check if any of the selected files are .b64
        if any(file.endswith('.b64') for file in f1):
            # Upload additional files for .B64 case
            f1_extra = filedialog.askopenfilenames(filetypes=[("Additional files for protein 1", "*.tiff;*.tif;*.czi;*.b64")])
            f2 = filedialog.askopenfilenames(filetypes=[("Control files for protein 2", "*.tiff;*.tif;*.czi,;*.b64")])
            f2_extra = filedialog.askopenfilenames(filetypes=[("Additional files for protein 2", "*.tiff;*.tif;*.czi;*.b64")])
            
            if f1_extra and f2_extra:
                file_list_label.config(text=(
                    "Files for protein 1: uploaded"
                    "Files for protein 2: uploaded"
                ))
                pass
            else:
                file_list_label.config(text="Not all additional files were selected.")
        else:
            f2 = filedialog.askopenfilenames(filetypes=[("Control files for protein 2", "*.tiff"), 
                                             ("Control files for protein 2", "*.tif"),
                                             ("Control files for protein 2", "*.czi"),
                                             ("Control files for protein 2", "*.b64")])
            if f2:
                file_list_label.config(text="Files uploaded")
                pass
            else:
                file_list_label.config(text="No files selected for protein 2")
    else:
        file_list_label.config(text="No files selected for protein 1")

def apply_filter_ccpcf(cross):
    '''
    this function takes the uploaded filtering files, transform them into matrixs, performes the filtering (using the bleeding_filter function)
    then plots the filter correlation. It provides two plots. Either ther pCF for both channels or the ccpCF in both directions.

    Parameters
    ----------
    cross : TYPE str
        DESCRIPTION. either 'ccpcf' or 'pcf'. 'pcf' will be interpreted otherwise

    Returns
    -------
    the plots

    '''
    global original_kimograms, f1, f2, f1_extra, f2_extra
    if f1 and f2:
        try:
            f1_data = []
            f2_data = []
            
            for file in f1:
                if file.endswith(('.tiff', '.tif')):
                    f1_data.append(np.array(Image.open(file)))
                elif file.endswith('.czi'):
                    f1_data = imread(file)  # Load CZI file
                elif file.endswith('.b64'):
                    f1_data.append(read_B64(file))
                    print('I appended the files in f1')
                    f1_data.append(read_B64(f1_extra[0]))
                    print('I appended the file for f1_extra')

            for file in f2:
                if file.endswith(('.tiff', '.tif')):
                    f2_data.append(np.array(Image.open(file)))
                elif file.endswith('.czi'):
                    f2_data = imread(file)  # Load CZI file
                elif file.endswith('.b64'):
                    f2_data.append(read_B64(file))
                    print('I appended f2')
                    f2_data.append(read_B64(f2_extra[0]))
                    print('I appended the file for f2_extra')
                    
            try:
                p = find_p(f1_data, f2_data)
                print('find_p output:', p)  # Debug: check the output of find_p
            except:
                print('error finding p')
            if p is None or not isinstance(p, np.ndarray):
                raise ValueError("Invalid output from find_p function.")
            data = get_table_data()
            first_line = int(data.get("First Line", 0))  # Default to 0 if not found
            last_line = int(data.get("Last Line", 0))    # Default to 0 if not found
            line_time = float(data.get("Line Time (ms)", 0.0))  # Default to 0.0
            dr = int(data.get("Distance (px)", 0))      # Default to 0 if not found
            sigma = [int(data.get("H Smoothing (px)", 0)), int(data.get("V Smoothing (lines)", 0))]  # Default to 0 if not found
            reverse = data.get('Reverse', False)  # Default to False if not found
            print('the table data is ok')
           
            
            ch1_filtrado, ch2_filtrado = bleeding_filter(original_kimograms,first_line, last_line,p)
            if cross=='ccpcf':
                print('Entré al if cross de apply_filter_ccpcf')
                fig = plot_filtered_data(ch1_filtrado, ch2_filtrado, line_time, dr, sigma, reverse, cruzado=True)
            else:
                try:
                    fig = plot_filtered_data(ch1_filtrado, ch2_filtrado, line_time, dr, sigma, reverse, cruzado=False)
                except:
                    print('error in plot_filtered_data')
                    pass
            
            return fig
        except:
            print("Error in apply_filter_ccpcf function")
            
    


def display_filter_ccpcf(cross):
    '''
    displays the filtered correlation plots in a new window

    Parameters
    ----------
    cross : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    fig = apply_filter_ccpcf(cross)
    plt.close('all')

    # Create a Tkinter window
    plot_window = tk.Toplevel()  # Use Toplevel instead of Tk
    if cross == 'ccpcf':
        plot_window.title("Filtered ccpCF Plot")
    else:
        plot_window.title("Filtered pCF Plot")
    
    plot_window.geometry("800x600")  # Set a size for visibility

    # Create a frame for the plot
    frame = Frame(plot_window)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create a canvas to put the figure on
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Create a frame for the buttons
    button_frame = Frame(plot_window)
    button_frame.pack(pady=10)  # Add some padding around the button frame
    
    # Create buttons and pack them side by side
   # print("Creating buttons...")  # Debug print
    save_plot_button = tk.Button(button_frame, text="Save Plot", command=lambda: save_plot(fig), **button_style)
    save_plot_button.pack(side=tk.LEFT, padx=1)

    save_data_button = tk.Button(button_frame, text="Save Data", command=save_data, **button_style)
    save_data_button.pack(side=tk.LEFT, padx=2)

    save_time_button = tk.Button(button_frame, text="Save Time", command=save_time, **button_style)
    save_time_button.pack(side=tk.LEFT, padx=3)

    profiles_button = tk.Button(button_frame, text="Profiles", command=open_profiles_window, **button_style)
    profiles_button.pack(side=tk.LEFT, padx=4)

    # Ensure the window is shown
    plot_window.mainloop()  # Ensure to call this for the window to stay open
    

    
                
                
def open_profiles_window():
    '''
    plot vertical profiles in kimograms, i.e. the temporal evolution for a pixel or an average of as many pixels as wanted. 

    Returns
    -------
    the plot

    '''
    profile_window = Toplevel(root)
    profile_window.title("Select Profile Pixel Range")

    def plot_profile():
        global G_to_save, T_to_save
        start_pixel = start_pixel_var.get()
        num_pixels = num_pixels_var.get()
        selected_channel = channel_var.get()        
        end_pixel = start_pixel + num_pixels
        try:
            if end_pixel > G_to_save[0].shape[0]:
                show_message("Error","End pixel index exceeds data range.")
                return

            # Calculate the average profile over the specified range for both channels
            profiles_channel1 = G_to_save[0][start_pixel:end_pixel, :]
            average_profile1 = np.mean(profiles_channel1, axis=0)
            
            profiles_channel2 = G_to_save[1][start_pixel:end_pixel, :]
            average_profile2 = np.mean(profiles_channel2, axis=0)
            
            # Create a new figure with adjusted size
            profile_fig = Figure(figsize=(12, 6), dpi=100)
            profile_ax1 = profile_fig.add_subplot(121)  # Channel 1 subplot
            profile_ax2 = profile_fig.add_subplot(122)  # Channel 2 subplot

            if selected_channel == 'Channel 1':
                profile_ax1.plot(T_to_save, average_profile1, color='black')
                profile_ax1.set_title(f'Channel 1 Profile from Pixel {start_pixel} to {end_pixel - 1}', fontsize=14)
                profile_ax1.set_xlabel('Time', fontsize=12)
                profile_ax1.set_ylabel('Intensity', fontsize=12)
                profile_ax1.tick_params(axis='both', which='major', labelsize=10)
                profile_ax1.tick_params(axis='both', which='minor', labelsize=8)
                profile_ax1.grid(True)
                profile_ax1.set_xscale('log')
                profile_ax1.set_yscale('linear')  # or 'log' depending on your data
                profile_ax1.set_xlim(T_to_save.min(), T_to_save.max())
                profile_ax1.set_ylim(np.min(average_profile1), np.max(average_profile1))

                # Hide Channel 2 subplot
                profile_ax2.axis('off')

            elif selected_channel == 'Channel 2':
                profile_ax2.plot(T_to_save, average_profile2, color='black')
                profile_ax2.set_title(f'Channel 2 Profile from Pixel {start_pixel} to {end_pixel - 1}', fontsize=14)
                profile_ax2.set_xlabel('Time', fontsize=12)
                profile_ax2.set_ylabel('Intensity', fontsize=12)
                profile_ax2.tick_params(axis='both', which='major', labelsize=10)
                profile_ax2.tick_params(axis='both', which='minor', labelsize=8)
                profile_ax2.grid(True)
                profile_ax2.set_xscale('log')
                profile_ax2.set_yscale('linear')  # or 'log' depending on your data
                profile_ax2.set_xlim(T_to_save.min(), T_to_save.max())
                profile_ax2.set_ylim(np.min(average_profile2), np.max(average_profile2))

            
        except:
            if type(G_to_save)==list:
                G_to_save=G_to_save[0]
                if end_pixel > G_to_save.shape[0]:
                    show_message("Error","End pixel index exceeds data range.")
                    return
                else:
                    # Calculate the average profile over the specified range
                    profiles = G_to_save[start_pixel:end_pixel, :]
                    average_profile = np.mean(profiles, axis=0)

                    # Create a new figure with adjusted size
                    profile_fig = Figure(figsize=(8, 6), dpi=100)
                    profile_ax = profile_fig.add_subplot(111)
                    profile_ax.plot(T_to_save, average_profile, color='black')
                    
                    # Set labels with larger font size
                    profile_ax.set_title(f'Average Profile from Pixel {start_pixel} to {end_pixel - 1}', fontsize=14)
                    profile_ax.set_xlabel('Time', fontsize=12)
                    profile_ax.set_ylabel('Intensity', fontsize=12)
                    
                    # Increase padding and font size for ticks
                    profile_ax.tick_params(axis='both', which='major', labelsize=10)
                    profile_ax.tick_params(axis='both', which='minor', labelsize=8)
                    
                    # Add grid for better readability
                    profile_ax.grid(True)
                    profile_ax.set_xscale('log')  
                    return
            if end_pixel > G_to_save.shape[0]:
                show_message("Error","End pixel index exceeds data range.")
                return
            else:
                # Calculate the average profile over the specified range
                profiles = G_to_save[start_pixel:end_pixel, :]
                average_profile = np.mean(profiles, axis=0)

                # Create a new figure with adjusted size
                profile_fig = Figure(figsize=(8, 6), dpi=100)
                profile_ax = profile_fig.add_subplot(111)
                profile_ax.plot(T_to_save, average_profile, color='black')
                
                # Set labels with larger font size
                profile_ax.set_title(f'Average Profile from Pixel {start_pixel} to {end_pixel - 1}', fontsize=14)
                profile_ax.set_xlabel('Time', fontsize=12)
                profile_ax.set_ylabel('Intensity', fontsize=12)
                
                # Increase padding and font size for ticks
                profile_ax.tick_params(axis='both', which='major', labelsize=10)
                profile_ax.tick_params(axis='both', which='minor', labelsize=8)
                
                # Add grid for better readability
                profile_ax.grid(True)
                profile_ax.set_xscale('log')       
                # Create and add the canvas
                           
        # Create and add the canvas
        profile_canvas = FigureCanvasTkAgg(profile_fig, master=profile_window)
        profile_canvas.draw()

        # Use grid() to place the canvas
        profile_canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, sticky='nsew')

        def on_click(event):
            if event.inaxes is not None:
                x_data = event.xdata
                y_data = event.ydata
                print(f"Clicked at x={x_data:.4f}, y={y_data:.4f}")
                # Update or create a label to show these coordinates
                coord_label.config(text=f"Coordinates: x={x_data:.4f}s, y={y_data:.6f}")

        # Connect the event handler
        profile_canvas.mpl_connect('button_press_event', on_click)

        # Create a label to display the coordinates
        coord_label = Label(profile_window, text="Coordinates: x= , y=")
        coord_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

    # Add entries for starting pixel and number of pixels
    start_pixel_var = IntVar(value=0)
    num_pixels_var = IntVar(value=1)
    channel_var = tk.StringVar(value='Channel 1')  # Default to 'Channel 1'
    
    start_pixel_label = Label(profile_window, text="Start Pixel:")
    start_pixel_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
    
    start_pixel_entry = Entry(profile_window, textvariable=start_pixel_var)
    start_pixel_entry.grid(row=0, column=1, padx=10, pady=10, sticky='ew')
    
    num_pixels_label = Label(profile_window, text="Number of Pixels to Average:")
    num_pixels_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')
    
    num_pixels_entry = Entry(profile_window, textvariable=num_pixels_var)
    num_pixels_entry.grid(row=1, column=1, padx=10, pady=10, sticky='ew')

    channel_label = Label(profile_window, text="Select Channel:")
    channel_label.grid(row=2, column=0, padx=10, pady=10, sticky='w')

    channel_menu = tk.OptionMenu(profile_window, channel_var, 'Channel 1', 'Channel 2')
    channel_menu.grid(row=2, column=1, padx=10, pady=10, sticky='ew')
    
    plot_button = tk.Button(profile_window, text="Plot Profile", command=plot_profile)
    plot_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky='ew')  # Adjust row to avoid overlap

    profile_window.grid_columnconfigure(1, weight=1)
    profile_window.grid_rowconfigure(3, weight=1)  # Allow row 3 to expand if needed
    
    
def compute_and_display_plot(cross):
    fig = apply_filter_ccpcf(cross)  # Run the filtering computation only once
    root.after(0, display_filter_ccpcf, fig)  # Schedule the plot in the main thread

def display_filter_ccpcf(fig):
    '''
    Displays the filtered correlation plots in a new window.

    Parameters
    ----------
    fig : matplotlib figure
        The computed figure to display.

    Returns
    -------
    None
    '''
    plt.close('all')

    # Create a Tkinter window
    plot_window = tk.Toplevel()  # Use Toplevel instead of Tk
    plot_window.title("Filtered ccpCF Plot")
    plot_window.geometry("800x600")  # Set a size for visibility

    # Create a frame for the plot
    frame = Frame(plot_window)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create a canvas to put the figure on
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Create a frame for the buttons
    button_frame = Frame(plot_window)
    button_frame.pack(pady=10)  # Add some padding around the button frame
    
    # Create buttons and pack them side by side
    save_plot_button = tk.Button(button_frame, text="Save Plot", command=lambda: save_plot(fig), **button_style)
    save_plot_button.pack(side=tk.LEFT, padx=1)

    save_data_button = tk.Button(button_frame, text="Save Data", command=save_data, **button_style)
    save_data_button.pack(side=tk.LEFT, padx=2)

    save_time_button = tk.Button(button_frame, text="Save Time", command=save_time, **button_style)
    save_time_button.pack(side=tk.LEFT, padx=3)

    profiles_button = tk.Button(button_frame, text="Profiles", command=open_profiles_window, **button_style)
    profiles_button.pack(side=tk.LEFT, padx=4)

    # Ensure the window is shown
    plot_window.mainloop()  # Ensure to call this for the window to stay open

    

import webbrowser

def open_help():
    url = "https://github.com/natiph/pcf_interface/wiki/Tutorial"  # Replace with your actual URL
    webbrowser.open(url)



# This function will just be an event handler that does nothing for now
#def on_cmap_change(event):
    #pass  # You can leave this empty or add any functionality you want

def on_cmap_change(event):
    selected_cmap = cmap_var.get()
    global original_kimograms
    for fig in current_figures:
        for ax in fig.axes:
            # Clear the current collections in the axes
            for collection in ax.collections:
                collection.remove()  # Remove existing collections

            # Check if original_kimograms is not None
            if original_kimograms is not None:
                # Assuming we want to plot the first kimogram
                kimogram = original_kimograms[0]
                print(f"Shape of original_kimograms[0]: {kimogram.shape}")

                try:
                    # Make sure to transpose the array if necessary
                    new_im = ax.pcolor(kimogram.T, cmap=selected_cmap, shading='nearest')

                    # Remove existing color bars if present
                    if hasattr(fig, 'colorbar'):
                        for cbar in fig.colorbars:
                            cbar.remove()
                    
                    # Add the new color bar
                    cbar = fig.colorbar(new_im, ax=ax, orientation='vertical', format='%.2f')
                    fig.colorbars.append(cbar)  # Keep track of colorbars

                except ValueError as e:
                    print(f"ValueError: {e} with shapes {kimogram.shape}")
                    continue  # Skip this axis if there's an error

        fig.canvas.draw_idle()  # Redraw the figure
    
# Initialize the Tkinter window
root = tk.Tk()

from tkinter import PhotoImage
from PIL import Image, ImageTk

# Load an image file (e.g., .jpg or .bmp) using Pillow and convert to PhotoImage
image = Image.open(codepath+r'\interface_icon.jpg')  # or .bmp
photo = ImageTk.PhotoImage(image)

# Set the window icon
root.iconphoto(False, photo)
# Create a main frame with white background
main_frame = tk.Frame(root, bg='white', padx=15, pady=15)
main_frame.grid(row=0, column=0, sticky='nsew')
root.option_add('*Font', 'Helvetica 10')
root.option_add('*Background', 'white')
root.option_add('*Foreground', 'black')

fondo = 'white'
# Button style
button_style = {
    'bg': 'lightgrey',              # Dark grey background color for buttons
    'fg': 'black',                # White text color
    'padx': 12,                   # More padding for a larger button
    'pady': 6,                    # More padding for a larger button
    'relief': 'flat',             # Flat button style for a modern look
    'borderwidth': 2,             # Slightly thicker border
    'highlightbackground': '#d3d3d3',  # Subtle border color when button is not focused
    'highlightcolor': '#a9a9a9',  # Subtle border color when button is focused
    'activebackground': '#5a5a5a', # Darker grey background when the button is pressed
    'activeforeground': 'white'   # White text color when the button is pressed
}
#load icons

try:
    original_image = Image.open(codepath+r'\zoomin.png')
    resized_image = original_image.resize((32, 32))  # Resize to 32x32 pixels
    zoomin_icon = ImageTk.PhotoImage(resized_image)
except Exception as e:
    print(f"Error loading or resizing image: {e}")
try:
    original_image = Image.open(codepath+r'\zoomout.png')
    resized_image = original_image.resize((32, 32))  # Resize to 32x32 pixels
    zoomout_icon = ImageTk.PhotoImage(resized_image)
except Exception as e:
    print(f"Error loading or resizing image: {e}")


#root.title("Kimogram Viewer with Editable Table")
root.title("pCF by Naty")
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

control_frame = tk.Frame(root, bg=fondo)
control_frame.grid(row=0, column=0, pady=10, sticky="ew")

load_button = tk.Button(control_frame, text="Load Lines", command=load_lines,**button_style)
load_button.grid(row=0, column=0, padx=5)

load_button = tk.Button(control_frame, text="Load pCF files", command=load_correlation,**button_style)
load_button.grid(row=0, column=1, padx=5)
image_frame = tk.Frame(root, bg=fondo)
image_frame.grid(row=1, column=0, sticky="nsew")

image_frame.grid_rowconfigure(0, weight=1)
image_frame.grid_columnconfigure(0, weight=1)

button_frame = tk.Frame(root, bg=fondo)
button_frame.grid(row=2, column=0, pady=5, sticky="ew")


# Frame for the table
table_frame = tk.Frame(root, bg=fondo)
table_frame.grid(row=3, column=0, pady=2, sticky="nsew")

# Table headers and entries
labels = ["Line Time (ms)","First Line", "Last Line", 'Distance (px)','H smoothing (px)' ,'V smoothing (lines)', 'Reverse', 'Normalize']
entries = []
checkbutton_vars = {}  # Dictionary to store BooleanVars for Checkbuttons

# for row, label in enumerate(labels):
#     tk.Label(table_frame, text=label, borderwidth=2, relief="solid").grid(row=row, column=0, padx=5, pady=5, sticky="e")
#     entries.append([])


for row, label in enumerate(labels):
    tk.Label(table_frame, text=label, borderwidth=2, relief="solid").grid(row=row, column=0, padx=5, pady=5, sticky="e")
    
    if label == 'Reverse' or label== 'Normalize':
        # Create a Checkbutton for the "Reverse" label
        var = tk.BooleanVar()  # Variable to hold the state of the checkbox
        checkbox = tk.Checkbutton(table_frame, variable=var)
        entries.append(checkbox)  # Append the Checkbutton to the entries list
        checkbutton_vars[label] = var  # Store the BooleanVar in the dictionary with the label as the key
        checkbox.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        
    elif label =='Line time (ms)':
        if dwell_time:
            entry = tk.Entry(table_frame)
            entry.insert(0, f'{dwell_time}')  # Default value for H smoothing
            entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
    elif label == 'H smoothing (px)':
        entry = tk.Entry(table_frame)
        entry.insert(0, '4')  # Default value for H smoothing
        entries.append(entry)  # Append the Entry widget to the entries list
        entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
    elif label == 'V smoothing (lines)':
        entry = tk.Entry(table_frame)
        entry.insert(0, '10')  # Default value for V smoothing
        entries.append(entry)  # Append the Entry widget to the entries list
        entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
    else:
        entry = tk.Entry(table_frame)
        entries.append(entry)  # Append the Entry widget to the entries list
        entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
# Add some default rows

# Frame for table buttons
table_button_frame = tk.Frame(root)
table_button_frame.grid(row=2, column=0, pady=10, sticky="ew")

# Button to apply pCF

apply_pCF_button = tk.Button(table_button_frame, text="Apply pCF", command=display_plot, **button_style)
apply_pCF_button.grid(row=0, column=0, padx=5)

# Button to save pCF data
#save_button = tk.Button(table_button_frame, text="Save correlation data", command=save_data, **button_style)
#save_button.grid(row=0, column=1, padx=6)

# Button to save plot (include an image if needed, replace 'icon' with actual image variable)
saveplot_button = tk.Button(table_button_frame, text="Save Plot", command=lambda: print("Save Plot"), **button_style)  # Replace 'image=icon' if needed
saveplot_button.grid(row=0, column=1, padx=6)

# Button to apply ccpCF
apply_ccpCF_button = tk.Button(table_button_frame, text="Apply ccpCF", command=display_ccplot, **button_style)
apply_ccpCF_button.grid(row=0, column=2, padx=10)

# Button to open the new window
filterfiles_button = tk.Button(table_button_frame, text="Spectral filter", command=upload_filter_files)
filterfiles_button.grid(row=0, column=4, padx=10)#, pady=20)  # Place button in the grid
file_list_label = tk.Label(table_button_frame, text="No files selected")
file_list_label.grid(row=1, column=4, pady=10)





apply_fccpCF_button = tk.Button(table_button_frame, text="Apply filtered ccpCF", 
                                 command=lambda: display_filter_computation('ccpcf'), **button_style)
apply_fccpCF_button.grid(row=0, column=6, padx=5)

apply_pCF_button = tk.Button(table_button_frame, text="Apply filtered pCF", 
                             command=lambda: display_filter_computation('pCF'), **button_style)
apply_pCF_button.grid(row=0, column=5, padx=5)


canvas = None
rect_selector = None

controls_frame = tk.Frame(root)
controls_frame.grid(row=0, column=0, sticky='ns', padx=15, pady=10)

# Create a frame for navigation buttons
nav_frame = tk.Frame(controls_frame)
nav_frame.grid(row=0, column=0, sticky='ew')

# Create and place navigation buttons
#profiles_button = tk.Button(nav_frame, text="Profiles", command=open_profiles_window, **button_style)
#profiles_button.pack(side=tk.LEFT, padx=5, pady=10)

h_lines_button = tk.Button(nav_frame, text="H-lines", command=open_hlines_window, **button_style)
h_lines_button.pack(side=tk.LEFT, padx=5, pady=10)

# Create a frame for zoom buttons
zoom_frame = tk.Frame(controls_frame)
zoom_frame.grid(row=0, column=1, sticky='ew')

zoom_in_button = tk.Button(zoom_frame, text="Zoom In", command=zoom_in, image=zoomin_icon, **button_style)
zoom_in_button.pack(side=tk.LEFT, padx=5, pady=10)

zoom_out_button = tk.Button(zoom_frame, text="Zoom Out", command=zoom_out, image=zoomout_icon, **button_style)
zoom_out_button.pack(side=tk.LEFT, padx=5, pady=10)



def update_kimogram_label():
    # Update the label to show the current kimogram
    kimogram_label.config(text=f"Current Kimogram: {current_index+1}")
    kimogram_label.grid(row=1, column=6, pady=(5, 0))  # Add some padding above the label
# Create a button to toggle between kimograms
toggle_button = tk.Button(controls_frame, text="Toggle Kimogram", command=toggle_kimogram, **button_style)
toggle_button.grid(row=0, column=2, padx=5, pady=10)
# Create a label to show the current kimogram
kimogram_label = tk.Label(controls_frame, text=f"Current Kimogram: {current_index+1}")



# Create a dropdown for color maps
cmap_var = tk.StringVar(value='choose color map')
cmap_dropdown = ttk.Combobox(controls_frame, textvariable=cmap_var, state='readonly')
cmap_dropdown['values'] = ['smooth_viridis', 'plasma', 'cividis', 'jet']
cmap_dropdown.bind("<<ComboboxSelected>>", on_cmap_change)  # Binding to the event handler
cmap_dropdown.grid(row=0, column=3, pady=10, padx=5, sticky='ew')

# Help button
help_button = tk.Button(controls_frame, text="Help", command=open_help, **button_style)
help_button.grid(row=0, column=4, padx=5, pady=10)

# Adjust the grid weights for better resizing
controls_frame.grid_columnconfigure(0, weight=1)
controls_frame.grid_columnconfigure(1, weight=1)
controls_frame.grid_columnconfigure(2, weight=1)
controls_frame.grid_columnconfigure(3, weight=1)
controls_frame.grid_columnconfigure(4, weight=1)

root.mainloop()
