import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import pandas as pd

# Colormaps
redmap = np.array([[255 - i, 0, 0] for i in range(1, 256)])
greenmap = np.array([[0, 255 - i, 0] for i in range(1, 256)])
bluemap = np.array([[0, 0, 255 - i] for i in range(1, 256)])

#__________Cargar Archivo____________
#La carpeta debe estar en el mismo sitio que este script.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Nombre de la carpeta que quieres abrir
folder_name = "analysis"
# Ruta completa a esa carpeta
myDir = os.path.join(script_dir, folder_name)
myFiles = [f for f in os.listdir(myDir) if f.endswith('.tiff')]

# Load calibration parameters
pars = np.loadtxt('CalibParameters.txt').reshape(3, 3)
redCali, greenCali, blueCali = pars[:, 0], pars[:, 1], pars[:, 2]

results = []

for filename in myFiles:
    path = os.path.join(myDir, filename)
    pic = cv2.imread(path)
    pic_rgb = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

    #plt.imshow(pic_rgb)
    #plt.title('Original image')
    #plt.show()

    # Crop manually (this opens an interactive crop window)
    from matplotlib.widgets import RectangleSelector
    from PIL import Image

    fig, ax = plt.subplots()
    ax.imshow(pic_rgb)
    coords = []
    

    def onselect(eclick, erelease):
        global coords
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        coords = [x1, y1, x2, y2]
        print(f'Selección: ({x1}, {y1}) --> ({x2}, {y2})')
    #coords.append((int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)))
    #rect_selector = RectangleSelector(ax, onselect)
    rect_selector = RectangleSelector(
    ax, onselect, 
    useblit=True,
    button=[1],  # Botón izquierdo del mouse
    minspanx=5, minspany=5,
    spancoords='pixels',
    interactive=True)    
   
    def on_double_click(event):
        if event.dblclick and coords:
            if len(coords) == 4:  # Asegurar que hay coordenadas válidas
                x, y = int(event.xdata), int(event.ydata)
                x1, y1, x2, y2 = coords
                if x1 <= x <= x2 and y1 <= y <= y2:
                    print(f"Doble clic dentro del área seleccionada en ({x}, {y}). Cerrando ventana.")
                    plt.close()
                else:
                    print(f"Doble clic fuera del área seleccionada ({x}, {y}). Ignorado.")
            else:
                print("No hay coordenadas válidas. Selecciona un área primero.")

  
    fig.canvas.mpl_connect('button_press_event', on_double_click)

    plt.show()
    if coords:
        x1, y1, x2, y2 = coords
        cut = pic_rgb[y1:y2, x1:x2]
    cut_R, cut_G, cut_B = cut[:, :, 0], cut[:, :, 1], cut[:, :, 2] 
    

    def mean2(x): return np.mean(x)
    def std2(x): return np.std(x)

    dose = [
        redCali[0] + (redCali[1] / (mean2(cut_R) - redCali[2])),
        greenCali[0] + (greenCali[1] / (mean2(cut_G) - greenCali[2])),
        blueCali[0] + (blueCali[1] / (mean2(cut_B) - blueCali[2]))
    ]

    std = [
        np.sqrt((-(redCali[1] / ((dose[0] - redCali[2])**2)))**2 * (std2(cut_R)**2)),
        np.sqrt((-(greenCali[1] / ((dose[1] - greenCali[2])**2)))**2 * (std2(cut_G)**2)),
        np.sqrt((-(blueCali[1] / ((dose[2] - blueCali[2])**2)))**2 * (std2(cut_B)**2))
    ]

    dose_av = np.mean(dose)
    std_dose = np.std(dose)

    # Correction factor
    s=2.2717 
    q=-0.699 
    p=1.75
    z=4 #Air gap that needs to be adjusted for each experiment
    r=2.08
    En=10
    EW = ((En - s * En**q)**p - z / r)**(1/p) #Taking into account the energy loss at the exit window
    E = (En**p - z / r)**(1/p) #Without taking into account the energy loss at the exit window
    a=4.1e5 
    b=2.88 
    c=22.5 
    d=0.142
    LET = (a * np.exp(-b * E)) + (c * np.exp(-d * E))
    LETW =(a * np.exp(-b * EW)) + (c * np.exp(-d * EW))
    A=0.010
    B=1.09
    RE = 1 - A * LET**B
    REW = 1 - A * LETW**B


    dose_q = [d * RE for d in dose]
    dose_W = [d * REW for d in dose]
    std_dose_q = std_dose * RE
    std_dose_W = std_dose * REW
    dose_av_q = dose_av * RE
    dose_av_W = dose_av * REW

    results.append({
        'picName': filename,
        'AvDose': dose_av,
        'Corr2AvDose': dose_av_W,
        'Corr2stdDose': std_dose_W
    })

# Save results
df = pd.DataFrame(results)
df.to_csv('DoseValues.txt', index=False)
