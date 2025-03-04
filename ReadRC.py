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

# Select folder
Tk().withdraw()
myDir = filedialog.askdirectory()
myFiles = [f for f in os.listdir(myDir) if f.endswith('.tiff')]

# Load calibration parameters
pars = np.loadtxt('CalibParameters.txt').reshape(3, 3)
redCali, greenCali, blueCali = pars[:, 0], pars[:, 1], pars[:, 2]

results = []

for filename in myFiles:
    path = os.path.join(myDir, filename)
    pic = cv2.imread(path)
    pic_rgb = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

    plt.imshow(pic_rgb)
    plt.title('Original image')
    plt.show()

    # Crop manually (this opens an interactive crop window)
    from matplotlib.widgets import RectangleSelector
    from PIL import Image

    fig, ax = plt.subplots()
    ax.imshow(pic_rgb)
    coords = []

    def onselect(eclick, erelease):
        coords.append((int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)))
    rect_selector = RectangleSelector(ax, onselect)
    plt.show()

    x1, y1, x2, y2 = coords[0]
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
    s, q, p, z, r, En = 2.2717, -0.699, 1.75, 4, 2.08, 10
    EW = ((En - s * En**q)**p - z / r)**(1/p)
    E = (En**p - z / r)**(1/p)
    a, b, c, d = 4.1e5, 2.88, 22.5, 0.142
    LET = a * np.exp(-b * E) + c * np.exp(-d * E)
    LETW = a * np.exp(-b * EW) + c * np.exp(-d * EW)
    A, B = 0.010, 1.09
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
        'Corr2AvDose': dose_av_W,
        'Corr2stdDose': std_dose_W
    })

# Save results
df = pd.DataFrame(results)
df.to_csv('DoseValues.txt', index=False)
