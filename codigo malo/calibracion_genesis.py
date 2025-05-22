import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askdirectory
from matplotlib.widgets import RectangleSelector
from scipy.optimize import curve_fit

# Hide the main Tkinter window
root = Tk()
root.withdraw()

# Ask user to select a directory containing TIFF images
directory = askdirectory(title='Select folder with TIFF images')
if not directory:
    raise SystemExit('No directory selected')

# Helper function to interactively crop an image
crop_coords = []
def onselect(eclick, erelease):
    """Callback to capture crop rectangle coordinates"""
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    crop_coords[:] = [min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2)]
    plt.close()

# Loop over TIFF files
results = []
files = sorted(glob.glob(os.path.join(directory, '*.tiff')))
for filepath in files:
    img = np.array(Image.open(filepath))

    # Display and crop until a valid selection is made
    crop_coords.clear()
    while len(crop_coords) < 4:
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(f'Crop image: {os.path.basename(filepath)}')
        selector = RectangleSelector(
            ax, onselect,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )
        plt.show()
        if len(crop_coords) < 4:
            print("No selection detected. Please select a crop region.")

    # Extract crop region
    y1, y2, x1, x2 = crop_coords
    cut = img[y1:y2, x1:x2]

    # Plot original vs cropped
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img); axes[0].set_title('Original image')
    axes[1].imshow(cut); axes[1].set_title('Cropped image')
    plt.tight_layout(); plt.show()

    # Separate channels
    cut_R, cut_G, cut_B = cut[...,0], cut[...,1], cut[...,2]
    from matplotlib.colors import ListedColormap
    redmap = ListedColormap(np.stack([(255-np.arange(255))/255, np.zeros(255), np.zeros(255)], axis=1))
    greenmap = ListedColormap(np.stack([np.zeros(255), (255-np.arange(255))/255, np.zeros(255)], axis=1))
    bluemap = ListedColormap(np.stack([np.zeros(255), np.zeros(255), (255-np.arange(255))/255], axis=1))

    # Plot heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(cut_R, cmap=redmap); axes[0].set_title('Optic Density - RED')
    axes[1].imshow(cut_G, cmap=greenmap); axes[1].set_title('Optic Density - GREEN')
    axes[2].imshow(cut_B, cmap=bluemap); axes[2].set_title('Optic Density - BLUE')
    plt.tight_layout(); plt.show()

    # Prompt for delivered dose
    dose = float(input('Dose delivered (Gy): '))
    results.append([dose, cut_R.mean(), cut_R.std(), cut_G.mean(), cut_G.std(), cut_B.mean(), cut_B.std()])

# Convert to NumPy array
results = np.array(results)

def model(x, a, b, c):
    return a + b / (x - c)

doses = results[:,0]
pixels_R, pixels_G, pixels_B = results[:,1], results[:,3], results[:,5]

# Initial guesses
def make_initial(pixels):
    return [np.median(doses), (doses.max()-doses.min())/(pixels.max()-pixels.min()), pixels.min()-1]

p0_R, p0_G, p0_B = make_initial(pixels_R), make_initial(pixels_G), make_initial(pixels_B)

# Robust fitting helper
from scipy.optimize import OptimizeWarning
import warnings

def do_fit(x, y, p0):
    mfes = [20000, 50000, 100000]
    for mf in mfes:
        try:
            popt, pcov = curve_fit(model, x, y, p0=p0, maxfev=mf)
            return popt, pcov
        except RuntimeError:
            print(f"curve_fit failed with maxfev={mf}; retrying...")
    raise RuntimeError("Calibration fit failed after increasing maxfev")

# Fit channels
popt_R, pcov_R = do_fit(pixels_R, doses, p0_R)
popt_G, pcov_G = do_fit(pixels_G, doses, p0_G)
popt_B, pcov_B = do_fit(pixels_B, doses, p0_B)

# Compute 95% CIs
from scipy.stats import t
tval = t.ppf(0.975, len(doses)-3)
ci_R = tval * np.sqrt(np.diag(pcov_R))
ci_G = tval * np.sqrt(np.diag(pcov_G))
ci_B = tval * np.sqrt(np.diag(pcov_B))

# Plot fits and residuals
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for pix, popt, title, ax1, ax2 in [
    (pixels_R, popt_R, 'Red', axes[0,0], axes[1,0]),
    (pixels_G, popt_G, 'Green', axes[0,1], axes[1,1]),
    (pixels_B, popt_B, 'Blue', axes[0,2], axes[1,2])
]:
    ax1.scatter(pix, doses, label='Data')
    xs = np.linspace(pix.min(), pix.max(), 100)
    ax1.plot(xs, model(xs, *popt), label='Fit')
    ax1.set(title=title, xlabel='Pixel value', ylabel='Dose (Gy)')
    ax1.legend()

    resid = doses - model(pix, *popt)
    ax2.scatter(pix, resid)
    ax2.hlines(0, pix.min(), pix.max(), linestyles='--')
    ax2.set(title=f'{title} Residuals', xlabel='Pixel value', ylabel='Residuals')

plt.tight_layout(); plt.show()

# Save calibration parameters
params = np.vstack((popt_R, popt_G, popt_B)).T
# Guardamos sin encabezado, en notación científica con 7 decimales,
# separando columnas por dos espacios
np.savetxt('CalibParameters.txt',
           params,
           fmt='% .7e',
           delimiter='  ')
# Save standard deviations, pixel values, and 95% CIs,
# in the same order as the parameters
# Guardamos con encabezado, en notación científica con 7 decimales,
stds = np.vstack((results[:,2], results[:,4], results[:,6])).T
# separando columnas por dos espacios
# Guardamos con encabezado, en notación científica con 7 decimales,
np.savetxt('DoseStd.txt',
           stds,
           fmt='% .7e',
           delimiter='  ')

print('Calibration parameters saved to CalibParameters.txt')