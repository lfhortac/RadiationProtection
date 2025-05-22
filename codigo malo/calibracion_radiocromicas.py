import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import cv2
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.widgets import RectangleSelector

class RadiochromicCalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Calibración de Radiocromicas")
        self.root.geometry("1200x800")
        
        # Variables para almacenar datos
        self.images = []
        self.cropped_images = []
        self.doses = []
        self.results = []
        self.file_paths = []
        self.current_image_index = 0
        self.current_image = None
        self.current_crop = None
        self.rect_selector = None
        
        # Crear el marco principal
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        title_label = tk.Label(main_frame, text="Calibración de Radiocromicas", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)
        
        # Marco para botones
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        # Botones
        self.load_button = tk.Button(button_frame, text="Seleccionar Imágenes", command=self.load_radiochromics, 
                                    font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
        self.load_button.grid(row=0, column=0, padx=10)
        
        self.save_button = tk.Button(button_frame, text="Guardar Calibración", command=self.save_calibration,
                                    font=("Arial", 12), bg="#FF9800", fg="white", padx=10, pady=5, state=tk.DISABLED)
        self.save_button.grid(row=0, column=1, padx=10)
        
        self.exit_button = tk.Button(button_frame, text="Salir", command=root.destroy,
                                    font=("Arial", 12), bg="#F44336", fg="white", padx=10, pady=5)
        self.exit_button.grid(row=0, column=2, padx=10)
        
        # Marco para la imagen
        self.image_frame = tk.Frame(main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Crear figura para la imagen
        self.image_fig = Figure(figsize=(10, 6))
        self.image_canvas = FigureCanvasTkAgg(self.image_fig, master=self.image_frame)
        self.image_canvas_widget = self.image_canvas.get_tk_widget()
        self.image_canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Marco para gráficos de calibración
        self.calib_frame = tk.Frame(main_frame)
        self.calib_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Crear figura para gráficos de calibración
        self.calib_fig = Figure(figsize=(12, 8))
        self.calib_canvas = FigureCanvasTkAgg(self.calib_fig, master=self.calib_frame)
        self.calib_canvas_widget = self.calib_canvas.get_tk_widget()
        self.calib_canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Inicializar mapas de colores personalizados
        self.redmap = self.create_custom_colormap('red')
        self.greenmap = self.create_custom_colormap('green')
        self.bluemap = self.create_custom_colormap('blue')
        
        # Parámetros de calibración
        self.calib_params = None
    
    def create_custom_colormap(self, color_name):
        """Crear mapas de colores personalizados similares a los del código MATLAB"""
        if color_name == 'red':
            colors = [(1, 0, 0), (0, 0, 0)]  # Rojo a negro
        elif color_name == 'green':
            colors = [(0, 1, 0), (0, 0, 0)]  # Verde a negro
        elif color_name == 'blue':
            colors = [(0, 0, 1), (0, 0, 0)]  # Azul a negro
        return mcolors.LinearSegmentedColormap.from_list(color_name, colors)
    
    def load_radiochromics(self):
        """Seleccionar manualmente los archivos de radiocromicas"""
        files = filedialog.askopenfilenames(
            title="Seleccionar imágenes de radiocromicas",
            filetypes=[("Archivos TIFF", "*.tiff *.tif")]
        )
        if not files:
            return
        
        # Limpiar datos anteriores
        self.file_paths = list(files)
        self.images.clear()
        self.cropped_images.clear()
        self.doses.clear()
        self.results.clear()
        self.current_image_index = 0
        
        # Procesar la primera imagen
        self.process_next_image()
    
    def process_next_image(self):
        """Procesar la siguiente imagen en la lista"""
        if self.current_image_index >= len(self.file_paths):
            # Todas las imágenes han sido procesadas
            self.perform_calibration()
            return
        
        file_path = self.file_paths[self.current_image_index]
        filename = os.path.basename(file_path)
        
        try:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {filename}")
                self.current_image_index += 1
                return self.process_next_image()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.current_image = img_rgb
            self.display_image_for_cropping(img_rgb, filename)
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar {filename}: {str(e)}")
            self.current_image_index += 1
            self.process_next_image()
    
    def display_image_for_cropping(self, img, filename):
        """Mostrar la imagen y permitir al usuario recortar una región"""
        # Limpiar la figura
        self.image_fig.clear()
        
        # Configurar el subplot
        ax = self.image_fig.add_subplot(111)
        ax.imshow(img)
        ax.set_title(f"Seleccione área para analizar: {filename}")
        
        # Crear selector de rectángulo
        self.rect_selector = RectangleSelector(
            ax, self.on_select_rectangle,
            useblit=True,
            button=[1],  # Solo botón izquierdo
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        
        self.image_canvas.draw()
    
    def on_select_rectangle(self, eclick, erelease):
        """Callback para cuando el usuario selecciona un rectángulo"""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Asegurar que las coordenadas estén dentro de los límites
        height, width = self.current_image.shape[:2]
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        # Asegurar que x1 < x2 y y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Recortar la imagen
        cropped = self.current_image[y1:y2, x1:x2]
        self.current_crop = cropped
        
        # Mostrar la imagen original y la recortada
        self.display_original_and_cropped(self.current_image, cropped)
    
    def display_original_and_cropped(self, original, cropped):
        """Mostrar la imagen original y la recortada"""
        # Limpiar la figura
        self.image_fig.clear()
        
        # Configurar los subplots
        ax1 = self.image_fig.add_subplot(121)
        ax1.imshow(original)
        ax1.set_title('Imagen original')
        
        ax2 = self.image_fig.add_subplot(122)
        ax2.imshow(cropped)
        ax2.set_title('Imagen analizada')
        
        self.image_fig.tight_layout()
        self.image_canvas.draw()
        
        # Mostrar los mapas de calor RGB
        self.display_rgb_heatmaps(cropped)
    
    def display_rgb_heatmaps(self, img):
        """Mostrar mapas de calor para los canales RGB"""
        # Crear una nueva figura para los mapas de calor
        heatmap_fig = plt.figure(figsize=(15, 5))
        
        # Canal R
        ax1 = heatmap_fig.add_subplot(131)
        sns.heatmap(img[:,:,0], ax=ax1, cmap=self.redmap)
        ax1.set_title('Densidad Óptica - ROJO')
        
        # Canal G
        ax2 = heatmap_fig.add_subplot(132)
        sns.heatmap(img[:,:,1], ax=ax2, cmap=self.greenmap)
        ax2.set_title('Densidad Óptica - VERDE')
        
        # Canal B
        ax3 = heatmap_fig.add_subplot(133)
        sns.heatmap(img[:,:,2], ax=ax3, cmap=self.bluemap)
        ax3.set_title('Densidad Óptica - AZUL')
        
        plt.tight_layout()
        plt.show()
        
        # Solicitar la dosis
        self.ask_for_dose()
    
    def ask_for_dose(self):
        """Solicitar al usuario la dosis administrada"""
        dose = simpledialog.askfloat("Dosis", "Dosis administrada:", parent=self.root)
        
        if dose is not None:
            # Calcular estadísticas de la imagen recortada
            r_mean = np.mean(self.current_crop[:,:,0])
            r_std = np.std(self.current_crop[:,:,0])
            g_mean = np.mean(self.current_crop[:,:,1])
            g_std = np.std(self.current_crop[:,:,1])
            b_mean = np.mean(self.current_crop[:,:,2])
            b_std = np.std(self.current_crop[:,:,2])
            
            # Almacenar resultados
            result_row = [dose, r_mean, r_std, g_mean, g_std, b_mean, b_std]
            self.results.append(result_row)
            
            # Almacenar la imagen y la región recortada
            self.images.append(self.current_image)
            self.cropped_images.append(self.current_crop)
            self.doses.append(dose)
            
            # Pasar a la siguiente imagen
            self.current_image_index += 1
            self.process_next_image()
        else:
            # El usuario canceló, preguntar si desea omitir esta imagen
            if messagebox.askyesno("Omitir imagen", "¿Desea omitir esta imagen?"):
                self.current_image_index += 1
                self.process_next_image()
    
    def perform_calibration(self):
        """Realizar la calibración basada en los valores RGB y dosis"""
        if not self.results:
            messagebox.showinfo("Información", "No hay datos para realizar la calibración.")
            return

        # Convertir resultados a array numpy
        results_array = np.array(self.results)

        # Extraer datos
        doses    = results_array[:, 0]
        r_means  = results_array[:, 1]
        g_means  = results_array[:, 3]
        b_means  = results_array[:, 5]

        # Chequear que hay al menos 3 puntos (necesarios para ajustar 3 parámetros)
        if len(doses) < 3:
            messagebox.showerror("Error", "Se necesitan al menos 3 puntos de datos para ajustar la calibración.")
            return

        # Función de ajuste: a + b/(x-c)
        def fit_func(x, a, b, c):
            return a + b/(x - c)

        def fit_parameters(x_data, y_data, channel_name):
            """Ajusta fit_func(x; a,b,c) a y_data versus x_data, con
            iniciales y bounds para evitar x-c=0."""
            try:
                # Parámetros iniciales:
                a0 = np.min(y_data)                    # nivel_base
                b0 = np.ptp(y_data)                    # amplitud aprox.
                c0 = np.min(x_data) - 1e-3             # ligeramente por debajo del mínimo x
                p0 = [a0, b0, c0]

                # Bounds: c siempre < min(x_data) para que x-c≠0 en los datos
                lower = [-np.inf, -np.inf, -np.inf]
                upper = [ np.inf,  np.inf, np.min(x_data) - 1e-6]

                popt, pcov = curve_fit(
                    fit_func,
                    x_data,
                    y_data,
                    p0=p0,
                    bounds=(lower, upper),
                    maxfev=10000
                )
                perr = np.sqrt(np.diag(pcov))
                return popt, perr

            except Exception as e:
                messagebox.showerror(
                    "Error de ajuste",
                    f"No se pudo ajustar el canal {channel_name}:\n{str(e)}"
                )
                return None, None

        # Ajustar cada canal
        r_params, r_err = fit_parameters(r_means, doses, "Rojo")
        g_params, g_err = fit_parameters(g_means, doses, "Verde")
        b_params, b_err = fit_parameters(b_means, doses, "Azul")

        # Si alguno falló, salimos
        if any(p is None for p in (r_params, g_params, b_params)):
                messagebox.showerror("Error", "No se pudieron ajustar los parámetros de calibración.")
                return  

        # Guardamos y mostramos resultados
        self.calib_params = np.column_stack((r_params, g_params, b_params))
        self.display_calibration_plots(
            r_means, g_means, b_means,
            doses,
            r_params, g_params, b_params
        )
        self.save_button.config(state=tk.NORMAL)

    
    def display_calibration_plots(self, r_data, g_data, b_data, y_data, r_params, g_params, b_params):
        """Mostrar los gráficos de calibración"""
        # Limpiar la figura
        self.calib_fig.clear()
        
        # Crear subplots
        axs = []
        for i in range(6):
            axs.append(self.calib_fig.add_subplot(2, 3, i+1))
        
        # Datos para las curvas de ajuste
        x_fit = np.linspace(min(min(r_data), min(g_data), min(b_data)) * 0.9,
                           max(max(r_data), max(g_data), max(b_data)) * 1.1, 1000)
        
        # Canal Rojo
        axs[0].scatter(r_data, y_data, color='red')
        if r_params is not None:
            y_fit = fit_func(x_fit, *r_params)
            axs[0].plot(x_fit, y_fit, 'k-')
            axs[0].set_title(f'Red: a={r_params[0]:.4f}, b={r_params[1]:.4f}, c={r_params[2]:.4f}')
        else:
            axs[0].set_title('Red')
        axs[0].set_xlabel('Pixel value')
        axs[0].set_ylabel('Dose (Gy)')
        
        # Residuos Rojo
        if r_params is not None:
            y_pred = fit_func(r_data, *r_params)
            residuals = y_data - y_pred
            axs[3].scatter(r_data, residuals, color='red')
            axs[3].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axs[3].set_xlabel('Densidad óptica')
            axs[3].set_ylabel('Residuals')
        
        # Canal Verde
        axs[1].scatter(g_data, y_data, color='green')
        if g_params is not None:
            y_fit = fit_func(x_fit, *g_params)
            axs[1].plot(x_fit, y_fit, 'k-')
            axs[1].set_title(f'Green: a={g_params[0]:.4f}, b={g_params[1]:.4f}, c={g_params[2]:.4f}')
        else:
            axs[1].set_title('Green')
        axs[1].set_xlabel('Pixel value')
        axs[1].set_ylabel('Dose (Gy)')
        
        # Residuos Verde
        if g_params is not None:
            y_pred = fit_func(g_data, *g_params)
            residuals = y_data - y_pred
            axs[4].scatter(g_data, residuals, color='green')
            axs[4].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axs[4].set_xlabel('Densidad óptica')
            axs[4].set_ylabel('Residuals')
        
        # Canal Azul
        axs[2].scatter(b_data, y_data, color='blue')
        if b_params is not None:
            y_fit = fit_func(x_fit, *b_params)
            axs[2].plot(x_fit, y_fit, 'k-')
            axs[2].set_title(f'Blue: a={b_params[0]:.4f}, b={b_params[1]:.4f}, c={b_params[2]:.4f}')
        else:
            axs[2].set_title('Blue')
        axs[2].set_xlabel('Pixel value')
        axs[2].set_ylabel('Dose (Gy)')
        
        # Residuos Azul
        if b_params is not None:
            y_pred = fit_func(b_data, *b_params)
            residuals = y_data - y_pred
            axs[5].scatter(b_data, residuals, color='blue')
            axs[5].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axs[5].set_xlabel('Densidad óptica')
            axs[5].set_ylabel('Residuals')
        
        self.calib_fig.tight_layout()
        self.calib_canvas.draw()
    
    def save_calibration(self):
        """Guardar los parámetros de calibración en un archivo"""
        if self.calib_params is None:
            messagebox.showinfo("Información", "No hay datos de calibración para guardar.")
            return
        
        try:
            # Guardar parámetros en un archivo de texto
            np.savetxt('CalibParametersmias.txt', self.calib_params)
            
            # Guardar gráficos
            self.calib_fig.savefig('CalibrationGraphs.png', dpi=300, bbox_inches='tight')
            
            messagebox.showinfo("Éxito", "Calibración guardada correctamente en 'CalibParametersmias.txt'\nGráficos guardados en 'CalibrationGraphs.png'")
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar la calibración: {str(e)}")

    


def fit_func(x, a, b, c):
    """Función de ajuste: a + b/(x-c)"""
    return a + b/(x-c)

if __name__ == "__main__":
    root = tk.Tk()
    app = RadiochromicCalibrationApp(root)
    root.mainloop()
