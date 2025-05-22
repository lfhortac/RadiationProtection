import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import cv2
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

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
        self.current_image_index = -1
        self.current_image = None
        self.current_crop = None
        
        # Variables para el área de selección fija
        self.selection_x = tk.IntVar(value=100)
        self.selection_y = tk.IntVar(value=100)
        self.selection_width = tk.IntVar(value=200)
        self.selection_height = tk.IntVar(value=200)
        
        # Crear el marco principal
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        title_label = tk.Label(self.main_frame, text="Calibración de Radiocromicas", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)
        
        # Marco para botones
        button_frame = tk.Frame(self.main_frame)
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
        
        # Crear el marco de contenido principal (dividido en izquierda y derecha)
        content_frame = tk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Marco izquierdo para la imagen
        self.left_frame = tk.Frame(content_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Marco para la imagen
        self.image_frame = tk.Frame(self.left_frame, bd=2, relief=tk.GROOVE)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Etiqueta para mostrar la imagen
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Botón para procesar la imagen actual
        self.process_button = tk.Button(self.left_frame, text="Procesar Imagen", command=self.process_current_image,
                                      font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5, state=tk.DISABLED)
        self.process_button.pack(pady=10, fill=tk.X, padx=10)
        
        # Marco derecho para controles
        self.right_frame = tk.Frame(content_frame, width=300, bd=2, relief=tk.GROOVE)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.right_frame.pack_propagate(False)  # Evitar que el frame se redimensione
        
        # Título del panel de control
        control_title = tk.Label(self.right_frame, text="Panel de Control", font=("Arial", 14, "bold"))
        control_title.pack(pady=10)
        
        # Marco para información de la imagen actual
        self.image_info_frame = tk.Frame(self.right_frame)
        self.image_info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.image_info_label = tk.Label(self.image_info_frame, text="No hay imagen seleccionada", font=("Arial", 10))
        self.image_info_label.pack(anchor=tk.W)
        
        # Marco para entrada de dosis
        dose_frame = tk.Frame(self.right_frame)
        dose_frame.pack(fill=tk.X, padx=10, pady=5)
        
        dose_label = tk.Label(dose_frame, text="Dosis (Gy):", font=("Arial", 10))
        dose_label.pack(anchor=tk.W)
        
        self.dose_entry = tk.Entry(dose_frame)
        self.dose_entry.pack(fill=tk.X, pady=5)
        self.dose_entry.config(state=tk.DISABLED)
        
        # Separador
        ttk.Separator(self.right_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)
        
        # Controles para el área de selección
        selection_frame = tk.Frame(self.right_frame)
        selection_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(selection_frame, text="Área de Selección", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=5)
        
        # Control de posición X
        x_frame = tk.Frame(selection_frame)
        x_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(x_frame, text="Posición X:").pack(side=tk.LEFT)
        x_scale = tk.Scale(x_frame, from_=0, to=2000, orient=tk.HORIZONTAL, 
                          variable=self.selection_x, command=self.update_selection)
        x_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Control de posición Y
        y_frame = tk.Frame(selection_frame)
        y_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(y_frame, text="Posición Y:").pack(side=tk.LEFT)
        y_scale = tk.Scale(y_frame, from_=0, to=3000, orient=tk.HORIZONTAL, 
                          variable=self.selection_y, command=self.update_selection)
        y_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Control de ancho
        width_frame = tk.Frame(selection_frame)
        width_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(width_frame, text="Ancho:").pack(side=tk.LEFT)
        width_scale = tk.Scale(width_frame, from_=50, to=500, orient=tk.HORIZONTAL, 
                              variable=self.selection_width, command=self.update_selection)
        width_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Control de alto
        height_frame = tk.Frame(selection_frame)
        height_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(height_frame, text="Alto:").pack(side=tk.LEFT)
        height_scale = tk.Scale(height_frame, from_=50, to=500, orient=tk.HORIZONTAL, 
                               variable=self.selection_height, command=self.update_selection)
        height_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Botón para restablecer selección
        reset_button = tk.Button(selection_frame, text="Restablecer Selección", command=self.reset_selection)
        reset_button.pack(fill=tk.X, pady=10)
        
        # Separador
        ttk.Separator(self.right_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)
        
        # Tabla de datos de calibración
        table_frame = tk.Frame(self.right_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Label(table_frame, text="Datos de Calibración", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=5)
        
        # Crear tabla con Treeview
        self.data_table = ttk.Treeview(table_frame, columns=("file", "dose"), show="headings", height=10)
        self.data_table.heading("file", text="Archivo")
        self.data_table.heading("dose", text="Dosis (Gy)")
        self.data_table.column("file", width=150)
        self.data_table.column("dose", width=100)
        self.data_table.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar para la tabla
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.data_table.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_table.configure(yscrollcommand=scrollbar.set)
        
        # Parámetros de calibración
        self.calib_params = None
        
        # Ventana de resultados
        self.results_window = None
    
    def load_radiochromics(self):
        """Seleccionar manualmente los archivos de radiocromicas"""
        files = filedialog.askopenfilenames(
            title="Seleccionar imágenes de radiocromicas",
            filetypes=[("Archivos de imagen", "*.tiff *.tif *.jpg *.jpeg *.png")]
        )
        if not files:
            return
        
        # Limpiar datos anteriores
        self.file_paths = list(files)
        self.images.clear()
        self.cropped_images.clear()
        self.doses.clear()
        self.results.clear()
        self.current_image_index = -1
        
        # Limpiar tabla
        for item in self.data_table.get_children():
            self.data_table.delete(item)
        
        # Cargar la primera imagen
        self.load_next_image()
    
    def load_next_image(self):
        """Cargar la siguiente imagen en la lista"""
        self.current_image_index += 1
        
        if self.current_image_index >= len(self.file_paths):
            # Todas las imágenes han sido cargadas
            messagebox.showinfo("Información", "Todas las imágenes han sido cargadas. Procese cada imagen.")
            return
        
        file_path = self.file_paths[self.current_image_index]
        filename = os.path.basename(file_path)
        
        try:
            # Cargar imagen con OpenCV para mantener compatibilidad con el código original
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {filename}")
                self.load_next_image()
                return
                
            # Convertir de BGR a RGB para mostrar correctamente
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.current_image = img_rgb
            
            # Actualizar información de la imagen
            self.image_info_label.config(text=f"Imagen {self.current_image_index + 1}/{len(self.file_paths)}\n{filename}")
            
            # Habilitar entrada de dosis y botón de proceso
            self.dose_entry.config(state=tk.NORMAL)
            self.process_button.config(state=tk.NORMAL)
            
            # Añadir a la tabla
            self.data_table.insert("", tk.END, values=(filename, ""))
            
            # Mostrar la imagen
            self.display_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar {filename}: {str(e)}")
            self.load_next_image()
    
    def display_image(self):
        """Mostrar la imagen actual con el área de selección"""
        if self.current_image is None:
            return
            
        # Crear una copia de la imagen para dibujar el rectángulo
        img_with_rect = self.current_image.copy()
        
        # Obtener coordenadas del área de selección
        x = self.selection_x.get()
        y = self.selection_y.get()
        width = self.selection_width.get()
        height = self.selection_height.get()
        
        # Dibujar rectángulo rojo
        cv2.rectangle(img_with_rect, (x, y), (x + width, y + height), (255, 0, 0), 2)
        
        # Convertir a formato PIL para mostrar en tkinter
        pil_img = Image.fromarray(img_with_rect)
        
        # Redimensionar si es necesario para ajustar a la ventana
        img_width, img_height = pil_img.size
        max_width = self.image_frame.winfo_width() - 20
        max_height = self.image_frame.winfo_height() - 20
        
        if max_width > 0 and max_height > 0:  # Evitar división por cero
            scale = min(max_width / img_width, max_height / img_height)
            if scale < 1:  # Solo redimensionar si la imagen es más grande que el marco
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convertir a formato tkinter
        tk_img = ImageTk.PhotoImage(pil_img)
        
        # Actualizar la etiqueta de imagen
        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img  # Mantener referencia
    
    def update_selection(self, *args):
        """Actualizar el área de selección cuando cambian los controles"""
        self.display_image()
    
    def reset_selection(self):
        """Restablecer el área de selección a los valores predeterminados"""
        self.selection_x.set(100)
        self.selection_y.set(100)
        self.selection_width.set(200)
        self.selection_height.set(200)
        self.display_image()
    
    def process_current_image(self):
        """Procesar la imagen actual"""
        if self.current_image is None:
            return
            
        # Obtener la dosis ingresada
        try:
            dose_str = self.dose_entry.get().strip()
            if not dose_str:
                messagebox.showwarning("Advertencia", "Por favor, ingrese la dosis para esta imagen.")
                return
                
            dose = float(dose_str)
        except ValueError:
            messagebox.showwarning("Advertencia", "Por favor, ingrese un valor numérico válido para la dosis.")
            return
        
        # Obtener coordenadas del área de selección
        x = self.selection_x.get()
        y = self.selection_y.get()
        width = self.selection_width.get()
        height = self.selection_height.get()
        
        # Verificar que el área de selección esté dentro de los límites de la imagen
        img_height, img_width = self.current_image.shape[:2]
        if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
            messagebox.showwarning("Advertencia", "El área de selección está fuera de los límites de la imagen.")
            return
        
        # Recortar la imagen
        cropped = self.current_image[y:y+height, x:x+width]
        
        # Calcular estadísticas de la imagen recortada
        r_mean = np.mean(cropped[:,:,0])
        r_std = np.std(cropped[:,:,0])
        g_mean = np.mean(cropped[:,:,1])
        g_std = np.std(cropped[:,:,1])
        b_mean = np.mean(cropped[:,:,2])
        b_std = np.std(cropped[:,:,2])
        
        # Almacenar resultados
        result_row = [dose, r_mean, r_std, g_mean, g_std, b_mean, b_std]
        self.results.append(result_row)
        
        # Almacenar la imagen y la región recortada
        self.images.append(self.current_image)
        self.cropped_images.append(cropped)
        self.doses.append(dose)
        
        # Actualizar la tabla
        filename = os.path.basename(self.file_paths[self.current_image_index])
        item_id = self.data_table.get_children()[self.current_image_index]
        self.data_table.item(item_id, values=(filename, dose))
        
        # Limpiar entrada de dosis
        self.dose_entry.delete(0, tk.END)
        
        # Cargar la siguiente imagen o finalizar
        if self.current_image_index < len(self.file_paths) - 1:
            self.load_next_image()
        else:
            # Todas las imágenes han sido procesadas
            self.perform_calibration()
    
    def perform_calibration(self):
        """Realizar la calibración basada en los valores RGB y dosis"""
        if not self.results:
            messagebox.showinfo("Información", "No hay datos para realizar la calibración.")
            return

        R = np.array(self.results)
        doses   = R[:,0]
        r_vals  = R[:,1]
        g_vals  = R[:,3]
        b_vals  = R[:,5]

        if len(doses) < 3:
            messagebox.showerror("Error", "Se necesitan al menos 3 puntos de datos.")
            return

        # Escalamos dosis y valores a rango [0,1]
        d_min, d_max = doses.min(), doses.max()
        doses_s = (doses - d_min) / (d_max - d_min)
        def scale_back(y_s): 
            return y_s * (d_max - d_min) + d_min

        for name, vals in [("Rojo", r_vals), ("Verde", g_vals), ("Azul", b_vals)]:
            if np.any(vals <= 0):
                messagebox.showerror("Error", f"Valores de {name} han de ser positivos para el ajuste.")
                return

        def fit_func(x, a, b, c):
            return a + b / (x - c)

        def fit_channel(x, y, channel):
            # iniciales más razonables
            a0 = y.min()
            b0 = y.max() - y.min()
            c0 = x.min() - 1e-3
            p0 = [a0, b0, c0]

            # bounds: c < min(x)
            lower = [-np.inf, -np.inf, -np.inf]
            upper = [ np.inf,  np.inf, x.min() - 1e-6]

            popt, pcov = curve_fit(
                fit_func,
                x, y,
                p0=p0,
                bounds=(lower, upper),
                method='trf',
                maxfev=20000,
                ftol=1e-8,
                xtol=1e-8
            )
            perr = np.sqrt(np.diag(pcov))
            return popt, perr

        # Ajustamos en dosis escaladas vs. valores
        r_p, r_e = fit_channel(r_vals,   doses_s, "Rojo")
        g_p, g_e = fit_channel(g_vals,   doses_s, "Verde")
        b_p, b_e = fit_channel(b_vals,   doses_s, "Azul")

        # Convertimos parámetros de vuelta a escala real
        # Dado fit: dosis_s = a + b/(x-c)
        # => dosis = doses_s*(d_max-d_min)+d_min
        # Trabajamos solo y mostramos gráficas, así que suele bastar con guardar popt directamente

        self.calib_params = np.vstack([r_p, g_p, b_p]).T
        self.display_calibration_plots(r_vals, g_vals, b_vals,
                                    scale_back(doses_s),
                                    r_p, g_p, b_p)
        self.save_button.config(state=tk.NORMAL)


    
    def show_calibration_results(self, r_data, g_data, b_data, y_data, r_params, g_params, b_params):
        """Mostrar los resultados de calibración en una ventana separada"""
        # Cerrar ventana anterior si existe
        if self.results_window is not None and self.results_window.winfo_exists():
            self.results_window.destroy()
        
        # Crear nueva ventana
        self.results_window = tk.Toplevel(self.root)
        self.results_window.title("Resultados de Calibración")
        self.results_window.geometry("1000x800")
        
        # Marco principal
        main_frame = tk.Frame(self.results_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        title_label = tk.Label(main_frame, text="Gráficos de Calibración", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Crear figura para los gráficos (2x2)
        fig = Figure(figsize=(10, 8))
        
        # Canal Rojo
        ax1 = fig.add_subplot(221)
        ax1.scatter(r_data, y_data, color='red')
        
        # Datos para la curva de ajuste
        x_fit = np.linspace(min(r_data) * 0.9, max(r_data) * 1.1, 1000)
        y_fit = fit_func(x_fit, *r_params)
        ax1.plot(x_fit, y_fit, 'k-')
        ax1.set_title(f'Canal Rojo: a={r_params[0]:.4f}, b={r_params[1]:.4f}, c={r_params[2]:.4f}')
        ax1.set_xlabel('Valor de píxel')
        ax1.set_ylabel('Dosis (Gy)')
        
        # Canal Verde
        ax2 = fig.add_subplot(222)
        ax2.scatter(g_data, y_data, color='green')
        y_fit = fit_func(x_fit, *g_params)
        ax2.plot(x_fit, y_fit, 'k-')
        ax2.set_title(f'Canal Verde: a={g_params[0]:.4f}, b={g_params[1]:.4f}, c={g_params[2]:.4f}')
        ax2.set_xlabel('Valor de píxel')
        ax2.set_ylabel('Dosis (Gy)')
        
        # Canal Azul
        ax3 = fig.add_subplot(223)
        ax3.scatter(b_data, y_data, color='blue')
        y_fit = fit_func(x_fit, *b_params)
        ax3.plot(x_fit, y_fit, 'k-')
        ax3.set_title(f'Canal Azul: a={b_params[0]:.4f}, b={b_params[1]:.4f}, c={b_params[2]:.4f}')
        ax3.set_xlabel('Valor de píxel')
        ax3.set_ylabel('Dosis (Gy)')
        
        # Gráfico combinado
        ax4 = fig.add_subplot(224)
        ax4.scatter(r_data, y_data, color='red', label='Rojo')
        ax4.scatter(g_data, y_data, color='green', label='Verde')
        ax4.scatter(b_data, y_data, color='blue', label='Azul')
        
        # Curvas de ajuste para cada canal
        ax4.plot(x_fit, fit_func(x_fit, *r_params), 'r-')
        ax4.plot(x_fit, fit_func(x_fit, *g_params), 'g-')
        ax4.plot(x_fit, fit_func(x_fit, *b_params), 'b-')
        
        ax4.set_title('Curvas de Calibración Combinadas')
        ax4.set_xlabel('Valor de píxel')
        ax4.set_ylabel('Dosis (Gy)')
        ax4.legend()
        
        fig.tight_layout()
        
        # Crear canvas para mostrar la figura
        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Marco para botones
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        # Botón para guardar gráficos
        save_graphs_button = tk.Button(button_frame, text="Guardar Gráficos", 
                                     command=lambda: self.save_graphs(fig),
                                     font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
        save_graphs_button.pack(side=tk.LEFT, padx=10)
        
        # Botón para guardar parámetros
        save_params_button = tk.Button(button_frame, text="Guardar Parámetros", 
                                     command=self.save_calibration,
                                     font=("Arial", 12), bg="#FF9800", fg="white", padx=10, pady=5)
        save_params_button.pack(side=tk.LEFT, padx=10)
        
        # Botón para cerrar
        close_button = tk.Button(button_frame, text="Cerrar", 
                               command=self.results_window.destroy,
                               font=("Arial", 12), bg="#F44336", fg="white", padx=10, pady=5)
        close_button.pack(side=tk.LEFT, padx=10)
    
    def save_graphs(self, fig):
        """Guardar los gráficos de calibración"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf")],
            title="Guardar gráficos de calibración"
        )
        
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Éxito", f"Gráficos guardados en {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar los gráficos: {str(e)}")
    
    def save_calibration(self):
        """Guardar los parámetros de calibración en un archivo"""
        if self.calib_params is None:
            messagebox.showinfo("Información", "No hay datos de calibración para guardar.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Archivo de texto", "*.txt")],
            title="Guardar parámetros de calibración"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("Parámetros de Calibración\n")
                    f.write("=======================\n\n")
                    
                    r_params = self.calib_params['red']
                    g_params = self.calib_params['green']
                    b_params = self.calib_params['blue']
                    
                    f.write(f"Canal Rojo: a={r_params[0]:.6f}, b={r_params[1]:.6f}, c={r_params[2]:.6f}\n")
                    f.write(f"Canal Verde: a={g_params[0]:.6f}, b={g_params[1]:.6f}, c={g_params[2]:.6f}\n")
                    f.write(f"Canal Azul: a={b_params[0]:.6f}, b={b_params[1]:.6f}, c={b_params[2]:.6f}\n")
                
                messagebox.showinfo("Éxito", f"Parámetros guardados en {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar los parámetros: {str(e)}")

def fit_func(x, a, b, c):
    """Función de ajuste: a + b/(x-c)"""
    return a + b/(x-c)

if __name__ == "__main__":
    root = tk.Tk()
    app = RadiochromicCalibrationApp(root)
    root.mainloop()