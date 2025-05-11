import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importar antes de usar 3D
import matplotlib.cm as cm
import customtkinter as ctk


# Parámetros de calibración
pars = np.loadtxt('CalibParameters.txt').reshape(3, 3)
redCali, greenCali, blueCali = pars[:, 0], pars[:, 1], pars[:, 2]

# Compatibilidad Pillow
try:
    RESAMPLING = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLING = Image.LANCZOS

class DoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dose Analysis Tool")
        self.detected_circles = []

        # --- Paleta de colores ---
        fondo = "#1E1E2F"
        texto = "#FFFFFF"
        boton_color = "#3E4A61"
        boton_activo = "#556178"
        entrada_fondo = "#2C2F48"

        # --- Frame principal con layout nuevo (controles a la izquierda) ---
        main_frame = tk.Frame(root, bg=fondo)
        main_frame.pack(fill="both", expand=True)

        # Frame de controles (ahora a la izquierda)
        self.info_frame = tk.Frame(main_frame, width=250, bg=fondo)
        self.info_frame.pack(side="left", fill="y", padx=10, pady=10)

        # --- Etiqueta título ---
        tk.Label(self.info_frame, text="📊 Dose Tool", font=("Segoe UI", 14, "bold"),
                bg=fondo, fg=texto).pack(pady=(0, 20))

         # Función para crear botones estilizados
        def styled_button(master, text, command):
            return tk.Button(master, text=text, command=command,
                            bg=boton_color, fg=texto, font=("Segoe UI", 10),
                            relief="flat", bd=0, padx=10, pady=6,
                            activebackground=boton_activo, activeforeground=texto)

        # Botones funcionales
        styled_button(self.info_frame, "Cargar Imagen", self.load_image).pack(pady=4, fill="x")
        styled_button(self.info_frame, "Detect Radiocromicas", self.detectar_areas_radiocromicas).pack(pady=4, fill="x")
        styled_button(self.info_frame, "Detectar círculos", self.detectar_circulos_y_calcular_dosis).pack(pady=4, fill="x")
        styled_button(self.info_frame, "Mapa 3D de dosis", self.generate_dose_map_3d).pack(pady=4, fill="x")


        # Selector de forma (círculo o rectángulo)
        shape_frame = tk.Frame(self.info_frame, bg=fondo)
        shape_frame.pack(pady=5, fill="x")
        
        tk.Label(shape_frame, text="Seleccionar forma:", bg=fondo, fg=texto).pack(anchor="w")
        
        self.shape_var = tk.StringVar(value="circle")
        
        shape_options = tk.Frame(shape_frame, bg=fondo)
        shape_options.pack(fill="x", pady=2)
        
        circle_rb = tk.Radiobutton(shape_options, text="Círculo", variable=self.shape_var, 
                                  value="circle", command=self.on_shape_change,
                                  bg=fondo, fg=texto, selectcolor=boton_color, 
                                  activebackground=fondo, activeforeground=texto)
        circle_rb.pack(side="left", padx=(0, 10))
        
        rect_rb = tk.Radiobutton(shape_options, text="Rectángulo", variable=self.shape_var, 
                                value="rectangle", command=self.on_shape_change,
                                bg=fondo, fg=texto, selectcolor=boton_color, 
                                activebackground=fondo, activeforeground=texto)
        rect_rb.pack(side="left")
        
        # Frame para dimensiones
        self.dimensions_frame = tk.Frame(self.info_frame, bg=fondo)
        self.dimensions_frame.pack(pady=5, fill="x")
        
        # Inicialmente mostramos el control de radio para círculo
        self.circle_controls = tk.Frame(self.dimensions_frame, bg=fondo)
        self.circle_controls.pack(fill="x")
        
        tk.Label(self.circle_controls, text="Radio (px):", bg=fondo, fg=texto).pack(side="left", padx=5)
        self.radius_entry = tk.Entry(self.circle_controls, width=5, bg=entrada_fondo, fg=texto, insertbackground=texto)
        self.radius_entry.pack(side="left", padx=5)
        self.radius_entry.insert(0, "25")






        # Entrada nombre
        tk.Label(self.info_frame, text="Nombre medición:", bg=fondo, fg=texto).pack()
        self.name_entry = tk.Entry(self.info_frame, width=20, bg=entrada_fondo, fg=texto, insertbackground=texto)
        self.name_entry.pack(pady=(0, 15))
        styled_button(self.info_frame, "Guardar medición", self.save_measurement).pack(pady=4, fill="x")
        # Tamaño selección
        size_frame = tk.Frame(self.info_frame, bg=fondo)
        size_frame.pack(pady=5)

        tk.Label(size_frame, text="Ancho (px):", bg=fondo, fg=texto).grid(row=0, column=0, sticky="e", padx=5)
        self.width_entry = tk.Entry(size_frame, width=5, bg=entrada_fondo, fg=texto, insertbackground=texto)
        self.width_entry.grid(row=0, column=1, padx=5)
        self.width_entry.insert(0, "25")

        tk.Label(size_frame, text="Alto (px):", bg=fondo, fg=texto).grid(row=1, column=0, sticky="e", padx=5)
        self.height_entry = tk.Entry(size_frame, width=5, bg=entrada_fondo, fg=texto, insertbackground=texto)
        self.height_entry.grid(row=1, column=1, padx=5)
        self.height_entry.insert(0, "25")

       

        # Recuadro para mostrar resultados
        self.result_frame = tk.Frame(self.info_frame, bg="#2C2F48", bd=1, relief="solid")
        self.result_frame.pack(pady=20, fill="x", padx=5)

        self.dose_label = tk.Label(self.result_frame, text="Dosis: -", font=("Segoe UI", 11),
                                bg="#2C2F48", fg="white", anchor="w", justify="left")
        self.dose_label.pack(fill="x", padx=10, pady=(8, 0))

        self.std_label = tk.Label(self.result_frame, text="Desviación estándar: -", font=("Segoe UI", 10),
                                bg="#2C2F48", fg="white", anchor="w", justify="left")
        self.std_label.pack(fill="x", padx=10, pady=(0, 8))

        # --- Frame del Canvas (ahora a la derecha) ---
        self.canvas_frame = tk.Frame(main_frame)
        self.canvas_frame.pack(side="right", fill="both", expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.x_scroll = tk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.x_scroll.grid(row=1, column=0, sticky="ew")

        self.y_scroll = tk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.y_scroll.grid(row=0, column=1, sticky="ns")

        self.canvas.configure(xscrollcommand=self.x_scroll.set, yscrollcommand=self.y_scroll.set)

        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        # Canvas bindings
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<ButtonPress-3>", self.start_drag)
        self.canvas.bind("<B3-Motion>", self.do_drag)
        self.canvas.bind("<Configure>", lambda e: self.canvas.config(scrollregion=self.canvas.bbox("all")))

        # --- Variables internas ---
        self.rect_width = 50
        self.rect_height = 50
        self.pil_img = None
        self.tk_image = None
        self.last_x = None
        self.last_y = None
        self.last_avg_dose = None
        self.last_std_dose = None
        self.next_area_id = 1  # Para generar IDs únicos para radiocromicas
        self.radiochromic_dosis = []  # Lista para almacenar dosis radiocromicas detectadas


    def calcular_dosis_promedio(self, bloque_rgb):
        R, G, B = bloque_rgb[:, :, 0], bloque_rgb[:, :, 1], bloque_rgb[:, :, 2]
        try:
            doses = [
                redCali[0] + (redCali[1] / (np.mean(R[R > 0]) - redCali[2])),
                greenCali[0] + (greenCali[1] / (np.mean(G[G > 0]) - greenCali[2])),
                blueCali[0] + (blueCali[1] / (np.mean(B[B > 0]) - blueCali[2]))
            ]
            return np.mean(doses)
        except:
            return 0    

    def _on_mousewheel(self, event):
        if event.state & 0x0001:
            self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        else:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def start_drag(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def do_drag(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)


    def update_size(self):
        try:
            self.rect_width = int(self.width_entry.get())
            self.rect_height = int(self.height_entry.get())
            print(f"Nuevo tamaño actualizado: {self.rect_width} x {self.rect_height}")
        except ValueError:
            print("Error: valores inválidos.")



    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tiff")])
        if not self.image_path:
            print("No se seleccionó imagen.")
            return

        self.original_img = cv2.imread(self.image_path)
        if self.original_img is None:
            print("Error al cargar la imagen.")
            return

        if self.original_img.dtype != np.uint8:
            self.original_img = cv2.convertScaleAbs(self.original_img)

        self.rgb_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        self.pil_img = Image.fromarray(self.rgb_img)

        max_size = 800
        if self.pil_img.width > max_size or self.pil_img.height > max_size:
            self.pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(self.pil_img)

        # 💥 Aquí ajustamos el Canvas
        self.canvas.config(width=self.pil_img.width, height=self.pil_img.height)
        self.canvas.delete("all")
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Leave>", self.on_leave)
        # Limpiar estadísticas
        self.update_statistics_display()

    def update_statistics_display(self):
        """Actualiza el panel de estadísticas con los datos de todas las áreas"""
        self.stats_text.config(state="normal")
        self.stats_text.delete(1.0, tk.END)
        
        if not self.radiochromic_areas:
            self.stats_text.insert(tk.END, "No hay áreas radiocromicas definidas.")
            self.stats_text.config(state="disabled")
            return
            
        for area in self.radiochromic_areas:
            # Encabezado del área
            self.stats_text.insert(tk.END, f"=== {area['name']} ===\n", "header")
            
            if not area["circles"]:
                self.stats_text.insert(tk.END, "  No hay círculos en esta área.\n\n")
                continue
                
            # Estadísticas generales del área
            doses = [circle["dose"] for circle in area["circles"]]
            mean_dose = np.mean(doses)
            std_dose = np.std(doses, ddof=1) if len(doses) > 1 else 0
            
            self.stats_text.insert(tk.END, f"  Círculos: {len(area['circles'])}\n")
            self.stats_text.insert(tk.END, f"  Dosis promedio: {mean_dose:.4f} Gy\n")
            self.stats_text.insert(tk.END, f"  Desviación: {std_dose:.4f} Gy\n\n")
            
            # Detalles de cada círculo
            self.stats_text.insert(tk.END, "  Círculos individuales:\n")
            for i, circle in enumerate(area["circles"]):
                self.stats_text.insert(tk.END, f"  {i+1}. Dosis: {circle['dose']:.4f} Gy, ")
                self.stats_text.insert(tk.END, f"σ: {circle['std']:.4f} Gy\n")
                
            self.stats_text.insert(tk.END, "\n")
            
        self.stats_text.config(state="disabled")
        
        # Configurar etiquetas para el texto
        self.stats_text.tag_configure("header", font=("Segoe UI", 10, "bold"))    

    def detectar_areas_radiocromicas(self):
        if self.pil_img is None:
            print("⚠️ No hay imagen cargada.")
            return
            
        img_rgb = np.array(self.pil_img)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Aplicar umbral para detectar áreas oscuras (radiocromicas)
        _, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos pequeños
        min_area = 5000  # Ajustar según el tamaño de las áreas radiocromicas
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not valid_contours:
            print("⚠️ No se detectaron áreas radiocromicas.")
            return
            
        print(f"🔍 Se detectaron {len(valid_contours)} áreas radiocromicas.")
        
        # Limpiar áreas anteriores
        self.canvas.delete("radiochromic")
        self.radiochromic_areas = []
        
        # Procesar cada área
        for idx, contour in enumerate(valid_contours):
            # Obtener rectángulo que encierra el contorno
            x, y, w, h = cv2.boundingRect(contour)
            
            # Nombre por defecto
            area_name = f"Area_{idx+1}"
            
            # Crear área radiocromica
            area = {
                "name": area_name,
                "coords": (x, y, x+w, y+h),
                "circles": []  # Lista para almacenar círculos dentro del área
            }
            
            self.radiochromic_areas.append(area)
            
            # Dibujar rectángulo
            self.canvas.create_rectangle(
                x, y, x+w, y+h,
                outline='blue', width=2, tags="radiochromic"
            )
            
            # Añadir etiqueta con nombre
            self.canvas.create_text(
                x+5, y+5,
                text=area_name,
                anchor="nw",
                fill="white",
                tags="radiochromic"
            )
            
        # Mostrar diálogo para nombrar áreas
        self.mostrar_dialogo_nombres()
        
        print("✅ Detección de áreas radiocromicas completada.")

    def mostrar_dialogo_nombres(self):
        """Muestra un diálogo para nombrar las áreas radiocromicas"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Nombrar áreas radiocromicas")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Frame para la lista de áreas
        frame = tk.Frame(dialog)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        tk.Label(frame, text="Asignar nombres a las áreas:").pack(anchor="w")
        
        # Crear entradas para cada área
        entries = []
        for i, area in enumerate(self.radiochromic_areas):
            area_frame = tk.Frame(frame)
            area_frame.pack(fill="x", pady=5)
            
            tk.Label(area_frame, text=f"Área {i+1}:").pack(side="left")
            entry = tk.Entry(area_frame, width=30)
            entry.pack(side="left", padx=5)
            entry.insert(0, area["name"])
            entries.append((area, entry))
        
        # Botón para guardar
        def guardar_nombres():
            for area, entry in entries:
                nombre = entry.get().strip()
                if nombre:
                    area["name"] = nombre
                    
                    # Actualizar etiqueta en el canvas
                    x, y, _, _ = area["coords"]
                    for item in self.canvas.find_withtag("radiochromic"):
                        if self.canvas.type(item) == "text":
                            coords = self.canvas.coords(item)
                            if abs(coords[0] - x - 5) < 10 and abs(coords[1] - y - 5) < 10:
                                self.canvas.itemconfig(item, text=nombre)
                                break
            
            dialog.destroy()
        
        tk.Button(dialog, text="Guardar", command=guardar_nombres).pack(pady=10)


    def on_shape_change(self):
        """Maneja el cambio entre círculo y rectángulo"""
        self.current_shape = self.shape_var.get()
        
        # Ocultar y mostrar los controles apropiados
        if self.current_shape == "circle":
            self.rect_controls.pack_forget()
            self.circle_controls.pack(fill="x")
        else:
            self.circle_controls.pack_forget()
            self.rect_controls.pack(fill="x")    


    def on_click(self, event):
        if self.pil_img is None:
            return

        self.update_size()

        # 🛠️ Corregir coordenadas absolutas
        x_center = self.canvas.canvasx(event.x)
        y_center = self.canvas.canvasy(event.y)

        x1 = int(x_center - self.rect_width // 2)
        y1 = int(y_center - self.rect_height // 2)
        x2 = int(x_center + self.rect_width // 2)
        y2 = int(y_center + self.rect_height // 2)

        # Asegurar límites
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, self.pil_img.width)
        y2 = min(y2, self.pil_img.height)

        self.canvas.delete("rect")
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", tags="rect")

        cut = np.array(self.pil_img)[y1:y2, x1:x2]
        if cut.size == 0:
            self.dose_label.config(text="Dosis: región vacía")
            return

        R, G, B = cut[:, :, 0], cut[:, :, 1], cut[:, :, 2]
        try:
            dose = [
                redCali[0] + (redCali[1] / (np.mean(R) - redCali[2])),
                greenCali[0] + (greenCali[1] / (np.mean(G) - greenCali[2])),
                blueCali[0] + (blueCali[1] / (np.mean(B) - blueCali[2]))
            ]
            avg_dose = np.mean(dose)
            std_dose = np.std(dose,ddof=1)#/np.sqrt(len(dose))

            self.dose_label.config(text=f"Dosis: {avg_dose:.4f} Gy")
            self.std_label.config(text=f"Desviación estándar: {std_dose:.4f} Gy")

            # Guardar la medición temporal
            self.last_x = x_center
            self.last_y = y_center
            self.last_avg_dose = avg_dose
            self.last_std_dose = std_dose

        except Exception as e:
            self.dose_label.config(text="Error en cálculo")
            print("Error:", e)



    def save_measurement(self):
        # Si la imagen es muy grande, reducirla
        max_size = 1000  # o 800 si prefieres más chico
        if self.pil_img.width > max_size or self.pil_img.height > max_size:
            self.pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)


        if self.last_x is None or self.last_y is None:
            print("No hay medición lista para guardar.")
            return

        filename = "Mediciones.csv"
        file_exists = os.path.isfile(filename)

        # Leer el nombre escrito
        name = self.name_entry.get().strip()
        if not name:
            name = f"Medicion_{self.last_x}_{self.last_y}"

        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file, delimiter='\t')

            if not file_exists:
                writer.writerow(["Imagen", "Nombre_Medicion", "Dosis_Promedio", "Desviacion_Estandar"])

            writer.writerow([
                os.path.basename(self.image_path),
                name,
                f"{self.last_avg_dose:.4f}",
                f"{self.last_std_dose:.4f}"
            ])

        # Limpiar nombre luego de guardar
        self.name_entry.delete(0, tk.END)

        # Opcional: limpiar medición temporal
        self.last_x = None
        self.last_y = None
        self.last_avg_dose = None
        self.last_std_dose = None

        print(f"Medición '{name}' guardada exitosamente.")

    def generate_dose_map_3d(self):
        if self.pil_img is None:
            print("No hay imagen cargada.")
            return

        # Convertir imagen a array
        img_array = np.array(self.pil_img)

        # Definir resolución del grid (más alto = menos detalle, más rápido)
        step = 5  # píxeles por bloque
        h, w, _ = img_array.shape
        dose_map = []

        for y in range(0, h, step):
            row = []
            for x in range(0, w, step):
                block = img_array[y:y+step, x:x+step]
                if block.size == 0:
                    row.append(0)
                    continue
                avg_dose = self.calcular_dosis_promedio(block)
                row.append(avg_dose)
                dose_map.append(row)

        dose_map = np.array(dose_map)

        # Crear malla de coordenadas
        X = np.arange(0, dose_map.shape[1])
        Y = np.arange(0, dose_map.shape[0])
        X, Y = np.meshgrid(X, Y)

        valores_validos = dose_map[dose_map > 0]
        mean_dose = np.mean(valores_validos)
        std_dose = np.std(valores_validos, ddof=1)
       
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        zmin = mean_dose - 4 * std_dose
        zmax = mean_dose + 4 * std_dose
        ax.set_zlim(zmin, zmax)
        surf = ax.plot_surface(X, Y, dose_map, cmap=cm.viridis)
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Dosis estimada (Gy)')
        ax.set_title("Mapa 3D de dosis")
        ax.set_xlabel("X (bloques)")
        ax.set_ylabel("Y (bloques)")
        ax.set_zlabel("Dosis (Gy)")
        plt.tight_layout()
        plt.show()    

    def detectar_circulos_y_calcular_dosis(self):
        if self.pil_img is None:
            print("⚠️ No hay imagen cargada.")
            return

        img_rgb = np.array(self.pil_img)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 5)

        # Detectar círculos con Hough
        circles = cv2.HoughCircles(
            img_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )

        if circles is None:
            print("⚠️ No se detectaron círculos.")
            return

        circles = np.uint16(np.around(circles[0]))
        print(f"🔍 Se detectaron {len(circles)} círculos.")

        self.detected_circles = []  # Limpia anteriores
        resultados = []

        for idx, (x, y, r) in enumerate(circles):
            resultado = self.procesar_circulo(img_rgb, x, y, r)
            x, y, r_seguro = resultado["x"], resultado["y"], resultado["r"]
            print(f"⭕ Círculo {idx+1}: Dosis = {resultado['mean_dose']:.3f} Gy")

            resultados.append(resultado)
            self.detected_circles.append((x, y, r_seguro))  # para clic derecho

            # Dibujar el círculo detectado correctamente
            self.canvas.create_oval(
                x - r_seguro, y - r_seguro, x + r_seguro, y + r_seguro,
                outline='green', width=2, tags="circle_detect"
    )


        # Dibujar todos los círculos
        for resultado in resultados:
            x, y, r = resultado["x"], resultado["y"], resultado["r"]
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                outline='green', width=2, tags="circle_detect"
            )

        # Activar clic derecho sobre círculo
        self.canvas.bind("<Button-3>", self.on_circle_click)

        print("✅ Análisis de círculos completado.")
    

    def mapa_3d_circulo(self, x, y, r):
        if self.pil_img is None:
            print("⚠️ No hay imagen cargada.")
            return

        img_rgb = np.array(self.pil_img)
        self.procesar_circulo(img_rgb, x, y, r, graficar=True)  


    def on_circle_click(self, event):
        if not self.detected_circles or self.pil_img is None:
            return

        x_click = self.canvas.canvasx(event.x)
        y_click = self.canvas.canvasy(event.y)

        img_rgb = np.array(self.pil_img)

        for (x, y, r) in self.detected_circles:
            dist = np.sqrt((x - x_click)**2 + (y - y_click)**2)
            if dist < r:
                print(f"🖱️ Clic derecho sobre círculo en ({x}, {y}) → mostrando mapa 3D")
                self.procesar_circulo(img_rgb, x, y, r, graficar=True)
                return

    def procesar_circulo(self, img_rgb, x, y, r, step=1, factor_radio=0.9, graficar=False):
        h, w, _ = img_rgb.shape
        radio_seguro = int(r * factor_radio)

        # Máscara circular
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (x, y), radio_seguro, 255, -1)
        masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

        # Recorte
        x1 = max(0, x - radio_seguro)
        y1 = max(0, y - radio_seguro)
        x2 = min(w, x + radio_seguro)
        y2 = min(h, y + radio_seguro)
        cut = masked_img[y1:y2, x1:x2]

        # Dosis promedio exacta (por píxeles)
        #mean_dose_exact = self.calcular_dosis_promedio(cut)

        # Mapa por bloques
        dose_map = []
        for j in range(0, cut.shape[0], step):
            row = []
            for i in range(0, cut.shape[1], step):
                block = cut[j:j+step, i:i+step]
                if block.size == 0:
                    row.append(0)
                    continue
                row.append(self.calcular_dosis_promedio(block))
            dose_map.append(row)

        dose_map = np.array(dose_map)

        # Estadísticas de homogeneidad
        valores_validos = dose_map[dose_map > 0]
        mean_dose = np.mean(valores_validos)
        std_dose = np.std(valores_validos, ddof=1)#/np.sqrt(len(valores_validos))
        min_dose = np.min(valores_validos)
        max_dose = np.max(valores_validos)

        homogeneity_std = 100 * (1 - (std_dose / mean_dose))
        homogeneity_range = 100 * (1 - ((max_dose - min_dose) / mean_dose))

        if graficar:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.cm as cm

            X = np.arange(dose_map.shape[1])
            Y = np.arange(dose_map.shape[0])
            X, Y = np.meshgrid(X, Y)

            

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, dose_map, cmap=cm.viridis)
            zmin = mean_dose - 10 * std_dose
            zmax = mean_dose + 10 * std_dose
            ax.set_zlim(zmin, zmax)
            # Posición personalizada: [izquierda, abajo, ancho, alto]
            cbar_ax = fig.add_axes([0.87, 0.25, 0.02, 0.5])  # mueve la barra más a la derecha
           
            fig.colorbar(surf, cax=cbar_ax, label='Dosis (Gy)')
            ax.set_title("Mapa 3D de dosis en círculo")
            ax.set_xlabel("X (bloques)")
            ax.set_ylabel("Y (bloques)")
            ax.set_zlabel("Dosis (Gy)")

            info_text = (
                f"Promedio: {mean_dose:.3f} Gy\n"
                f"Desviación estándar: {std_dose:.3f} Gy\n"
                f"Homo. σ/μ: {homogeneity_std:.1f}%\n"
                f"Homo. rango: {homogeneity_range:.1f}%"
            )

            # Ajustar layout del gráfico para dejar espacio
            plt.subplots_adjust(right=0.8, bottom=0.2)

            # Mostrar texto en una posición fija debajo de la colorbar
            fig.text(0.1, 0.05, info_text,
                ha='left', va='bottom',
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85, boxstyle='round,pad=0.5'))
            plt.show()    

           

        return {
            "x": x,
            "y": y,
            "r": radio_seguro,
            "mean_dose": mean_dose,
            "std": std_dose,
            "min": min_dose,
            "max": max_dose,
            "homo_std": homogeneity_std,
            "homo_range": homogeneity_range
        }

        


if __name__ == "__main__":
    root = tk.Tk()
    app = DoseApp(root)
    root.mainloop()
