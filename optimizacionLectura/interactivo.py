import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import csv
import matplotlib.pyplot as plt

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
        self.last_x = None
        self.last_y = None
        self.last_avg_dose = None
        self.last_std_dose = None
        self.root = root
        self.root.title("Dose Analysis Tool")

        self.canvas = tk.Canvas(root, cursor="cross", bg="black", width=800, height=800)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.info_frame = tk.Frame(root, width=250)
        self.info_frame.pack(side="right", fill="y", padx=10)

        self.dose_label = tk.Label(self.info_frame, text="Dosis: -", font=("Arial", 14))
        self.dose_label.pack(pady=10)

        size_frame = tk.Frame(self.info_frame)
        size_frame.pack(pady=10)

          # Campo para nombre de medición
        tk.Label(self.info_frame, text="Nombre medición:").pack(pady=(10, 0))
        self.name_entry = tk.Entry(self.info_frame, width=20)
        self.name_entry.pack(pady=5)

        tk.Label(size_frame, text="Ancho (px):").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.width_entry = tk.Entry(size_frame, width=5)
        self.width_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        self.width_entry.insert(0, "50")

        tk.Label(size_frame, text="Alto (px):").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.height_entry = tk.Entry(size_frame, width=5)
        self.height_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        self.height_entry.insert(0, "50")

        self.update_size_button = tk.Button(self.info_frame, text="Actualizar tamaño", command=self.update_size)
        self.update_size_button.pack(pady=5)

        self.load_button = tk.Button(self.info_frame, text="Cargar Imagen", command=self.load_image)
        self.load_button.pack(pady=5)

        self.save_button = tk.Button(self.info_frame, text="Guardar medición", command=self.save_measurement)
        self.save_button.pack(pady=10)


        self.rect_width = 100
        self.rect_height = 100

        self.pil_img = None
        self.tk_image = None

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


    def on_click(self, event):
        if self.pil_img is None:
            return

        self.update_size()

        x_center, y_center = event.x, event.y
        x1 = x_center - self.rect_width // 2
        y1 = y_center - self.rect_height // 2
        x2 = x_center + self.rect_width // 2
        y2 = y_center + self.rect_height // 2

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
            std_dose = np.std(dose)

            self.dose_label.config(text=f"Dosis: {avg_dose:.4f}\nDesviación: {std_dose:.4f}")

            # Guardar medición temporalmente
            self.last_x = x_center
            self.last_y = y_center
            self.last_avg_dose = avg_dose
            self.last_std_dose = std_dose

        except Exception as e:
            self.dose_label.config(text="Error en cálculo")
            print("Error:", e)


    def save_measurement(self):
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
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["Imagen", "Nombre_Medicion", "Centro_X", "Centro_Y", "Ancho", "Alto", "Dosis_Promedio", "Desviacion_Estandar"])

            writer.writerow([
                os.path.basename(self.image_path),
                name,
                self.last_x,
                self.last_y,
                self.rect_width,
                self.rect_height,
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

   
        


if __name__ == "__main__":
    root = tk.Tk()
    app = DoseApp(root)
    root.mainloop()
