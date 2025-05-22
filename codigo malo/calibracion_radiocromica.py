import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, 
                            QListWidget, QListWidgetItem, QMessageBox, QGroupBox, 
                            QFormLayout, QSpinBox, QTableWidget, QTableWidgetItem, 
                            QHeaderView, QSplitter)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPalette, QIcon
from PyQt5.QtCore import Qt, QRect, QSize
from PIL import Image
from scipy.optimize import curve_fit

class CalibrationModel:
    def __init__(self):
        self.images = []
        self.doses = []
        self.red_values = []
        self.green_values = []
        self.blue_values = []
        self.red_std = []
        self.green_std = []
        self.blue_std = []
        self.red_params = None
        self.green_params = None
        self.blue_params = None
        self.red_ci = None
        self.green_ci = None
        self.blue_ci = None
    
    def add_image(self, image_path, dose, crop_area):
        """Añade una imagen al modelo de calibración"""
        try:
            # Abrir la imagen con PIL
            img = np.array(Image.open(image_path))
            
            # Extraer el área recortada
            x, y, width, height = crop_area
            cut = img[y:y+height, x:x+width]
            
            # Separar canales
            cut_R, cut_G, cut_B = cut[...,0], cut[...,1], cut[...,2]
            
            # Calcular medias y desviaciones estándar
            red_mean, red_std = cut_R.mean(), cut_R.std()
            green_mean, green_std = cut_G.mean(), cut_G.std()
            blue_mean, blue_std = cut_B.mean(), cut_B.std()
            
            # Guardar datos
            self.images.append({
                'path': image_path,
                'dose': dose,
                'crop_area': crop_area,
                'red_mean': red_mean,
                'red_std': red_std,
                'green_mean': green_mean,
                'green_std': green_std,
                'blue_mean': blue_mean,
                'blue_std': blue_std
            })
            
            # Actualizar listas para ajuste
            self.doses.append(dose)
            self.red_values.append(red_mean)
            self.green_values.append(green_mean)
            self.blue_values.append(blue_mean)
            self.red_std.append(red_std)
            self.green_std.append(green_std)
            self.blue_std.append(blue_std)
            
            return True
        except Exception as e:
            print(f"Error al procesar la imagen: {e}")
            return False
    
    def remove_image(self, index):
        """Elimina una imagen del modelo"""
        if 0 <= index < len(self.images):
            self.images.pop(index)
            self.doses.pop(index)
            self.red_values.pop(index)
            self.green_values.pop(index)
            self.blue_values.pop(index)
            self.red_std.pop(index)
            self.green_std.pop(index)
            self.blue_std.pop(index)
            return True
        return False
    
    def update_dose(self, index, dose):
        """Actualiza la dosis de una imagen"""
        if 0 <= index < len(self.images):
            self.images[index]['dose'] = dose
            self.doses[index] = dose
            return True
        return False
    
    def fit_calibration_curves(self):
        """Ajusta las curvas de calibración"""
        if len(self.doses) < 3:
            return False
        
        # Función modelo: Dosis = a + b / (pixel - c)
        def model(x, a, b, c):
            return a + b / (x - c)
        
        # Estimaciones iniciales
        def make_initial(pixels):
            return [np.median(self.doses), 
                   (max(self.doses)-min(self.doses))/(max(pixels)-min(pixels)), 
                   min(pixels)-1]
        
        p0_R = make_initial(self.red_values)
        p0_G = make_initial(self.green_values)
        p0_B = make_initial(self.blue_values)
        
        # Ajuste robusto
        def do_fit(x, y, p0):
            mfes = [20000, 50000, 100000]
            for mf in mfes:
                try:
                    popt, pcov = curve_fit(model, x, y, p0=p0, maxfev=mf)
                    return popt, pcov
                except RuntimeError:
                    print(f"curve_fit falló con maxfev={mf}; reintentando...")
            raise RuntimeError("El ajuste de calibración falló después de aumentar maxfev")
        
        try:
            # Ajustar canales
            self.red_params, pcov_R = do_fit(self.red_values, self.doses, p0_R)
            self.green_params, pcov_G = do_fit(self.green_values, self.doses, p0_G)
            self.blue_params, pcov_B = do_fit(self.blue_values, self.doses, p0_B)
            
            # Calcular intervalos de confianza del 95%
            from scipy.stats import t
            tval = t.ppf(0.975, len(self.doses)-3)
            self.red_ci = tval * np.sqrt(np.diag(pcov_R))
            self.green_ci = tval * np.sqrt(np.diag(pcov_G))
            self.blue_ci = tval * np.sqrt(np.diag(pcov_B))
            
            return True
        except Exception as e:
            print(f"Error en el ajuste: {e}")
            return False
    
    def save_parameters(self, filename="CalibParameters.txt"):
        """Guarda los parámetros de calibración en un archivo"""
        if self.red_params is None or self.green_params is None or self.blue_params is None:
            return False
        
        try:
            params = np.vstack((self.red_params, self.green_params, self.blue_params)).T
            np.savetxt(filename, params, fmt='% .7e', delimiter='  ')
            return True
        except Exception as e:
            print(f"Error al guardar parámetros: {e}")
            return False
    
    def save_std_dev(self, filename="DoseStd.txt"):
        """Guarda las desviaciones estándar en un archivo"""
        if not self.red_std or not self.green_std or not self.blue_std:
            return False
        
        try:
            stds = np.vstack((self.red_std, self.green_std, self.blue_std)).T
            np.savetxt(filename, stds, fmt='% .7e', delimiter='  ')
            return True
        except Exception as e:
            print(f"Error al guardar desviaciones estándar: {e}")
            return False

class ImageCanvas(FigureCanvas):
    """Canvas para mostrar y seleccionar áreas en imágenes"""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(ImageCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        self.image = None
        self.image_array = None
        self.crop_rect = None
        self.crop_start = None
        self.crop_size = (100, 100)
        self.is_dragging = False
        
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)
    
    def load_image(self, image_path):
        """Carga una imagen en el canvas"""
        try:
            self.image_array = np.array(Image.open(image_path))
            self.image = self.axes.imshow(self.image_array)
            self.crop_rect = None
            self.fig.canvas.draw()
            return True
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            return False
    
    def set_crop_size(self, width, height):
        """Establece el tamaño del área de recorte"""
        self.crop_size = (width, height)
        if self.crop_rect:
            self.update_crop_rect(self.crop_rect.get_x(), self.crop_rect.get_y())
    
    def on_press(self, event):
        """Maneja el evento de presionar el botón del ratón"""
        if event.inaxes != self.axes or self.image is None:
            return
        
        self.is_dragging = True
        self.crop_start = (event.xdata, event.ydata)
        
        if self.crop_rect:
            self.crop_rect.remove()
        
        self.crop_rect = self.axes.add_patch(
            plt.Rectangle((event.xdata, event.ydata), 
                         self.crop_size[0], self.crop_size[1],
                         linewidth=2, edgecolor='r', facecolor='none')
        )
        self.fig.canvas.draw()
    
    def on_motion(self, event):
        """Maneja el evento de mover el ratón"""
        if not self.is_dragging or event.inaxes != self.axes or self.crop_rect is None:
            return
        
        self.update_crop_rect(event.xdata, event.ydata)
    
    def on_release(self, event):
        """Maneja el evento de soltar el botón del ratón"""
        self.is_dragging = False
    
    def update_crop_rect(self, x, y):
        """Actualiza la posición del rectángulo de recorte"""
        if self.image is None or self.crop_rect is None:
            return
        
        # Asegurar que el rectángulo esté dentro de los límites de la imagen
        height, width = self.image_array.shape[:2]
        
        x = max(0, min(width - self.crop_size[0], x))
        y = max(0, min(height - self.crop_size[1], y))
        
        self.crop_rect.set_xy((x, y))
        self.fig.canvas.draw()
    
    def get_crop_area(self):
        """Devuelve el área de recorte actual"""
        if self.crop_rect is None:
            return None
        
        x, y = self.crop_rect.get_xy()
        return (int(x), int(y), int(self.crop_size[0]), int(self.crop_size[1]))

class CalibrationCurvesCanvas(FigureCanvas):
    """Canvas para mostrar las curvas de calibración"""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(CalibrationCurvesCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # Configurar estilo oscuro
        self.fig.patch.set_facecolor('#2D2D30')
        self.axes.set_facecolor('#1E1E1E')
        self.axes.tick_params(colors='white')
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')
        self.axes.title.set_color('white')
        
        # Configurar ejes
        self.axes.set_xlabel('Valor de Píxel')
        self.axes.set_ylabel('Dosis (Gy)')
        self.axes.set_title('Curvas de Calibración')
        self.axes.grid(True, linestyle='--', alpha=0.7)
    
    def plot_calibration(self, model):
        """Dibuja las curvas de calibración"""
        if not model.red_params or not model.green_params or not model.blue_params:
            return
        
        self.axes.clear()
        
        # Configurar ejes
        self.axes.set_xlabel('Valor de Píxel')
        self.axes.set_ylabel('Dosis (Gy)')
        self.axes.set_title('Curvas de Calibración')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        
        # Función modelo
        def model_func(x, a, b, c):
            return a + b / (x - c)
        
        # Dibujar puntos de datos
        self.axes.scatter(model.red_values, model.doses, color='red', label='Rojo', alpha=0.7)
        self.axes.scatter(model.green_values, model.doses, color='green', label='Verde', alpha=0.7)
        self.axes.scatter(model.blue_values, model.doses, color='blue', label='Azul', alpha=0.7)
        
        # Generar puntos para las curvas
        all_values = model.red_values + model.green_values + model.blue_values
        x_min, x_max = min(all_values), max(all_values)
        x_range = np.linspace(x_min, x_max, 100)
        
        # Dibujar curvas ajustadas
        try:
            y_red = [model_func(x, *model.red_params) for x in x_range 
                    if abs(x - model.red_params[2]) > 1]
            x_red = [x for x in x_range if abs(x - model.red_params[2]) > 1]
            self.axes.plot(x_red, y_red, 'r-', label='Ajuste Rojo')
            
            y_green = [model_func(x, *model.green_params) for x in x_range 
                      if abs(x - model.green_params[2]) > 1]
            x_green = [x for x in x_range if abs(x - model.green_params[2]) > 1]
            self.axes.plot(x_green, y_green, 'g-', label='Ajuste Verde')
            
            y_blue = [model_func(x, *model.blue_params) for x in x_range 
                     if abs(x - model.blue_params[2]) > 1]
            x_blue = [x for x in x_range if abs(x - model.blue_params[2]) > 1]
            self.axes.plot(x_blue, y_blue, 'b-', label='Ajuste Azul')
        except Exception as e:
            print(f"Error al dibujar curvas: {e}")
        
        self.axes.legend()
        self.fig.tight_layout()
        self.draw()

class ResidualsCanvas(FigureCanvas):
    """Canvas para mostrar los residuos de calibración"""
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(ResidualsCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # Configurar estilo oscuro
        self.fig.patch.set_facecolor('#2D2D30')
        self.axes.set_facecolor('#1E1E1E')
        self.axes.tick_params(colors='white')
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')
        self.axes.title.set_color('white')
        
        # Configurar ejes
        self.axes.set_xlabel('Valor de Píxel')
        self.axes.set_ylabel('Residuos (Gy)')
        self.axes.set_title('Análisis de Residuos')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.axhline(y=0, color='gray', linestyle='--')
    
    def plot_residuals(self, model):
        """Dibuja los residuos de calibración"""
        if not model.red_params or not model.green_params or not model.blue_params:
            return
        
        self.axes.clear()
        
        # Configurar ejes
        self.axes.set_xlabel('Valor de Píxel')
        self.axes.set_ylabel('Residuos (Gy)')
        self.axes.set_title('Análisis de Residuos')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.axhline(y=0, color='gray', linestyle='--')
        
        # Función modelo
        def model_func(x, a, b, c):
            return a + b / (x - c)
        
        # Calcular residuos
        red_residuals = [model.doses[i] - model_func(x, *model.red_params) 
                         for i, x in enumerate(model.red_values)]
        green_residuals = [model.doses[i] - model_func(x, *model.green_params) 
                          for i, x in enumerate(model.green_values)]
        blue_residuals = [model.doses[i] - model_func(x, *model.blue_params) 
                         for i, x in enumerate(model.blue_values)]
        
        # Dibujar residuos
        self.axes.scatter(model.red_values, red_residuals, color='red', label='Rojo', alpha=0.7)
        self.axes.scatter(model.green_values, green_residuals, color='green', label='Verde', alpha=0.7)
        self.axes.scatter(model.blue_values, blue_residuals, color='blue', label='Azul', alpha=0.7)
        
        self.axes.legend()
        self.fig.tight_layout()
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.model = CalibrationModel()
        self.current_image_path = None
        
        self.init_ui()
        self.set_dark_theme()
    
    def set_dark_theme(self):
        """Configura el tema oscuro para la aplicación"""
        dark_palette = QPalette()
        
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        QApplication.setPalette(dark_palette)
        QApplication.setStyle("Fusion")
    
    def init_ui(self):
        """Inicializa la interfaz de usuario"""
        self.setWindowTitle("Calibración de Películas Radiocrómicas")
        self.setGeometry(100, 100, 1200, 800)
        
        # Widget central y layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Crear pestañas
        self.tabs = QTabWidget()
        self.home_tab = QWidget()
        self.load_tab = QWidget()
        self.visualize_tab = QWidget()
        
        self.tabs.addTab(self.home_tab, "Inicio")
        self.tabs.addTab(self.load_tab, "Cargar Imágenes")
        self.tabs.addTab(self.visualize_tab, "Visualizar Calibración")
        
        main_layout.addWidget(self.tabs)
        
        # Configurar pestañas
        self.setup_home_tab()
        self.setup_load_tab()
        self.setup_visualize_tab()
    
    def setup_home_tab(self):
        """Configura la pestaña de inicio"""
        layout = QVBoxLayout(self.home_tab)
        
        # Título
        title_label = QLabel("Calibración de Películas Radiocrómicas")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        
        subtitle_label = QLabel("Una interfaz moderna para calibración y análisis de películas radiocrómicas")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 16px; color: #AAAAAA; margin-bottom: 40px;")
        
        # Botones principales
        buttons_layout = QHBoxLayout()
        
        # Botón Cargar Imágenes
        load_group = QGroupBox("Cargar Imágenes")
        load_layout = QVBoxLayout(load_group)
        load_desc = QLabel("Selecciona y procesa imágenes de películas radiocrómicas")
        load_button = QPushButton("Cargar Imágenes")
        load_button.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        load_layout.addWidget(load_desc)
        load_layout.addWidget(load_button)
        
        # Botón Visualizar Calibración
        visualize_group = QGroupBox("Visualizar Calibración")
        visualize_layout = QVBoxLayout(visualize_group)
        visualize_desc = QLabel("Visualiza curvas de calibración y parámetros")
        visualize_button = QPushButton("Ver Calibración")
        visualize_button.clicked.connect(lambda: self.tabs.setCurrentIndex(2))
        visualize_layout.addWidget(visualize_desc)
        visualize_layout.addWidget(visualize_button)
        
        # Botón Salir
        exit_group = QGroupBox("Salir")
        exit_layout = QVBoxLayout(exit_group)
        exit_desc = QLabel("Cierra la aplicación")
        exit_button = QPushButton("Salir")
        exit_button.clicked.connect(self.close)
        exit_layout.addWidget(exit_desc)
        exit_layout.addWidget(exit_button)
        
        buttons_layout.addWidget(load_group)
        buttons_layout.addWidget(visualize_group)
        buttons_layout.addWidget(exit_group)
        
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addLayout(buttons_layout)
        layout.addStretch()
    
    def setup_load_tab(self):
        """Configura la pestaña de carga de imágenes"""
        layout = QHBoxLayout(self.load_tab)
        
        # Panel izquierdo (controles)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Grupo de carga de imágenes
        load_group = QGroupBox("Controles")
        load_layout = QVBoxLayout(load_group)
        
        load_button = QPushButton("Cargar Imágenes")
        load_button.clicked.connect(self.load_images)
        load_layout.addWidget(load_button)
        
        # Grupo de tamaño de selección
        size_group = QGroupBox("Tamaño de Selección")
        size_layout = QFormLayout(size_group)
        
        self.width_input = QSpinBox()
        self.width_input.setRange(10, 1000)
        self.width_input.setValue(100)
        self.width_input.valueChanged.connect(self.update_crop_size)
        
        self.height_input = QSpinBox()
        self.height_input.setRange(10, 1000)
        self.height_input.setValue(100)
        self.height_input.valueChanged.connect(self.update_crop_size)
        
        size_layout.addRow("Ancho (px):", self.width_input)
        size_layout.addRow("Alto (px):", self.height_input)
        
        # Grupo de dosis
        dose_group = QGroupBox("Dosis")
        dose_layout = QFormLayout(dose_group)
        
        self.dose_input = QLineEdit()
        self.dose_input.setPlaceholderText("Ingrese la dosis en Gy")
        
        dose_layout.addRow("Dosis (Gy):", self.dose_input)
        
        # Botón de procesamiento
        process_button = QPushButton("Procesar Imágenes")
        process_button.clicked.connect(self.process_images)
        
        # Añadir grupos al panel izquierdo
        left_layout.addWidget(load_group)
        left_layout.addWidget(size_group)
        left_layout.addWidget(dose_group)
        left_layout.addWidget(process_button)
        left_layout.addStretch()
        
        # Panel central (visualización de imagen)
        central_panel = QWidget()
        central_layout = QVBoxLayout(central_panel)
        
        self.image_canvas = ImageCanvas(central_panel, width=8, height=6)
        central_layout.addWidget(self.image_canvas)
        
        # Panel inferior (lista de imágenes)
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        
        list_label = QLabel("Lista de Imágenes")
        list_label.setStyleSheet("font-weight: bold;")
        
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.select_image_from_list)
        
        bottom_layout.addWidget(list_label)
        bottom_layout.addWidget(self.image_list)
        
        # Crear un splitter para dividir los paneles
        splitter_v = QSplitter(Qt.Vertical)
        splitter_v.addWidget(central_panel)
        splitter_v.addWidget(bottom_panel)
        splitter_v.setSizes([600, 200])
        
        splitter_h = QSplitter(Qt.Horizontal)
        splitter_h.addWidget(left_panel)
        splitter_h.addWidget(splitter_v)
        splitter_h.setSizes([300, 900])
        
        layout.addWidget(splitter_h)
    
    def setup_visualize_tab(self):
        """Configura la pestaña de visualización de calibración"""
        layout = QVBoxLayout(self.visualize_tab)
        
        # Botones superiores
        top_layout = QHBoxLayout()
        
        save_params_button = QPushButton("Guardar Parámetros")
        save_params_button.clicked.connect(self.save_parameters)
        
        save_std_button = QPushButton("Guardar Desviaciones")
        save_std_button.clicked.connect(self.save_std_dev)
        
        save_images_button = QPushButton("Guardar Imágenes")
        save_images_button.clicked.connect(self.save_calibration_images)
        
        top_layout.addWidget(save_params_button)
        top_layout.addWidget(save_std_button)
        top_layout.addWidget(save_images_button)
        top_layout.addStretch()
        
        # Pestañas de visualización
        vis_tabs = QTabWidget()
        
        # Pestaña de curvas
        curves_tab = QWidget()
        curves_layout = QVBoxLayout(curves_tab)
        
        self.calibration_canvas = CalibrationCurvesCanvas(curves_tab, width=8, height=6)
        self.residuals_canvas = ResidualsCanvas(curves_tab, width=8, height=4)
        
        curves_layout.addWidget(self.calibration_canvas)
        curves_layout.addWidget(self.residuals_canvas)
        
        # Pestaña de parámetros
        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)
        
        params_label = QLabel("Parámetros de Calibración")
        params_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        params_label.setAlignment(Qt.AlignCenter)
        
        model_desc = QLabel("Modelo: Dosis = a + b / (pixel - c)")
        model_desc.setAlignment(Qt.AlignCenter)
        
        self.params_table = QTableWidget(3, 5)
        self.params_table.setHorizontalHeaderLabels(["Canal", "a", "b", "c", "95% CI"])
        self.params_table.verticalHeader().setVisible(False)
        self.params_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Configurar filas de la tabla
        red_item = QTableWidgetItem("Rojo")
        red_item.setForeground(QColor(255, 0, 0))
        self.params_table.setItem(0, 0, red_item)
        
        green_item = QTableWidgetItem("Verde")
        green_item.setForeground(QColor(0, 255, 0))
        self.params_table.setItem(1, 0, green_item)
        
        blue_item = QTableWidgetItem("Azul")
        blue_item.setForeground(QColor(0, 0, 255))
        self.params_table.setItem(2, 0, blue_item)
        
        params_layout.addWidget(params_label)
        params_layout.addWidget(model_desc)
        params_layout.addWidget(self.params_table)
        
        # Pestaña de datos
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        
        data_label = QLabel("Datos de Medición")
        data_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        data_label.setAlignment(Qt.AlignCenter)
        
        self.data_table = QTableWidget(0, 7)
        self.data_table.setHorizontalHeaderLabels([
            "Dosis (Gy)", "Media Rojo", "Desv. Rojo", 
            "Media Verde", "Desv. Verde", 
            "Media Azul", "Desv. Azul"
        ])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        data_layout.addWidget(data_label)
        data_layout.addWidget(self.data_table)
        
        # Añadir pestañas
        vis_tabs.addTab(curves_tab, "Curvas de Calibración")
        vis_tabs.addTab(params_tab, "Parámetros")
        vis_tabs.addTab(data_tab, "Datos")
        
        layout.addLayout(top_layout)
        layout.addWidget(vis_tabs)
    
    def load_images(self):
        """Carga imágenes desde el sistema de archivos"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Imágenes (*.tiff *.tif *.png *.jpg *.jpeg)")
        
        if file_dialog.exec_():
            filenames = file_dialog.selectedFiles()
            
            for filename in filenames:
                # Añadir a la lista de imágenes
                item = QListWidgetItem(os.path.basename(filename))
                item.setData(Qt.UserRole, filename)
                self.image_list.addItem(item)
            
            # Seleccionar la primera imagen
            if self.image_list.count() > 0 and not self.current_image_path:
                self.image_list.setCurrentRow(0)
                self.select_image_from_list(self.image_list.item(0))
    
    def select_image_from_list(self, item):
        """Selecciona una imagen de la lista"""
        if item is None:
            return
        
        image_path = item.data(Qt.UserRole)
        self.current_image_path = image_path
        
        # Cargar imagen en el canvas
        self.image_canvas.load_image(image_path)
        
        # Buscar si ya existe dosis para esta imagen
        for img in self.model.images:
            if img['path'] == image_path:
                self.dose_input.setText(str(img['dose']))
                return
        
        # Si no existe, limpiar el campo de dosis
        self.dose_input.clear()
    
    def update_crop_size(self):
        """Actualiza el tamaño del área de recorte"""
        width = self.width_input.value()
        height = self.height_input.value()
        self.image_canvas.set_crop_size(width, height)
    
    def process_images(self):
        """Procesa las imágenes cargadas"""
        if self.image_list.count() == 0:
            QMessageBox.warning(self, "Advertencia", "No hay imágenes cargadas.")
            return
        
        # Verificar que todas las imágenes tengan dosis y área de recorte
        missing_data = False
        
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            image_path = item.data(Qt.UserRole)
            
            # Seleccionar la imagen para verificar
            self.image_list.setCurrentRow(i)
            self.select_image_from_list(item)
            
            # Verificar dosis
            try:
                dose = float(self.dose_input.text())
                if dose <= 0:
                    raise ValueError("La dosis debe ser mayor que cero")
            except ValueError:
                QMessageBox.warning(
                    self, 
                    "Advertencia", 
                    f"Falta la dosis para la imagen {os.path.basename(image_path)} o no es válida."
                )
                missing_data = True
                break
            
            # Verificar área de recorte
            if self.image_canvas.get_crop_area() is None:
                QMessageBox.warning(
                    self, 
                    "Advertencia", 
                    f"Seleccione un área de recorte para la imagen {os.path.basename(image_path)}."
                )
                missing_data = True
                break
            
            # Añadir o actualizar imagen en el modelo
            found = False
            for j, img in enumerate(self.model.images):
                if img['path'] == image_path:
                    self.model.update_dose(j, dose)
                    found = True
                    break
            
            if not found:
                self.model.add_image(
                    image_path, 
                    dose, 
                    self.image_canvas.get_crop_area()
                )
        
        if missing_data:
            return
        
        # Ajustar curvas de calibración
        if self.model.fit_calibration_curves():
            # Actualizar visualizaciones
            self.update_calibration_view()
            
            # Cambiar a la pestaña de visualización
            self.tabs.setCurrentIndex(2)
            
            QMessageBox.information(
                self, 
                "Éxito", 
                "Calibración completada con éxito."
            )
        else:
            QMessageBox.critical(
                self, 
                "Error", 
                "Error al ajustar las curvas de calibración. Asegúrese de tener al menos 3 imágenes con diferentes dosis."
            )
    
    def update_calibration_view(self):
        """Actualiza la vista de calibración"""
        # Actualizar gráficos
        self.calibration_canvas.plot_calibration(self.model)
        self.residuals_canvas.plot_residuals(self.model)
        
        # Actualizar tabla de parámetros
        if self.model.red_params and self.model.green_params and self.model.blue_params:
            # Parámetros rojos
            self.params_table.setItem(0, 1, QTableWidgetItem(f"{self.model.red_params[0]:.4f}"))
            self.params_table.setItem(0, 2, QTableWidgetItem(f"{self.model.red_params[1]:.4f}"))
            self.params_table.setItem(0, 3, QTableWidgetItem(f"{self.model.red_params[2]:.4f}"))
            self.params_table.setItem(0, 4, QTableWidgetItem(
                f"±{self.model.red_ci[0]:.4f}, ±{self.model.red_ci[1]:.4f}, ±{self.model.red_ci[2]:.4f}"
            ))
            
            # Parámetros verdes
            self.params_table.setItem(1, 1, QTableWidgetItem(f"{self.model.green_params[0]:.4f}"))
            self.params_table.setItem(1, 2, QTableWidgetItem(f"{self.model.green_params[1]:.4f}"))
            self.params_table.setItem(1, 3, QTableWidgetItem(f"{self.model.green_params[2]:.4f}"))
            self.params_table.setItem(1, 4, QTableWidgetItem(
                f"±{self.model.green_ci[0]:.4f}, ±{self.model.green_ci[1]:.4f}, ±{self.model.green_ci[2]:.4f}"
            ))
            
            # Parámetros azules
            self.params_table.setItem(2, 1, QTableWidgetItem(f"{self.model.blue_params[0]:.4f}"))
            self.params_table.setItem(2, 2, QTableWidgetItem(f"{self.model.blue_params[1]:.4f}"))
            self.params_table.setItem(2, 3, QTableWidgetItem(f"{self.model.blue_params[2]:.4f}"))
            self.params_table.setItem(2, 4, QTableWidgetItem(
                f"±{self.model.blue_ci[0]:.4f}, ±{self.model.blue_ci[1]:.4f}, ±{self.model.blue_ci[2]:.4f}"
            ))
        
        # Actualizar tabla de datos
        self.data_table.setRowCount(len(self.model.doses))
        
        for i in range(len(self.model.doses)):
            self.data_table.setItem(i, 0, QTableWidgetItem(f"{self.model.doses[i]:.2f}"))
            self.data_table.setItem(i, 1, QTableWidgetItem(f"{self.model.red_values[i]:.2f}"))
            self.data_table.setItem(i, 2, QTableWidgetItem(f"{self.model.red_std[i]:.2f}"))
            self.data_table.setItem(i, 3, QTableWidgetItem(f"{self.model.green_values[i]:.2f}"))
            self.data_table.setItem(i, 4, QTableWidgetItem(f"{self.model.green_std[i]:.2f}"))
            self.data_table.setItem(i, 5, QTableWidgetItem(f"{self.model.blue_values[i]:.2f}"))
            self.data_table.setItem(i, 6, QTableWidgetItem(f"{self.model.blue_std[i]:.2f}"))
    
    def save_parameters(self):
        """Guarda los parámetros de calibración"""
        if not self.model.red_params or not self.model.green_params or not self.model.blue_params:
            QMessageBox.warning(self, "Advertencia", "No hay parámetros de calibración para guardar.")
            return
        
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("Archivos de texto (*.txt)")
        file_dialog.setDefaultSuffix("txt")
        file_dialog.selectFile("CalibParameters.txt")
        
        if file_dialog.exec_():
            filename = file_dialog.selectedFiles()[0]
            if self.model.save_parameters(filename):
                QMessageBox.information(self, "Éxito", f"Parámetros guardados en {filename}")
            else:
                QMessageBox.critical(self, "Error", "Error al guardar los parámetros.")
    
    def save_std_dev(self):
        """Guarda las desviaciones estándar"""
        if not self.model.red_std or not self.model.green_std or not self.model.blue_std:
            QMessageBox.warning(self, "Advertencia", "No hay datos de desviación estándar para guardar.")
            return
        
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("Archivos de texto (*.txt)")
        file_dialog.setDefaultSuffix("txt")
        file_dialog.selectFile("DoseStd.txt")
        
        if file_dialog.exec_():
            filename = file_dialog.selectedFiles()[0]
            if self.model.save_std_dev(filename):
                QMessageBox.information(self, "Éxito", f"Desviaciones guardadas en {filename}")
            else:
                QMessageBox.critical(self, "Error", "Error al guardar las desviaciones.")
    
    def save_calibration_images(self):
        """Guarda las imágenes de calibración"""
        if not self.model.red_params or not self.model.green_params or not self.model.blue_params:
            QMessageBox.warning(self, "Advertencia", "No hay curvas de calibración para guardar.")
            return
        
        directory = QFileDialog.getExistingDirectory(self, "Seleccionar directorio para guardar imágenes")
        if not directory:
            return
        
        # Guardar imagen de curvas de calibración
        self.calibration_canvas.fig.savefig(
            os.path.join(directory, "calibration_curves.png"), 
            dpi=300, 
            bbox_inches='tight'
        )
        
        # Guardar imagen de residuos
        self.residuals_canvas.fig.savefig(
            os.path.join(directory, "residuals.png"), 
            dpi=300, 
            bbox_inches='tight'
        )
        
        QMessageBox.information(
            self, 
            "Éxito", 
            f"Imágenes guardadas en {directory}"
        )

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()