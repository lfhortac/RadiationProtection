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
        
        # Crear el marco de contenido principal (izquierda y derecha)
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
        self.right_frame.pack_propagate(False)
        
        # Panel de control
        control_title = tk.Label(self.right_frame, text="Panel de Control", font=("Arial", 14, "bold"))
        control_title.pack(pady=10)
        
        self.image_info_label = tk.Label(self.right_frame, text="No hay imagen seleccionada", font=("Arial", 10))
        self.image_info_label.pack(anchor=tk.W, padx=10)
        
        # Entrada de dosis
        dose_frame = tk.Frame(self.right_frame)
        dose_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(dose_frame, text="Dosis (Gy):", font=("Arial",10)).pack(anchor=tk.W)
        self.dose_entry = tk.Entry(dose_frame, state=tk.DISABLED)
        self.dose_entry.pack(fill=tk.X, pady=5)
        
        ttk.Separator(self.right_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)
        
        # Controles de área de selección
        sel_frame = tk.Frame(self.right_frame)
        sel_frame.pack(fill=tk.X, padx=10)
        tk.Label(sel_frame, text="Área de Selección", font=("Arial",12,"bold")).pack(anchor=tk.W, pady=5)
        for label, var, frm, to in [("X", self.selection_x,0,2000), ("Y", self.selection_y,0,3000),
                                   ("Ancho", self.selection_width,50,500), ("Alto", self.selection_height,50,500)]:
            row = tk.Frame(sel_frame); row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=f"{label}:").pack(side=tk.LEFT)
            tk.Scale(row, from_=frm, to=to, orient=tk.HORIZONTAL, variable=var, command=lambda *a: self.display_image()).pack(side=tk.RIGHT, fill=tk.X, expand=True)
        tk.Button(sel_frame, text="Restablecer Selección", command=self.reset_selection).pack(fill=tk.X, pady=10)
        
        ttk.Separator(self.right_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)
        
        # Tabla de datos
        table_frame = tk.Frame(self.right_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        tk.Label(table_frame, text="Datos de Calibración", font=("Arial",12,"bold")).pack(anchor=tk.W)
        self.data_table = ttk.Treeview(table_frame, columns=("file","dose"), show="headings", height=8)
        self.data_table.heading("file", text="Archivo"); self.data_table.heading("dose", text="Dosis (Gy)")
        self.data_table.column("file", width=150); self.data_table.column("dose", width=100)
        self.data_table.pack(fill=tk.BOTH, expand=True)
        ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.data_table.yview).pack(side=tk.RIGHT, fill=tk.Y)
        self.data_table.configure(yscrollcommand=lambda f,l: None)
        
        self.calib_params = None
        self.results_window = None

    def load_radiochromics(self):
        files = filedialog.askopenfilenames(title="Seleccionar imágenes...", filetypes=[("Imagen", "*.tiff *.tif *.jpg *.png")])
        if not files: return
        self.file_paths = list(files); self.current_image_index = -1; self.results.clear()
        for item in self.data_table.get_children(): self.data_table.delete(item)
        self.load_next_image()

    def load_next_image(self):
        self.current_image_index += 1
        if self.current_image_index >= len(self.file_paths):
            messagebox.showinfo("Info","Todas las imágenes cargadas.")
            return
        path = self.file_paths[self.current_image_index]
        name = os.path.basename(path)
        img = self._imread_unicode(path)
        if img is None:
            messagebox.showerror("Error",f"No pudo cargar {name}"); self.load_next_image(); return
        self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image_info_label.config(text=f"{self.current_image_index+1}/{len(self.file_paths)}: {name}")
        self.dose_entry.config(state=tk.NORMAL); self.dose_entry.delete(0,tk.END)
        self.process_button.config(state=tk.NORMAL)
        self.data_table.insert("",tk.END,values=(name,"") )
        self.display_image()

    def _imread_unicode(self, path):
        try:
            with open(path,'rb') as f: data = f.read()
            arr = np.frombuffer(data, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        except: return None

    def display_image(self):
        if self.current_image is None: return
        img = self.current_image.copy(); h,w = img.shape[:2]
        x,y = self.selection_x.get(), self.selection_y.get()
        ww,hh = self.selection_width.get(), self.selection_height.get()
        cv2.rectangle(img,(x,y),(x+ww,y+hh),(255,0,0),2)
        pil = Image.fromarray(img)
        mw,mh = self.image_frame.winfo_width()-20, self.image_frame.winfo_height()-20
        if mw>0 and mh>0:
            sc = min(mw/pil.width, mh/pil.height)
            if sc<1: pil = pil.resize((int(pil.width*sc),int(pil.height*sc)),Image.LANCZOS)
        tkimg = ImageTk.PhotoImage(pil)
        self.image_label.config(image=tkimg); self.image_label.image=tkimg

    def reset_selection(self):
        self.selection_x.set(100); self.selection_y.set(100)
        self.selection_width.set(200); self.selection_height.set(200)
        self.display_image()

    def process_current_image(self):
        if self.current_image is None: return
        try:
            dose=float(self.dose_entry.get())
        except:
            messagebox.showwarning("Aviso","Ingrese dosis válida"); return
        x,y=self.selection_x.get(),self.selection_y.get()
        ww,hh=self.selection_width.get(),self.selection_height.get()
        h,w=self.current_image.shape[:2]
        if x<0 or y<0 or x+ww> w or y+hh>h:
            messagebox.showwarning("Aviso","Selección fuera de imagen"); return
        crop=self.current_image[y:y+hh,x:x+ww]
        r,mr,gr = np.mean(crop[:,:,0]),np.std(crop[:,:,0]),None
        g,mg,gg = np.mean(crop[:,:,1]),np.std(crop[:,:,1]),None
        b,mb,gb = np.mean(crop[:,:,2]),np.std(crop[:,:,2]),None
        self.results.append([dose,r,mr,g,mg,b,mb])
        self.images.append(self.current_image); self.cropped_images.append(crop)
        iid=self.data_table.get_children()[self.current_image_index]
        self.data_table.item(iid,values=(os.path.basename(self.file_paths[self.current_image_index]),dose))
        if self.current_image_index< len(self.file_paths)-1:
            self.load_next_image()
        else:
            self.perform_calibration()

    def perform_calibration(self):
        if not self.results:
            messagebox.showinfo("Info","No hay datos para calibrar"); return
        R=np.array(self.results); doses, rv, gv, bv=R[:,0],R[:,1],R[:,3],R[:,5]
        if len(doses)<3:
            messagebox.showerror("Error","Se necesitan ≥3 puntos"); return
        # escalar
        dmin,dmax=doses.min(),doses.max()
        ds=(doses-dmin)/(dmax-dmin)
        def fitf(x,a,b,c):return a+b/(x-c)
        def fitch(x,y):
            p0=[y.min(), np.ptp(y), x.min()-1e-3]
            lb=[-np.inf,-np.inf,-np.inf]; ub=[np.inf,np.inf,x.min()-1e-6]
            popt,pcov=curve_fit(fitf,x,y,p0=p0,bounds=(lb,ub),method='trf',maxfev=20000,ftol=1e-8,xtol=1e-8)
            return popt
        rp=fitch(rv,ds); gp=fitch(gv,ds); bp=fitch(bv,ds)
        self.calib_params=np.vstack([rp,gp,bp]).T
        self.show_calibration_results(rv,gv,bv,(ds*(dmax-dmin)+dmin),rp,gp,bp)
        self.save_button.config(state=tk.NORMAL)

    def show_calibration_results(self, r_data, g_data, b_data, y_data, r_params, g_params, b_params):
        if self.results_window and self.results_window.winfo_exists(): self.results_window.destroy()
        self.results_window=tk.Toplevel(self.root); self.results_window.title("Resultados")
        fig=Figure(figsize=(10,8))
        xfit=np.linspace(min(r_data.min(),g_data.min(),b_data.min())*0.9,
                         max(r_data.max(),g_data.max(),b_data.max())*1.1,500)
        for idx,(data,p,col,name) in enumerate([(r_data,r_params,'red','Rojo'),(g_data,g_params,'green','Verde'),(b_data,b_params,'blue','Azul')]):
            ax=fig.add_subplot(2,3,idx+1); ax.scatter(data,y_data,color=col)
            ax.plot(xfit,fitf(xfit,*p),'k-'); ax.set_title(f"{name}: a={p[0]:.3f},b={p[1]:.3f},c={p[2]:.3f}")
            ax.set_xlabel('Pixel'); ax.set_ylabel('Dosis')
            res=y_data-fitf(data,*p); axr=fig.add_subplot(2,3,idx+4)
            axr.scatter(data,res,color=col); axr.axhline(0,linestyle='--'); axr.set_xlabel('Pixel'); axr.set_ylabel('Residuo')
        ax4=fig.add_subplot(2,3,3)
        ax4.scatter(r_data,y_data,color='red',label='Rojo'); ax4.scatter(g_data,y_data,color='green',label='Verde')
        ax4.scatter(b_data,y_data,color='blue',label='Azul')
        ax4.plot(xfit,fitf(xfit,*r_params),'r-'); ax4.plot(xfit,fitf(xfit,*g_params),'g-'); ax4.plot(xfit,fitf(xfit,*b_params),'b-')
        ax4.set_title('Combinado'); ax4.set_xlabel('Pixel'); ax4.set_ylabel('Dosis'); ax4.legend()
        fig.tight_layout()
        canvas=FigureCanvasTkAgg(fig,master=self.results_window); canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        btnf=tk.Frame(self.results_window); btnf.pack(pady=10)
        tk.Button(btnf,text="Guardar Gráficos",command=lambda:self.save_graphs(fig),bg='#4CAF50',fg='white').pack(side=tk.LEFT,padx=5)
        tk.Button(btnf,text="Guardar Parámetros",command=self.save_calibration,bg='#FF9800',fg='white').pack(side=tk.LEFT,padx=5)
        tk.Button(btnf,text="Cerrar",command=self.results_window.destroy,bg='#F44336',fg='white').pack(side=tk.LEFT,padx=5)

    def save_graphs(self, fig):
        path=filedialog.asksaveasfilename(defaultextension='.png',filetypes=[('PNG','*.png')],title='Guardar gráficos')
        if path:
            fig.savefig(path,dpi=300,bbox_inches='tight'); messagebox.showinfo('Éxito',f'Guardado en {path}')

    def save_calibration(self):
        if self.calib_params is None: messagebox.showinfo('Info','Nada que guardar'); return
        path=filedialog.asksaveasfilename(defaultextension='.txt',filetypes=[('Txt','*.txt')],title='Guardar parámetros')
        if path:
            with open(path,'w') as f:
                f.write('Canal Rojo: a={:.6f}, b={:.6f}, c={:.6f}\n'.format(*self.calib_params[:,0]))
                f.write('Canal Verde: a={:.6f}, b={:.6f}, c={:.6f}\n'.format(*self.calib_params[:,1]))
                f.write('Canal Azul: a={:.6f}, b={:.6f}, c={:.6f}\n'.format(*self.calib_params[:,2]))
            messagebox.showinfo('Éxito',f'Parámetros guardados en {path}')

def fitf(x,a,b,c):
    return a+b/(x-c)

if __name__ == "__main__":
    root=tk.Tk()
    app=RadiochromicCalibrationApp(root)
    root.mainloop()
