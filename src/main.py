# src/main.py
import sys
import os
import joblib
import pandas as pd
from PySide6 import QtWidgets
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QFormLayout, QDoubleSpinBox
)
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from db import DB

HERE = os.path.dirname(os.path.abspath(__file__))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Electric Motor Life Monitoring")
        self.setGeometry(150, 80, 1200, 720)

        self.resize(1200, 720)                # safer alternative
        self.showMaximized()                
        
        self.model = None
        self.db = DB(os.path.join(HERE, "predictions.db"))

        # Data lists for live plotting
        self.recent_vib = []
        self.recent_temp = []
        self.recent_curr = []

        self._build_ui()

        # Load default model if exists
        default_model = os.path.join(HERE, "..", "models", "motor_pipeline.pkl")
        if os.path.exists(default_model):
            try:
                self.model = joblib.load(default_model)
                self.status_label.setText(f"Model loaded: {default_model}")
            except Exception as e:
                self.status_label.setText(f"Model found but failed to load: {e}")
        else:
            self.status_label.setText("No model loaded. Click 'Load Model' to load pipeline .pkl")

        self.load_history_into_table()

        # Timer for live plotting
        self.timer = QTimer()
        self.timer.timeout.connect(self.live_plot_step)

    def _build_ui(self):
        # --- Left panel ---
        left = QVBoxLayout()
        form = QFormLayout()

        motor_types = [
            'ID Fan Motor',
            'Dust Collector Fan Motor',
            'Automatic Mould-Handling Motor',
            'Hybrid Moulding Machine Motor',
            'Hydraulic Gesa Moulding Motor'
        ]
        self.cb_motor = QComboBox(); self.cb_motor.addItems(motor_types)
        self.spin_temp = QDoubleSpinBox(); self.spin_temp.setRange(-50,200); self.spin_temp.setDecimals(2); self.spin_temp.setValue(60.0)
        self.spin_vib = QDoubleSpinBox(); self.spin_vib.setRange(0,200); self.spin_vib.setDecimals(2); self.spin_vib.setValue(3.0)
        self.spin_current = QDoubleSpinBox(); self.spin_current.setRange(0,1000); self.spin_current.setDecimals(2); self.spin_current.setValue(10.0)
        self.spin_speed = QDoubleSpinBox(); self.spin_speed.setRange(0,20000); self.spin_speed.setDecimals(1); self.spin_speed.setValue(1500.0)

        # Connect input changes to auto-predict
        self.cb_motor.currentTextChanged.connect(self.on_predict)
        self.spin_temp.valueChanged.connect(self.on_predict)
        self.spin_vib.valueChanged.connect(self.on_predict)
        self.spin_current.valueChanged.connect(self.on_predict)
        self.spin_speed.valueChanged.connect(self.on_predict)

        form.addRow("Motor Type:", self.cb_motor)
        form.addRow("Temperature (°C):", self.spin_temp)
        form.addRow("Vibration (mm/s):", self.spin_vib)
        form.addRow("Current (A):", self.spin_current)
        form.addRow("Speed (RPM):", self.spin_speed)

        left.addLayout(form)

        btn_predict = QPushButton("Predict")
        btn_predict.clicked.connect(self.on_predict)
        btn_load_model = QPushButton("Load Model")
        btn_load_model.clicked.connect(self.on_load_model)
        btn_load_csv = QPushButton("Load CSV (batch)")
        btn_load_csv.clicked.connect(self.on_load_csv)
        btn_export = QPushButton("Export History")
        btn_export.clicked.connect(self.on_export_history)
        btn_sim_start = QPushButton("Start Simulate Live")
        btn_sim_start.clicked.connect(self.toggle_simulation)

        left.addWidget(btn_predict)
        left.addWidget(btn_load_model)
        left.addWidget(btn_load_csv)
        left.addWidget(btn_export)
        left.addWidget(btn_sim_start)
        left.addStretch()

        left_widget = QWidget(); 
        left_widget.setLayout(left)
        
        left_widget.setFixedWidth(350)
    
        # --- Right panel ---
        right = QVBoxLayout()

        # Status label
        self.status_label = QLabel("Status")
        self.status_label.setAlignment(Qt.AlignCenter)   # Center text
        self.status_label.setWordWrap(True)              # Wrap if long
        self.status_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, 
            QtWidgets.QSizePolicy.Fixed
        )
        right.addWidget(self.status_label)

        # Result label (main message)
        self.result_label = QLabel("No prediction yet")
        self.result_label.setAlignment(Qt.AlignCenter)   # Center text horizontally
        self.result_label.setWordWrap(True)              # Wrap if text is long
        self.result_label.setFixedHeight(100)            # Enough height for multi-line
        self.result_label.setStyleSheet(
            "font-size:18pt; border-radius:8px; padding:10px;"
        )
        self.result_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, 
            QtWidgets.QSizePolicy.Fixed
        )
        right.addWidget(self.result_label)

        # History table
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ['Timestamp','Motor Type','Temp','Vib','Current','Speed','Pred','Prob']
        )
        self.table.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, 
            QtWidgets.QSizePolicy.Expanding
        )
        
        # ✅ Make columns auto-stretch to take full width
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        
        right.addWidget(self.table, 1)

        # Matplotlib canvas
        self.fig = Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Recent Vibration (mm/s)")
        self.ax.set_xlabel("Steps")
        self.ax.set_ylabel("Vibration")
        right.addWidget(self.canvas, 1)

        # Dropdown to select graph
        self.graph_selector = QComboBox()
        self.graph_selector.addItems(["Vibration", "Temperature", "Current"])
        self.graph_selector.currentTextChanged.connect(self.update_graph_visibility)
        right.addWidget(self.graph_selector, 0)

        # --- Main layout ---
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget, 1)  # left side fixed ratio
        main_layout.addLayout(right, 2)        # right side takes remaining space

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # ✅ Window sizing
        self.showMaximized()   # Start in maximized mode (recommended)
        # OR if you want it to open in available screen size but resizable:
        # self.resize(1200, 720)

    # ----------------- Live Plot Methods -----------------
    def toggle_simulation(self):
        if self.timer.isActive():
            self.timer.stop()
            self.status_label.setText("Stopped simulation")
        else:
            self.timer.start(1000)
            self.status_label.setText("Simulating live data (1s)")

    def live_plot_step(self):
        selected_graph = self.graph_selector.currentText()
        vib = float(self.spin_vib.value())
        temp = float(self.spin_temp.value())
        curr = float(self.spin_current.value())

        # Append to lists
        self.recent_vib.append(vib)
        self.recent_temp.append(temp)
        self.recent_curr.append(curr)

        # Keep last 60
        if len(self.recent_vib) > 60: self.recent_vib.pop(0)
        if len(self.recent_temp) > 60: self.recent_temp.pop(0)
        if len(self.recent_curr) > 60: self.recent_curr.pop(0)

        self.ax.clear()
        if selected_graph == "Vibration":
            self.ax.plot(self.recent_vib, marker='o', linestyle='-')
            self.ax.set_title("Recent Vibration (mm/s)")
            self.ax.set_ylabel("Vibration")
        elif selected_graph == "Temperature":
            self.ax.plot(self.recent_temp, marker='o', linestyle='-', color='orange')
            self.ax.set_title("Recent Temperature (°C)")
            self.ax.set_ylabel("Temperature")
        elif selected_graph == "Current":
            self.ax.plot(self.recent_curr, marker='o', linestyle='-', color='green')
            self.ax.set_title("Recent Current (A)")
            self.ax.set_ylabel("Current")
        self.ax.set_xlabel("Steps")
        self.canvas.draw()

    def update_graph_visibility(self, selected_graph):
        self.live_plot_step()  # redraw selected graph

    # ----------------- Existing methods unchanged -----------------
    def on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load model pipeline (.pkl)", "", "Pickle files (*.pkl *.joblib);;All files (*)")
        if path:
            try:
                self.model = joblib.load(path)
                self.status_label.setText(f"Loaded model: {path}")
                QMessageBox.information(self, "Model loaded", "Model pipeline loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Load error", f"Failed to load model: {e}")

    def on_predict(self):
        if self.model is None:
            QMessageBox.warning(self, "No model", "No model loaded.")
            return
        motor_type = self.cb_motor.currentText()
        temperature = float(self.spin_temp.value())
        vibration = float(self.spin_vib.value())
        current = float(self.spin_current.value())
        speed = float(self.spin_speed.value())

        X = pd.DataFrame([{
            'motor_type': motor_type,
            'temperature': temperature,
            'vibration': vibration,
            'current': current,
            'speed': speed
        }])

        try:
            pred = int(self.model.predict(X)[0])
            prob = float(self.model.predict_proba(X)[0][1]) if hasattr(self.model,'predict_proba') else None
        except Exception as e:
            QMessageBox.critical(self, "Prediction error", f"Model prediction failed: {e}")
            return

        # --- Determine which features triggered the maintenance warning ---
        maintenance_reasons = []
        if temperature > 80:  # example threshold, adjust based on your system
            maintenance_reasons.append(f"High Temperature ({temperature}°C)")
        if vibration > 10:  # example threshold
            maintenance_reasons.append(f"High Vibration ({vibration} mm/s)")
        if current > 50:  # example threshold
            maintenance_reasons.append(f"High Current ({current} A)")
        if speed > 5000:  # optional, if speed matters
            maintenance_reasons.append(f"High Speed ({speed} RPM)")

        reasons_text = ""
        if maintenance_reasons:
            reasons_text = " | Reason(s): " + ", ".join(maintenance_reasons)

        # Show result with colored style + reason
        if pred == 1:
            self.result_label.setText(f"⚠️ Maintenance required{reasons_text}")
            self.result_label.setStyleSheet(
                "background-color:#e74c3c; color:white; font-size:18pt; padding:10px; border-radius:8px;")
        else:
            self.result_label.setText("✅ Motor healthy")
            self.result_label.setStyleSheet(
                "background-color:#2ecc71; color:white; font-size:18pt; padding:10px; border-radius:8px;")

        # Insert into DB and table
        self.db.insert(motor_type, temperature, vibration, current, speed, pred, prob if prob else -1.0)
        self.append_row_to_table([
            pd.Timestamp.now().isoformat(), motor_type, temperature, vibration,
            current, speed, pred, round(prob,3) if prob else None
        ])

        # --- LIVE PLOT UPDATE ---
        self.recent_vib.append(vibration)
        self.recent_temp.append(temperature)
        self.recent_curr.append(current)
        if len(self.recent_vib) > 60: self.recent_vib.pop(0)
        if len(self.recent_temp) > 60: self.recent_temp.pop(0)
        if len(self.recent_curr) > 60: self.recent_curr.pop(0)
        self.live_plot_step()

    def on_load_csv(self):
        # unchanged; batch prediction
        pass

    def append_row_to_table(self, row_values):
        row_count = self.table.rowCount()
        self.table.insertRow(0)
        for col, val in enumerate(row_values):
            self.table.setItem(0, col, QTableWidgetItem(str(val)))

    def load_history_into_table(self):
        rows = self.db.fetch_all()
        for r in rows:
            self.append_row_to_table(list(r))

    def on_export_history(self):
        rows = self.db.fetch_all()
        if not rows: return
        df = pd.DataFrame(rows, columns=['timestamp','motor_type','temperature','vibration','current','speed','prediction','prob'])
        path, _ = QFileDialog.getSaveFileName(self, "Save history CSV", "history_export.csv", "CSV files (*.csv)")
        if path:
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Exported", f"History exported to {path}")


if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    # Load custom QSS styles
    with open(os.path.join(HERE, "..", "ui", "styles.qss")) as f:
        app.setStyleSheet(f.read())

    win = MainWindow()
    win.show()
    sys.exit(app.exec())
