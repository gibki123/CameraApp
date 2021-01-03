from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
# import darknet_video as dv
import numpy as np
import cv2

import os
import sys
import time


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

class MainWindow(QWidget):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.title = "Yolo_Detector_V_0.1"
        self.left = 0
        self.top = 0
        self.camera_width = 640
        self.camera_height = 480
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):

        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.camera_width, self.camera_height)
        self.resize(1024, 768)

        mainLayout = QVBoxLayout()
        hLayout1 = QHBoxLayout()

        self.label = QLabel(self)
        self.label.move(0, 0)
        self.label.resize(640, 480)
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.show()

        hLayout1.addWidget(self.label)

        vLayout1_1 = QVBoxLayout()

        button1 = QPushButton("Wybierz cfg")
        button2 = QPushButton("Wybierz data")
        button3 = QPushButton("Wybierz wagi")
        button4 = QPushButton("Zacznij detekcje")
        vLayout1_1.addWidget(button1)
        vLayout1_1.addWidget(button2)
        vLayout1_1.addWidget(button3)
        vLayout1_1.addWidget(button4)

        hLayout1.addLayout(vLayout1_1)

        hLayout2 = QHBoxLayout()

        self.rb1 = QRadioButton("Backbone_1")
        self.rb1.setChecked(True)
        self.rb1.backbone = "Backbone_1"
        self.rb1.toggled.connect(self.onClicked)
        hLayout2.addWidget(self.rb1)

        self.rb2 = QRadioButton("Backbone_2")
        self.rb2.backbone = "Backbone_2"
        self.rb2.toggled.connect(self.onClicked)
        hLayout2.addWidget(self.rb2)

        self.rb3 = QRadioButton("Backbone_3")
        self.rb3.backbone = "Backbone_3"
        self.rb3.toggled.connect(self.onClicked)
        hLayout2.addWidget(self.rb3)

        mainLayout.addLayout(hLayout1)
        mainLayout.addLayout(hLayout2)
        self.setLayout(mainLayout)

    def onClicked(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            print(f"Backbone is {radioButton.backbone}")




if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setApplicationName("NSAViewer")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
