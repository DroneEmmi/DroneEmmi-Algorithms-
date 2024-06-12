import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout, QTabWidget, \
    QProgressBar
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, Qt
from djitellopy import tello
import cv2
import time
import os
from ultralytics import YOLO


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.base_flag = 0
        self.initDrone()
        self.updateBatteryStatus()
        self.showPlaceholder()

    def initUI(self):
        self.setWindowTitle('Tello Drone Controller')
        self.setGeometry(100, 100, 800, 600)

        self.background_label = QLabel(self)
        self.background_pixmap = QPixmap('emmiWallpaper.jpeg')
        self.background_label.setPixmap(self.background_pixmap)
        self.background_label.setScaledContents(True)
        self.background_label.setGeometry(0, 0, 800, 600)

        self.tabs = QTabWidget(self)
        self.control_tab = QWidget()
        self.calibrate_tab = QWidget()

        self.tabs.addTab(self.control_tab, "Control")
        self.tabs.addTab(self.calibrate_tab, "Calibrate")

        self.initControlTab()
        self.initCalibrateTab()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

        self.tabs.hide()

    def showPlaceholder(self):
        self.placeholder_timer = QTimer(self)
        self.placeholder_timer.timeout.connect(self.showMainUI)
        self.placeholder_timer.start(5000)  # Show placeholder for 5 seconds

    def showMainUI(self):
        self.placeholder_timer.stop()
        self.background_label.hide()
        self.tabs.show()

    def initControlTab(self):
        layout = QVBoxLayout()

        self.button = QPushButton('Auto Land', self)
        self.button.setIcon(QIcon('land.png'))
        self.button.clicked.connect(self.set_base_flag)
        layout.addWidget(self.button)

        self.start_button = QPushButton('Start', self)
        self.start_button.setIcon(QIcon('start.png'))
        self.start_button.clicked.connect(self.start_main)
        layout.addWidget(self.start_button)

        self.status_label = QLabel('Status: Ready', self)
        layout.addWidget(self.status_label)

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.battery_bar = QProgressBar(self)
        self.battery_bar.setAlignment(Qt.AlignCenter)
        self.battery_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: green;
                width: 20px;
            }
        """)
        layout.addWidget(self.battery_bar)

        self.control_tab.setLayout(layout)

    def initCalibrateTab(self):
        layout = QVBoxLayout()

        self.calibrate_button = QPushButton('Calibrate', self)
        self.calibrate_button.setIcon(QIcon('path/to/calibrate_icon.png'))
        self.calibrate_button.clicked.connect(self.calibrate_drone)
        layout.addWidget(self.calibrate_button)

        self.calibrate_status_label = QLabel('Calibration Status: Not started', self)
        layout.addWidget(self.calibrate_status_label)

        self.calibrate_tab.setLayout(layout)

    def calibrate_drone(self):
        self.calibrate_status_label.setText("Calibration Status: In Progress")
        print("Drone's battery is this: ", str(self.me.get_battery()))
        self.me.takeoff()
        time.sleep(3)
        self.me.land()
        self.calibrate_status_label.setText("Calibration Status: Completed")

    def set_base_flag(self):
        self.base_flag = 1
        print("Base flag set to:", self.base_flag)
        self.status_label.setText('Status: Base flag set')

    def initDrone(self):
        self.model = YOLO('best.pt')
        self.class_names = self.model.names

        self.me = tello.Tello()
        self.me.connect()
        print(self.me.get_battery())

        self.me.streamon()

        self.cap = cv2.VideoCapture(1)

        self.hsvVals = [0, 0, 133, 179, 41, 242]
        self.sensors = 3
        self.threshold = 0.2
        self.width, self.height = 480, 360
        self.senstivity = 3
        self.weights = [-25, -15, 0, 15, 25]
        self.fSpeed = 7
        self.curve = 0

        self.isBaseDetectedForRotating = False

    def updateBatteryStatus(self):
        battery_level = self.me.get_battery()
        self.battery_bar.setValue(battery_level)
        if battery_level < 30:
            self.battery_bar.setStyleSheet("""
                QProgressBar::chunk { 
                    background-color: red; 
                    width: 20px;
                }
            """)
        else:
            self.battery_bar.setStyleSheet("""
                QProgressBar::chunk { 
                    background-color: green; 
                    width: 20px;
                }
            """)
        self.battery_bar.setFormat(f"Battery: {battery_level}%")
        print(f"Drone's battery is this: {battery_level}%")

    def thresholding(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([self.hsvVals[0], self.hsvVals[1], self.hsvVals[2]])
        upper = np.array([self.hsvVals[3], self.hsvVals[4], self.hsvVals[5]])
        mask = cv2.inRange(hsv, lower, upper)
        return mask

    def getContours(self, imgThres, img):
        cx = 0
        contours, hieracrhy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            biggest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(biggest)
            cx = x + w // 2
            cy = y + h // 2
            cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        return cx

    def getSensorOutput(self, imgThres, sensors):
        imgs = np.hsplit(imgThres, sensors)
        totalPixels = (imgThres.shape[1] // sensors) * imgThres.shape[0]
        senOut = []
        for x, im in enumerate(imgs):
            pixelCount = cv2.countNonZero(im)
            if pixelCount > self.threshold * totalPixels:
                senOut.append(1)
            else:
                senOut.append(0)
            cv2.imshow(str(x), im)
        print(senOut)
        return senOut

    def sendCommands(self, senOut, cx, base_detected, base_coords):
        if base_detected and self.base_flag == 1:
            self.me.send_rc_control(0, 0, 0, 0)

            bx, by, bw, bh = base_coords
            bcx = int(bx + bw // 2)
            bcy = int(by + bh // 2)
            print("Base coordinates: ", base_coords)
            x_axis = 0
            y_axis = 0
            center_x, center_y = self.width // 2, self.height // 2

            while True:
                print("center x = " + str(center_x) + "center y = " + str(center_y))
                print("base center x = " + str(bcx) + "base center y = " + str(bcy))
                print("Base ortalanıyor...")
                self.me.send_rc_control(0, 0, 0, 0)
                _, img = self.cap.read()
                img = self.me.get_frame_read().frame
                img = cv2.flip(img, 0)
                img = cv2.resize(img, (self.width, self.height))
                bcx, bcy = self.getBaseCenter(img, self.model)

                if bcx > center_x + 50 and x_axis < 2:
                    self.me.move_right(20)
                    print(bcx)
                    print("sağ yaptı")
                    continue
                elif bcx < center_x - 35 and x_axis < 2:
                    print(bcx)
                    print("sol yaptı")
                    self.me.move_left(20)
                    continue
                else:
                    x_axis += 1
                    print("x ekseninde dönüş tamamlandı")

                if bcy > center_y + 35 and y_axis < 2:
                    self.me.move_back(20)
                    continue
                elif bcy < center_y - 35 and y_axis < 2:
                    self.me.move_forward(20)
                    continue
                else:
                    y_axis += 1
                    self.me.send_rc_control(-15, 0, 0, 0)
                    time.sleep(0.4)
                    self.me.land()
            return

        if base_detected and self.base_flag == 0:
            if not self.isBaseDetectedForRotating:
                self.isBaseDetectedForRotating = True
                self.me.rotate_clockwise(180)

        if not base_detected:
            self.isBaseDetectedForRotating = False

        lr = (cx - self.width // 2) // self.senstivity
        lr = int(np.clip(lr, -10, 10))
        if 2 > lr > -2: lr = 0

        print("senOut=" + str(senOut))
        if senOut == [1, 0, 0]:
            self.curve = self.weights[0]
        elif senOut == [1, 1, 0]:
            self.curve = self.weights[1]
        elif senOut == [0, 1, 0]:
            self.curve = self.weights[2]
        elif senOut == [0, 1, 1]:
            self.curve = self.weights[3]
        elif senOut == [0, 0, 1]:
            self.curve = self.weights[4]
        elif senOut == [0, 0, 0]:
            self.curve = self.weights[2]
        elif senOut == [1, 1, 1]:
            self.curve = self.weights[2]
        elif senOut == [1, 0, 1]:
            self.curve = self.weights[2]

        self.me.send_rc_control(lr, self.fSpeed, 0, self.curve)
        time.sleep(0.2)

    def getBaseCenter(self, img, model):
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img2)
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_index = int(box.cls)
                    cls_name = self.class_names[cls_index]
                    if cls_name == 'base':
                        bx, by, bw, bh = box.xyxy[0].cpu().numpy()
                        bcx = int(bx + bw // 2)
                        bcy = int(by + bh // 2)
                        return bcx, bcy
        return self.width // 2, self.height // 2

    def start_main(self):
        self.status_label.setText('Status: Running')
        self.i = 0
        self.sikCount = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)  # Update every 100ms

    def update_frame(self):
        _, img = self.cap.read()
        img = self.me.get_frame_read().frame

        if self.i == 0:
            self.me.takeoff()
            self.i += 1

        img = cv2.resize(img, (self.width, self.height))
        img = cv2.flip(img, 0)

        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(img2, show=False, conf=0.6)

        result_image = results[0].plot()

        base_detected = False
        base_coords = None
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_index = int(box.cls)
                    cls_name = self.class_names[cls_index]
                    if cls_name == 'base':
                        base_detected = True
                        base_coords = box.xyxy[0].cpu().numpy()
                        break
                if base_detected:
                    self.sikCount += 1
                    break
        if self.sikCount < 10:
            base_detected = False

        imgThres = self.thresholding(img)
        cx = self.getContours(imgThres, img)
        senOut = self.getSensorOutput(imgThres, self.sensors)

        self.sendCommands(senOut, cx, base_detected, base_coords)

        self.display_image(result_image)

    def display_image(self, img):
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(img2.data, img2.shape[1], img2.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)


def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
