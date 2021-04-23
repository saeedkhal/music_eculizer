# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'task_equalizer.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(974, 669)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.o_signal = PlotWidget(self.centralwidget)
        self.o_signal.setGeometry(QtCore.QRect(10, 50, 471, 201))
        self.o_signal.setObjectName("o_signal")
        self.m_signal = PlotWidget(self.centralwidget)
        self.m_signal.setGeometry(QtCore.QRect(10, 390, 481, 201))
        self.m_signal.setObjectName("m_signal")
        self.m_sig = QtWidgets.QLabel(self.centralwidget)
        self.m_sig.setGeometry(QtCore.QRect(20, 590, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.m_sig.setFont(font)
        self.m_sig.setObjectName("m_sig")
        self.o_sig = QtWidgets.QLabel(self.centralwidget)
        self.o_sig.setGeometry(QtCore.QRect(10, 260, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.o_sig.setFont(font)
        self.o_sig.setObjectName("o_sig")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(790, 260, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(770, 600, 221, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.signal_equilizer = QtWidgets.QLabel(self.centralwidget)
        self.signal_equilizer.setGeometry(QtCore.QRect(300, 0, 391, 41))
        font = QtGui.QFont()
        font.setFamily("Ravie")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.signal_equilizer.setFont(font)
        self.signal_equilizer.setMouseTracking(False)
        self.signal_equilizer.setFrameShadow(QtWidgets.QFrame.Plain)
        self.signal_equilizer.setObjectName("signal_equilizer")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(240, 280, 421, 86))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalSlider_10 = QtWidgets.QSlider(self.layoutWidget)
        self.verticalSlider_10.setMaximum(5)
        self.verticalSlider_10.setSliderPosition(1)
        self.verticalSlider_10.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_10.setObjectName("verticalSlider_10")
        self.horizontalLayout.addWidget(self.verticalSlider_10)
        self.verticalSlider_9 = QtWidgets.QSlider(self.layoutWidget)
        self.verticalSlider_9.setMaximum(5)
        self.verticalSlider_9.setSliderPosition(1)
        self.verticalSlider_9.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_9.setObjectName("verticalSlider_9")
        self.horizontalLayout.addWidget(self.verticalSlider_9)
        self.verticalSlider_8 = QtWidgets.QSlider(self.layoutWidget)
        self.verticalSlider_8.setMaximum(5)
        self.verticalSlider_8.setSliderPosition(1)
        self.verticalSlider_8.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_8.setObjectName("verticalSlider_8")
        self.horizontalLayout.addWidget(self.verticalSlider_8)
        self.verticalSlider_7 = QtWidgets.QSlider(self.layoutWidget)
        self.verticalSlider_7.setMaximum(5)
        self.verticalSlider_7.setSliderPosition(1)
        self.verticalSlider_7.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_7.setObjectName("verticalSlider_7")
        self.horizontalLayout.addWidget(self.verticalSlider_7)
        self.verticalSlider_6 = QtWidgets.QSlider(self.layoutWidget)
        self.verticalSlider_6.setMaximum(5)
        self.verticalSlider_6.setSliderPosition(1)
        self.verticalSlider_6.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_6.setObjectName("verticalSlider_6")
        self.horizontalLayout.addWidget(self.verticalSlider_6)
        self.verticalSlider_5 = QtWidgets.QSlider(self.layoutWidget)
        self.verticalSlider_5.setMaximum(5)
        self.verticalSlider_5.setSliderPosition(1)
        self.verticalSlider_5.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_5.setObjectName("verticalSlider_5")
        self.horizontalLayout.addWidget(self.verticalSlider_5)
        self.verticalSlider_4 = QtWidgets.QSlider(self.layoutWidget)
        self.verticalSlider_4.setMaximum(5)
        self.verticalSlider_4.setSliderPosition(1)
        self.verticalSlider_4.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_4.setObjectName("verticalSlider_4")
        self.horizontalLayout.addWidget(self.verticalSlider_4)
        self.verticalSlider_3 = QtWidgets.QSlider(self.layoutWidget)
        self.verticalSlider_3.setMaximum(5)
        self.verticalSlider_3.setSliderPosition(1)
        self.verticalSlider_3.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_3.setObjectName("verticalSlider_3")
        self.horizontalLayout.addWidget(self.verticalSlider_3)
        self.verticalSlider_2 = QtWidgets.QSlider(self.layoutWidget)
        self.verticalSlider_2.setMaximum(5)
        self.verticalSlider_2.setSliderPosition(1)
        self.verticalSlider_2.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_2.setObjectName("verticalSlider_2")
        self.horizontalLayout.addWidget(self.verticalSlider_2)
        self.verticalSlider = QtWidgets.QSlider(self.layoutWidget)
        self.verticalSlider.setMaximum(5)
        self.verticalSlider.setSliderPosition(1)
        self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider.setObjectName("verticalSlider")
        self.horizontalLayout.addWidget(self.verticalSlider)
        self.new_window = QtWidgets.QPushButton(self.centralwidget)
        self.new_window.setGeometry(QtCore.QRect(20, 10, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Lucida Handwriting")
        font.setPointSize(10)
        self.new_window.setFont(font)
        self.new_window.setObjectName("new_window")
        self.color_pallets = QtWidgets.QComboBox(self.centralwidget)
        self.color_pallets.setGeometry(QtCore.QRect(690, 330, 111, 31))
        self.color_pallets.setObjectName("color_pallets")
        self.color_pallets.addItem("")
        self.color_pallets.addItem("")
        self.color_pallets.addItem("")
        self.color_pallets.addItem("")
        self.color_pallets.addItem("")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(700, 300, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.spectro_reset = QtWidgets.QPushButton(self.centralwidget)
        self.spectro_reset.setGeometry(QtCore.QRect(870, 320, 75, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.spectro_reset.setFont(font)
        self.spectro_reset.setObjectName("spectro_reset")
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(810, 320, 52, 41))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.high_freq_slider = QtWidgets.QSlider(self.layoutWidget1)
        self.high_freq_slider.setMaximum(1)
        self.high_freq_slider.setSingleStep(1)
        self.high_freq_slider.setProperty("value", 1)
        self.high_freq_slider.setOrientation(QtCore.Qt.Vertical)
        self.high_freq_slider.setObjectName("high_freq_slider")
        self.horizontalLayout_2.addWidget(self.high_freq_slider)
        self.low_freq_slider = QtWidgets.QSlider(self.layoutWidget1)
        self.low_freq_slider.setMaximum(1)
        self.low_freq_slider.setSingleStep(1)
        self.low_freq_slider.setProperty("value", 1)
        self.low_freq_slider.setOrientation(QtCore.Qt.Vertical)
        self.low_freq_slider.setObjectName("low_freq_slider")
        self.horizontalLayout_2.addWidget(self.low_freq_slider)
        self.pdf = QtWidgets.QPushButton(self.centralwidget)
        self.pdf.setGeometry(QtCore.QRect(14, 303, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pdf.setFont(font)
        self.pdf.setObjectName("pdf")
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(810, 300, 51, 16))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label = QtWidgets.QLabel(self.layoutWidget2)
        self.label.setObjectName("label")
        self.horizontalLayout_4.addWidget(self.label)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_4.addWidget(self.label_3)
        self.layoutWidget3 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget3.setGeometry(QtCore.QRect(160, 261, 41, 116))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget3)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.o_play = QtWidgets.QPushButton(self.layoutWidget3)
        self.o_play.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("t.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.o_play.setIcon(icon)
        self.o_play.setObjectName("o_play")
        self.verticalLayout.addWidget(self.o_play)
        self.zoomin = QtWidgets.QPushButton(self.layoutWidget3)
        self.zoomin.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("zoomin.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.zoomin.setIcon(icon1)
        self.zoomin.setObjectName("zoomin")
        self.verticalLayout.addWidget(self.zoomin)
        self.zoomout = QtWidgets.QPushButton(self.layoutWidget3)
        self.zoomout.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("zoomout.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.zoomout.setIcon(icon2)
        self.zoomout.setObjectName("zoomout")
        self.verticalLayout.addWidget(self.zoomout)
        self.m_play = QtWidgets.QPushButton(self.layoutWidget3)
        self.m_play.setText("")
        self.m_play.setIcon(icon)
        self.m_play.setObjectName("m_play")
        self.verticalLayout.addWidget(self.m_play)
        self.o_spect = GraphicsLayoutWidget(self.centralwidget)
        self.o_spect.setGeometry(QtCore.QRect(520, 51, 431, 201))
        self.o_spect.setObjectName("o_spect")
        self.m_spect = GraphicsLayoutWidget(self.centralwidget)
        self.m_spect.setGeometry(QtCore.QRect(520, 390, 431, 201))
        self.m_spect.setObjectName("m_spect")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 974, 21))
        self.menubar.setObjectName("menubar")
        self.menuopen = QtWidgets.QMenu(self.menubar)
        self.menuopen.setObjectName("menuopen")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionopen_signal = QtWidgets.QAction(MainWindow)
        self.actionopen_signal.setObjectName("actionopen_signal")
        self.actionNew_window = QtWidgets.QAction(MainWindow)
        self.actionNew_window.setObjectName("actionNew_window")
        self.menuopen.addAction(self.actionopen_signal)
        self.menubar.addAction(self.menuopen.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.m_sig.setText(_translate("MainWindow", "Modified signal"))
        self.o_sig.setText(_translate("MainWindow", "Original signal "))
        self.label_14.setText(_translate("MainWindow", "original spectrogram"))
        self.label_15.setText(_translate("MainWindow", "Modified spectrogram"))
        self.signal_equilizer.setText(_translate("MainWindow", "<<  SIGNAL  EQUIaLIZER  >>"))
        self.new_window.setText(_translate("MainWindow", "New Window"))
        self.new_window.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.color_pallets.setItemText(0, _translate("MainWindow", "yellowy"))
        self.color_pallets.setItemText(1, _translate("MainWindow", "greyclip"))
        self.color_pallets.setItemText(2, _translate("MainWindow", "spectrum"))
        self.color_pallets.setItemText(3, _translate("MainWindow", "grey"))
        self.color_pallets.setItemText(4, _translate("MainWindow", "magma"))
        self.label_2.setText(_translate("MainWindow", "color pallet"))
        self.spectro_reset.setText(_translate("MainWindow", "Reset"))
        self.spectro_reset.setShortcut(_translate("MainWindow", "Ctrl+R"))
        self.pdf.setText(_translate("MainWindow", "Create PDF"))
        self.pdf.setShortcut(_translate("MainWindow", "Ctrl+F"))
        self.label.setText(_translate("MainWindow", "HF"))
        self.label_3.setText(_translate("MainWindow", "LF"))
        self.o_play.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.zoomin.setShortcut(_translate("MainWindow", "Ctrl+X"))
        self.zoomout.setShortcut(_translate("MainWindow", "Ctrl+Z"))
        self.m_play.setShortcut(_translate("MainWindow", "Ctrl+M"))
        self.menuopen.setTitle(_translate("MainWindow", "file"))
        self.actionopen_signal.setText(_translate("MainWindow", "open signal"))
        self.actionopen_signal.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionNew_window.setText(_translate("MainWindow", "New window "))
from pyqtgraph import GraphicsLayoutWidget
from pyqtgraph import PlotWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
