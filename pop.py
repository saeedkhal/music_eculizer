
import pyqtgraph as pg
from UI import Ui_MainWindow
from PyQt5 import QtWidgets ,QtCore, QtGui
import shutil
from scipy.io import wavfile
import numpy as np
import sys
import os
from scipy.fftpack import fft
#from funcations import funcation as f
import wave
import struct
from scipy import signal
from playsound import playsound
from PDF import PDF
from fpdf import FPDF
from PDF import PDF
import pyqtgraph.exporters   #for taking image then take an pdf
from collections import OrderedDict



class popWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(popWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # <<<<<<<<<<<<<<<<<        ############    actions         >>>>>>>>>>>>>>>>>>>>>>>>>>

        self.ui.new_window.clicked.connect(lambda: self.pop())
        self.ui.pdf.clicked.connect(lambda: self.create_pdf())
        self.ui.spectro_reset.clicked.connect(lambda: self.spectro_reset())
        self.ui.actionopen_signal.triggered.connect(lambda: self.load_audio_feil())
        self.ui.zoomout.clicked.connect(lambda: self.zoomout())
        self.ui.zoomin.clicked.connect(lambda: self.zoomin())
        self.ui.o_play.clicked.connect(lambda: self.play_audio())
        self.ui.m_play.clicked.connect(lambda: self.play2())
        self.ui.verticalSlider.valueChanged.connect(lambda: self.draw_the_new_signal_and_spectrogram())
        self.ui.verticalSlider_2.valueChanged.connect(lambda: self.draw_the_new_signal_and_spectrogram())
        self.ui.verticalSlider_3.valueChanged.connect(lambda: self.draw_the_new_signal_and_spectrogram())
        self.ui.verticalSlider_4.valueChanged.connect(lambda: self.draw_the_new_signal_and_spectrogram())
        self.ui.verticalSlider_5.valueChanged.connect(lambda: self.draw_the_new_signal_and_spectrogram())
        self.ui.verticalSlider_6.valueChanged.connect(lambda: self.draw_the_new_signal_and_spectrogram())
        self.ui.verticalSlider_7.valueChanged.connect(lambda: self.draw_the_new_signal_and_spectrogram())
        self.ui.verticalSlider_8.valueChanged.connect(lambda: self.draw_the_new_signal_and_spectrogram())
        self.ui.verticalSlider_9.valueChanged.connect(lambda: self.draw_the_new_signal_and_spectrogram())
        self.ui.verticalSlider_10.valueChanged.connect(lambda: self.draw_the_new_signal_and_spectrogram())

    # <<<<<<<<<<<<<<<<<        ############    poping          >>>>>>>>>>>>>>>>>>>>>>>>>>

    def pop(self):
        self.pop_window = popWindow()
        self.pop_window.show()

    # <<<<<<<<<<<<<<<<<        ############    pdf         >>>>>>>>>>>>>>>>>>>>>>>>>>

    def create_pdf(self):  ##take pdf to spectrogram of chaneel 1
        PLOT_DIR = 'pdf_plots'
        PDF_DIR = 'pdf_pdf'

        try:
            shutil.rmtree(PDF_DIR)
            shutil.rmtree(PLOT_DIR)
            os.mkdir(PLOT_DIR)
            os.mkdir(PDF_DIR)
        except FileNotFoundError:
            os.mkdir(PDF_DIR)
            os.mkdir(PLOT_DIR)

        exporter = pg.exporters.ImageExporter(
            self.ui.o_signal.scene())
        exporter.export(f'{PLOT_DIR}/0.jpg')

        exporter = pg.exporters.ImageExporter(
            self.ui.m_signal.scene())
        exporter.export(f'{PLOT_DIR}/1.jpg')

        exporter = pg.exporters.ImageExporter(
            self.ui.o_spect.scene())
        exporter.export(f'{PLOT_DIR}/3.jpg')

        exporter = pg.exporters.ImageExporter(
            self.ui.m_spect.scene())
        exporter.export(f'{PLOT_DIR}/4.jpg')

        pdf = PDF()
        images = pdf.construct(PLOT_DIR)
        print(images[0:])
        pdf.print_page(images, PLOT_DIR)
        pdf.output(f'{PDF_DIR}/pdf.pdf', 'F')

    # <<<<<<<<<<<<<<<<<        ############    zooming         >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def zoomout(self):
        self.ui.o_signal.plotItem.getViewBox().scaleBy((.5, .5))
        self.ui.m_signal.plotItem.getViewBox().scaleBy((.5, .5))

    def zoomin(self):
        self.ui.o_signal.plotItem.getViewBox().scaleBy((2, 2))
        self.ui.m_signal.plotItem.getViewBox().scaleBy((2, 2))

    # <<<<<<<<<<<<<<<<<        ############    orignal file          >>>>>>>>>>>>>>>>>>>>>>>>>>

    def load_audio_feil(self):
        self.fname1 = QtGui.QFileDialog.getOpenFileName(None, 'Open only wav', os.getenv('HOME'),
                                                        "wav(*.wav)")  ##open to for browsing
        path = self.fname1[0]  # get fiel path
        self.infile = path
        self.num_samples = 308700
        self.sampling_rate = 308700.0 / 7.0
        wav_file = wave.open(self.infile, 'r')  # open the  file
        data = wav_file.readframes(self.num_samples)  # data is an arry have the signal with hexa
        wav_file.close()
        data = struct.unpack('{n}h'.format(n=self.num_samples), data)  # convert every elemnt from hexa to decimle
        self.data = np.array(data)

        self.time = np.arange(0, 7, 1 / (
            self.sampling_rate))  ######### time array for x axes ==> [0,1/sampling_rate,2/sampling_rate,...........................,7]sec

        #######

        self.data_fft = np.fft.fft(data)  # array to convert the wave data sample to fourer ==> x+yj

        self.frequencies = np.abs(self.data_fft)  # array to get the absolute of every componant ==> sqer(x.x+y.y)
        print(
            "The frequency is {} Hz".format(np.argmax(self.frequencies)))  # this will get the maximum maximum freqency
        self.ui.o_signal.plot(self.time, data)
        self.draw_spectrogram_of_channel_1()

    ############

    # <<<<<<<<<<<<<<<<<        ############    orignal spectrogram         >>>>>>>>>>>>>>>>>>>>>>>>>>

    def draw_spectrogram_of_channel_1(self):
        fs = self.sampling_rate
        array_of_sample = np.array(self.data)
        f, t, Sxx = signal.spectrogram(array_of_sample, fs)
        pg.setConfigOptions(imageAxisOrder='row-major')
        ########## limits
        self.ui.o_spect.plotItem.setLimits(xMin=0, xMax=7, yMin=min(f), yMax=max(f))

        win = self.ui.o_spect
        img = pg.ImageItem()
        win.addItem(img)
        hist = pg.HistogramLUTItem()
        hist.setImageItem(img)
        win.addItem(hist)
        hist.setLevels(np.min(Sxx), np.max(Sxx))

        self.color_pallets()
        hist.gradient.restoreState(self.Gradients[self.text])

        img.setImage(Sxx)
        img.scale(t[-1] / np.size(Sxx, axis=1), f[-1] / np.size(Sxx, axis=0))
        win.setLabel('bottom', "Time", units='s')
        win.setLabel('left', "Frequency", units='Hz')
        ######33 limits
        self.ui.o_signal.plotItem.setLimits(xMin=0, xMax=7, yMin=min(self.data), yMax=max(self.data))

        self.ui.o_spect.plotItem.setXRange(0, 7)
        self.ui.o_spect.plotItem.setYRange(0, 4000)

    ##############

    # <<<<<<<<<<<<<<<<<        ############    play orignal signal         >>>>>>>>>>>>>>>>>>>>>>>>>>

    def play_audio(self):
        playsound(self.infile)  # for play the audio    def play_audio(self):

    # <<<<<<<<<<<<<<<<<        ############    draw modified signal         >>>>>>>>>>>>>>>>>>>>>>>>>>

    def draw_the_new_signal_and_spectrogram(self):
        self.frequencies = np.abs(self.data_fft)  # array to get the absolute of every componant ==> sqer(x.x+y.y)

        if (self.ui.high_freq_slider.value() == 0):
            first_band_from_originalsignal = np.array([])
            first_band_from_originalsignal = (self.frequencies[
                                              :2000]) * self.ui.verticalSlider.value()  ### get the band freqency from 0HZ to 1600HZ and multible in gain of slidr value

            secound_band_from_originalsignal = np.array([])
            secound_band_from_originalsignal = (self.frequencies[2000:4000]) * self.ui.verticalSlider_2.value()

            third_band_from_originalsignal = np.array([])
            third_band_from_originalsignal = (self.frequencies[4000:6000]) * self.ui.verticalSlider_3.value()

            fourth_band_from_originalsignal = np.array([])
            fourth_band_from_originalsignal = (self.frequencies[6000:8000]) * self.ui.verticalSlider_4.value()

            fifth_band_from_originalsignal = np.array([])
            fifth_band_from_originalsignal = (self.frequencies[8000:10000]) * self.ui.verticalSlider_5.value()

            sixth_band_from_originalsignal = np.array([])
            sixth_band_from_originalsignal = (self.frequencies[10000:12000]) * 0

            seventh_band_from_originalsignal = np.array([])
            seventh_band_from_originalsignal = (self.frequencies[12000:14000]) * 0

            eighth_band_from_originalsignal = np.array([])
            eighth_band_from_originalsignal = (self.frequencies[14000:16000]) * 0

            ninth_band_from_originalsignal = np.array([])
            ninth_band_from_originalsignal = (self.frequencies[16000:18000]) * 0

            tenth_band_from_originalsignal = np.array([])
            tenth_band_from_originalsignal = (self.frequencies[18000:]) * 0


        elif (self.ui.low_freq_slider.value() == 0):
            first_band_from_originalsignal = np.array([])
            first_band_from_originalsignal = (self.frequencies[
                                              :2000]) * 0  ### get the band freqency from 0HZ to 1600HZ and multible in gain of slidr value

            secound_band_from_originalsignal = np.array([])
            secound_band_from_originalsignal = (self.frequencies[2000:4000]) * 0

            third_band_from_originalsignal = np.array([])
            third_band_from_originalsignal = (self.frequencies[4000:6000]) * 0

            fourth_band_from_originalsignal = np.array([])
            fourth_band_from_originalsignal = (self.frequencies[6000:8000]) * 0

            fifth_band_from_originalsignal = np.array([])
            fifth_band_from_originalsignal = (self.frequencies[8000:10000]) * 0

            sixth_band_from_originalsignal = np.array([])
            sixth_band_from_originalsignal = (self.frequencies[10000:12000]) * self.ui.verticalSlider_6.value()

            seventh_band_from_originalsignal = np.array([])
            seventh_band_from_originalsignal = (self.frequencies[12000:14000]) * self.ui.verticalSlider_7.value()

            eighth_band_from_originalsignal = np.array([])
            eighth_band_from_originalsignal = (self.frequencies[14000:16000]) * self.ui.verticalSlider_8.value()

            ninth_band_from_originalsignal = np.array([])
            ninth_band_from_originalsignal = (self.frequencies[16000:18000]) * self.ui.verticalSlider_9.value()

            tenth_band_from_originalsignal = np.array([])
            tenth_band_from_originalsignal = (self.frequencies[18000:]) * self.ui.verticalSlider_10.value()



        else:
            first_band_from_originalsignal = np.array([])
            first_band_from_originalsignal = (self.frequencies[
                                              :2000]) * self.ui.verticalSlider.value()  ### get the band freqency from 0HZ to 1600HZ and multible in gain of slidr value

            secound_band_from_originalsignal = np.array([])
            secound_band_from_originalsignal = (self.frequencies[2000:4000]) * self.ui.verticalSlider_2.value()

            third_band_from_originalsignal = np.array([])
            third_band_from_originalsignal = (self.frequencies[4000:6000]) * self.ui.verticalSlider_3.value()

            fourth_band_from_originalsignal = np.array([])
            fourth_band_from_originalsignal = (self.frequencies[6000:8000]) * self.ui.verticalSlider_4.value()

            fifth_band_from_originalsignal = np.array([])
            fifth_band_from_originalsignal = (self.frequencies[8000:10000]) * self.ui.verticalSlider_5.value()

            sixth_band_from_originalsignal = np.array([])
            sixth_band_from_originalsignal = (self.frequencies[10000:12000]) * self.ui.verticalSlider_6.value()

            seventh_band_from_originalsignal = np.array([])
            seventh_band_from_originalsignal = (self.frequencies[12000:14000]) * self.ui.verticalSlider_7.value()

            eighth_band_from_originalsignal = np.array([])
            eighth_band_from_originalsignal = (self.frequencies[14000:16000]) * self.ui.verticalSlider_8.value()

            ninth_band_from_originalsignal = np.array([])
            ninth_band_from_originalsignal = (self.frequencies[16000:18000]) * self.ui.verticalSlider_9.value()

            tenth_band_from_originalsignal = np.array([])
            tenth_band_from_originalsignal = (self.frequencies[18000:]) * self.ui.verticalSlider_10.value()

        self.new_freqencyes = np.concatenate((first_band_from_originalsignal, secound_band_from_originalsignal,
                                              third_band_from_originalsignal, fourth_band_from_originalsignal,
                                              fifth_band_from_originalsignal, sixth_band_from_originalsignal,
                                              seventh_band_from_originalsignal, eighth_band_from_originalsignal,
                                              ninth_band_from_originalsignal,
                                              tenth_band_from_originalsignal,
                                              ))  ###make anew arrey after multiply the diffrent band in gain

        phase = np.angle(self.data_fft)  # get the phase

        self.new_signal_after_add_gain_and_phase = np.multiply(self.new_freqencyes, np.exp(
            1j * phase))  # new_signal_after_add_gain_and_phase==>gain*(x+yj)

        self.new_signal = np.real(
            np.fft.ifft(self.new_signal_after_add_gain_and_phase))  ### convrt the signal to the time domain
        ########## limits
        self.ui.m_signal.plotItem.setLimits(xMin=0, xMax=7, yMin=min(self.data), yMax=max(self.data))
        self.ui.m_signal.clear()
        self.ui.m_signal.plot(self.time, self.new_signal)
        self.draw_the_spectrogram_of_modified_signal()

    # <<<<<<<<<<<<<<<<<        ############    modified spectrogram         >>>>>>>>>>>>>>>>>>>>>>>>>>

    def draw_the_spectrogram_of_modified_signal(self):
        fs = self.sampling_rate
        array_of_sample = np.array(self.new_signal)
        f, t, Sxx = signal.spectrogram(array_of_sample, fs)
        pg.setConfigOptions(imageAxisOrder='row-major')
        ########## limits
        self.ui.m_spect.plotItem.setLimits(xMin=0, xMax=7, yMin=min(f), yMax=max(f))

        win = self.ui.m_spect
        img = pg.ImageItem()
        win.addItem(img)
        hist = pg.HistogramLUTItem()
        hist.setImageItem(img)
        win.addItem(hist)
        hist.setLevels(np.min(Sxx), np.max(Sxx))

        self.color_pallets()
        hist.gradient.restoreState(self.Gradients[self.text])

        img.setImage(Sxx)
        img.scale(t[-1] / np.size(Sxx, axis=1), f[-1] / np.size(Sxx, axis=0))
        win.setLabel('bottom', "Time", units='s')
        win.setLabel('left', "Frequency", units='Hz')
        self.ui.m_spect.plotItem.setXRange(0, 7)
        self.ui.m_spect.plotItem.setYRange(0, 4000)

    # <<<<<<<<<<<<<<<<<        ############    play modified signal        >>>>>>>>>>>>>>>>>>>>>>>>>>

    def play_the_modified_signal(self):
        file = "modified_signal.wav"
        wav_file = wave.open(file, 'w')
        comptype = "NONE"
        compname = "not compressed"
        nchannels = 1
        sampwidth = 2
        wav_file.setparams((nchannels, sampwidth, int(self.sampling_rate), self.num_samples, comptype,
                            compname))  ###make awav file with this specifications

        for s in self.new_signal:  #####take every sample then put it in the file we just created
            wav_file.writeframes(struct.pack('q', int(s)))

    def play2(self):
        self.play_the_modified_signal()
        playsound("modified_signal.wav")  # for play the audio

    # <<<<<<<<<<<<<<<<<        ############    color pallets          >>>>>>>>>>>>>>>>>>>>>>>>>>

    def color_pallets(self):
        self.Gradients = OrderedDict([
            ('magma', {'ticks': [(0.0, (0, 0, 3, 255)), (0.25, (80, 18, 123, 255)), (0.5, (182, 54, 121, 255)),
                                 (0.75, (251, 136, 97, 255)), (1.0, (251, 252, 191, 255))], 'mode': 'rgb'}),

            ('yellowy', {'ticks': [(0.0, (0, 0, 0, 255)), (0.2328863796753704, (32, 0, 129, 255)),
                                   (0.8362738179251941, (255, 255, 0, 255)), (0.5257586450247, (115, 15, 255, 255)),
                                   (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}),
            ('spectrum', {'ticks': [(1.0, (255, 0, 255, 255)), (0.0, (255, 0, 0, 255))], 'mode': 'hsv'}),
            ('greyclip',
             {'ticks': [(0.0, (0, 0, 0, 255)), (0.99, (255, 255, 255, 255)), (1.0, (255, 0, 0, 255))], 'mode': 'rgb'}),

            ('grey', {'ticks': [(0.0, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}),
        ])

        self.text = self.ui.color_pallets.currentText()

    def spectro_reset(self):
        self.draw_the_new_signal_and_spectrogram()
        self.draw_spectrogram_of_channel_1()
        # self.draw_the_spectrogram_of_modified_signal()

# <<<<<<<<<<<<<<<<<        ############    end         >>>>>>>>>>>>>>>>>>>>>>>>>>

