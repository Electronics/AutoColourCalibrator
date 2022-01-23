import scipy.interpolate
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import math
import cv2 as cv
import numpy as np
import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QSlider, QGroupBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QEvent, QPoint
import scipy.interpolate

from LabeledSlider import LabeledSlider

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

def scipy_bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Closed curve
    if periodic:
        kv = np.arange(-degree,count+degree+1)
        factor, fraction = divmod(count+degree+1, count)
        cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)),-1,axis=0)
        degree = np.clip(degree,1,degree)

    # Opened curve
    else:
        degree = np.clip(degree,1,count-1)
        kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

    # Return samples
    max_param = count - (degree * (1-periodic))
    spl = scipy.interpolate.BSpline(kv, cv, degree)
    return spl(np.linspace(0,max_param,n))

class App(QWidget):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("Lightroom-style tone curve LUT generator")

		hbox = QHBoxLayout()
		self.plt = MplCanvas(self)
		hbox.addWidget(self.plt)
		vBoxRight = QVBoxLayout()

		self.sliderHighlights = LabeledSlider(-100,100,50)
		self.sliderHighlights.sl.valueChanged.connect(self.computeToneCurve)
		self.sliderWhites = LabeledSlider(-100, 100, 50)
		self.sliderWhites.sl.valueChanged.connect(self.computeToneCurve)
		self.sliderBlacks = LabeledSlider(-100, 100, 50)
		self.sliderBlacks.sl.valueChanged.connect(self.computeToneCurve)
		self.sliderShadows = LabeledSlider(-100, 100, 50)
		self.sliderShadows.sl.valueChanged.connect(self.computeToneCurve)
		vBoxRight.addWidget(self.sliderHighlights)
		vBoxRight.addWidget(self.sliderWhites)
		vBoxRight.addWidget(self.sliderBlacks)
		vBoxRight.addWidget(self.sliderShadows)
		self.showQuantised = QCheckBox("Show Quantised")
		self.showQuantised.clicked.connect(self.computeToneCurve)
		vBoxRight.addWidget(self.showQuantised)
		vBoxRight.addStretch()
		hbox.addLayout(vBoxRight)
		hbox.setContentsMargins(0, 0, 0, 0)

		self.setLayout(hbox)
		self.computeToneCurve()

	def computeToneCurve(self):
		# I can probably totally replicate the lightroom slider set, but more effort
		# (to do this, generate a pair (highest and lowest) curves (cubic for most) for each adjustment range, limit their outputs and then average all curves at all points)

		axes = self.plt.axes
		axes.cla()

		variance = 7
		gradient = 10
		highlights = self.sliderHighlights.sl.value()
		shadows = self.sliderShadows.sl.value()
		blacks = self.sliderBlacks.sl.value()/2
		whites = self.sliderWhites.sl.value()/2

		shadows += blacks*0.8 # blacks and whites need to move the near points as well
		highlights += whites*0.8

		blackTemp = blacks + whites*0.2
		whites += blacks*0.2
		blacks = blackTemp

		shadow_midpoint = 27
		highlights_midpoint = 200
		blacks_midpoint = 105
		whites_midpoint = 149


		cv = np.array([[0, 0],
					   [variance * shadows / (-10*gradient) + shadow_midpoint + variance, variance * shadows / (10) + shadow_midpoint + variance],
					   [variance * blacks / (5*gradient) + blacks_midpoint + variance, variance * blacks / (5) + blacks_midpoint + variance],
					   [variance * whites / (-5*gradient) + whites_midpoint + variance, variance * whites / (5) + whites_midpoint + variance],
					   [variance * highlights / (10*gradient) + highlights_midpoint + variance, variance * highlights / (10) + highlights_midpoint + variance],
					   [255, 255]])


		p = scipy_bspline(cv, n=256, degree=2, periodic=False)
		xout, yout = p.T

		axes.plot(cv[:, 0], cv[:, 1], 'o-', label='Control Points')

		yout[np.where(yout<0)] = 0
		yout[np.where(yout>255)] = 255

		if self.showQuantised.isChecked():
			# now we want to fit to our rigid 0-255 int scale
			f = scipy.interpolate.splrep(xout,yout) # make a spline representation
			xout = np.arange(0,256)
			yout = scipy.interpolate.splev(xout, f).astype(np.uint8)
		axes.plot(xout, yout)

		axes.minorticks_on()
		axes.legend()
		axes.set_xlabel('x')
		axes.set_ylabel('y')
		axes.set_xlim(0, 255)
		axes.set_ylim(0, 255)
		axes.set_aspect('equal', adjustable='box')

		self.plt.draw()

	def closeEvent(self, event):
		event.accept()

if __name__=="__main__":
	app = QApplication(sys.argv)
	a = App()
	a.show()
	sys.exit(app.exec_())