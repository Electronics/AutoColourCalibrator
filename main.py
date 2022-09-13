import math
import os

import cv2
import cv2 as cv
import numpy as np
import sys

import scipy.interpolate
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QSlider, QGroupBox, QTabWidget, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QEvent, QPoint
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from CubeFile import CubeFile
from LabeledSlider import LabeledSlider


def distance(x1,y1,x2,y2):
	return math.sqrt((abs(x1)-abs(x2))**2+(abs(y1)-abs(y2))**2)

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

bins = np.arange(256).reshape(256,1)
def hist_curve(im):
    h = np.zeros((300,256,3))
    if len(im.shape) == 2:
        color = [(255,255,255)]
    elif im.shape[2] == 3:
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv.calcHist([im],[ch],None,[256],[0,256])
        cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins,hist)))
        cv.polylines(h,[pts],False,col)
    y=np.flipud(h)
    return y

class App(QWidget):
	points = []
	dragingPointIndex = -1

	initialised = False
	imgOutput = None
	imgOrig = None
	img = np.zeros((1080, 1920, 3), np.uint8)
	imgCropped = np.zeros((400, 280, 3), np.uint8)
	imgCroppedCorners = np.float32([(0, 0), (0, imgCropped.shape[0]), (imgCropped.shape[1], imgCropped.shape[0]), (imgCropped.shape[1], 0)])
	imgOutput = None
	lutimg = None

	DOTSIZE = 5  # size +/- the center point of a dot
	SAMPLE_REGION=10 # +- pixel region

	def __init__(self):
		super().__init__()
		self.setWindowTitle("Colour Calibration Camera Tool")
		# create the label that holds the image
		self.rawImage = QLabel(self)
		self.rawImage.resize(1400, 787)

		self.croppedImage = QLabel(self)
		self.croppedImage.resize(400,300)

		self.inputHistogram = QLabel(self)
		self.inputHistogram.resize(300,150)
		self.inputHistogram.move(100,100) # just temporarily move it somewhere
		self.inputHistogram.show()

		self.outputHistogram = QLabel(self)
		self.outputHistogram.resize(300, 150)
		self.outputHistogram.move(100, 100)  # just temporarily move it somewhere
		self.update_image_qt(np.zeros((150, 300, 3), np.uint8), self.outputHistogram)
		self.outputHistogram.show()

		self.colourPoints = []
		# colour list is sRGB!! (in RGB order), vertical columns, then left to right
		self.colourList = [(249,242,238), # grays - E
						   (202,198,195),
						   (161,157,154),
						   (122,118,116),
						   (80,80,78),
						   (43,41,43),
						   (0,127,159), #F
						   (192,75,145),
						   (245,205,0),
						   (186,26,51),
						   (57,146,64),
						   (25,55,135),
						   (222,118,32), #G
						   (58,88,159),
						   (195,79,95),
						   (83,58,106),
						   (157,188,54),
						   (238,158,25),
						   (98,187,166), #H
						   (126,125,174),
						   (82,106,60),
						   (87,120,155),
						   (197,145,125),
						   (112,76,60)]

		# create a vertical box layout and add the two labels
		hbox = QHBoxLayout()
		hbox.addWidget(self.rawImage)
		vBoxLeft = QVBoxLayout()
		hbox.addLayout(vBoxLeft)
		hbox.setContentsMargins(0,0,0,0)

		hBoxfile = QHBoxLayout()
		hBoxfile.setContentsMargins(0,0,0,0)
		fileOps = QWidget()
		fileOps.setLayout(hBoxfile)
		vBoxLeft.addWidget(fileOps)
		openButton = QPushButton("Open File")
		hBoxfile.addWidget(openButton)
		saveButton = QPushButton("Save LUT")
		hBoxfile.addWidget(saveButton)
		saveCubeButton = QPushButton("Save CUBE")
		hBoxfile.addWidget(saveCubeButton)
		openButton.clicked.connect(self.openFile)
		saveButton.clicked.connect(self.saveLUT)
		saveCubeButton.clicked.connect(self.saveCUBE)

		self.buttonUndo = QPushButton("<-- Undo Point")
		self.buttonUndo.clicked.connect(lambda:self.undoPoint())
		self.buttonClear = QPushButton("<-- Clear All Points")
		self.buttonClear.clicked.connect(lambda:self.clearPoints())
		vBoxLeft.addWidget(self.buttonUndo)
		vBoxLeft.addWidget(self.buttonClear)
		vBoxLeft.addWidget(QLabel("White should be top-left, black bottom-left"))
		vBoxLeft.addWidget(QLabel("Start clicking points by white, and go anti-clockwise"))
		self.labelPointsStatus = QLabel()
		vBoxLeft.addWidget(self.labelPointsStatus)
		vBoxLeft.addStretch()

		self.tabs = QTabWidget()
		vBoxLeft.addWidget(self.tabs)

		vBoxTone = QVBoxLayout()
		tone = QWidget()
		tone.setLayout(vBoxTone)
		self.tabs.addTab(tone, "Tone")
		self.sliderHighlights = LabeledSlider(-100, 100, 50, height=20)
		self.sliderHighlights.sl.valueChanged.connect(self.updateUI)
		self.sliderWhites = LabeledSlider(-100, 100, 50, height=20)
		self.sliderWhites.sl.valueChanged.connect(self.updateUI)
		self.sliderBlacks = LabeledSlider(-100, 100, 50, height=20)
		self.sliderBlacks.sl.valueChanged.connect(self.updateUI)
		self.sliderShadows = LabeledSlider(-100, 100, 50, height=20)
		self.sliderShadows.sl.valueChanged.connect(self.updateUI)
		toneClear = QPushButton("Reset All")
		def clearToneSliders():
			self.sliderBrightness.sl.setValue(0)
			self.sliderHighlights.sl.setValue(0)
			self.sliderWhites.sl.setValue(0)
			self.sliderBlacks.sl.setValue(0)
			self.sliderShadows.sl.setValue(0)
			self.updateUI()

		toneClear.clicked.connect(clearToneSliders)
		vBoxTone.addWidget(toneClear)
		self.sliderBrightness = LabeledSlider(-40, 40, 10, height=20)
		self.sliderBrightness.sl.valueChanged.connect(self.updateUI)
		vBoxTone.addWidget(QLabel("Brightness:"))
		vBoxTone.addWidget(self.sliderBrightness)
		vBoxTone.addWidget(QLabel("Highlights"))
		vBoxTone.addWidget(self.sliderHighlights)
		vBoxTone.addWidget(QLabel("Whites"))
		vBoxTone.addWidget(self.sliderWhites)
		vBoxTone.addWidget(QLabel("Blacks"))
		vBoxTone.addWidget(self.sliderBlacks)
		vBoxTone.addWidget(QLabel("Shadows"))
		vBoxTone.addWidget(self.sliderShadows)
		# self.showQuantised = QCheckBox("Show Quantised")
		# self.showQuantised.clicked.connect(self.updateUI)
		# vBoxTone.addWidget(self.showQuantised)

		self.groupBoxAdjustments = QGroupBox("Gamma Correction")
		self.tabs.addTab(self.groupBoxAdjustments, "Gamma (For Log Cameras)")
		vBoxAdjustments = QVBoxLayout()
		self.groupBoxAdjustments.setCheckable(True)
		self.groupBoxAdjustments.setChecked(False)
		self.groupBoxAdjustments.setLayout(vBoxAdjustments)
		vBoxAdjustments.addWidget(QLabel("These require some gamma correction to work..."))
		self.sliderGamma = LabeledSlider(0,250,50, divider=100, height=20)
		self.sliderGamma.sl.setValue(100)
		self.sliderGamma.sl.valueChanged.connect(self.updateUI)
		vBoxAdjustments.addWidget(QLabel("Gamma:"))
		vBoxAdjustments.addWidget(self.sliderGamma)
		self.sliderGammaHighlights = LabeledSlider(0, 600, 100, divider=200, height=20)
		self.sliderGammaHighlights.sl.setValue(200)
		self.sliderGammaHighlights.sl.valueChanged.connect(self.updateUI)
		vBoxAdjustments.addWidget(QLabel("Highlights:"))
		vBoxAdjustments.addWidget(self.sliderGammaHighlights)
		self.sliderGammaShadows = LabeledSlider(0, 600, 100, divider=200, height=20)
		self.sliderGammaShadows.sl.setValue(200)
		self.sliderGammaShadows.sl.valueChanged.connect(self.updateUI)
		vBoxAdjustments.addWidget(QLabel("Shadows:"))
		vBoxAdjustments.addWidget(self.sliderGammaShadows)

		groupBoxCalibration = QGroupBox("Calibration")
		vBoxCalibration = QVBoxLayout()
		groupBoxCalibration.setLayout(vBoxCalibration)
		vBoxLeft.addWidget(groupBoxCalibration)

		self.checkboxShowOutput = QCheckBox("Show output")
		self.checkboxShowOutput.setChecked(True)
		self.checkboxShowOutput.clicked.connect(self.updateUI)
		self.buttonResetColourPoints = QPushButton("Reset Colour Points")
		self.buttonResetColourPoints.clicked.connect(lambda:self.initaliseColourPoints())
		vBoxCalibration.addWidget(self.checkboxShowOutput)
		vBoxCalibration.addWidget(self.buttonResetColourPoints)
		vBoxCalibration.addWidget(self.croppedImage)
		self.buttonDoCalibration = QPushButton("Do Calibration")
		self.buttonDoCalibration.clicked.connect(self.doCalibration)
		vBoxCalibration.addWidget(self.buttonDoCalibration)


		self.setLayout(hbox)

		self.setMouseTracking(True)
		self.installEventFilter(self)

		self.initialised = True

		self.updateUI()
		self.initaliseColourPoints()

	def openFile(self,file=None):
		if not file:
			file,_ = QFileDialog.getOpenFileName(self,"Open Source Image", "", "PNG File (*.png);;All Files (*)")
			if not file:
				return
		self.imgOrig = cv.imread(file)
		self.img = self.imgOrig.copy()
		self.imgCropped = np.zeros((400, 280, 3), np.uint8)
		self.imgCroppedCorners = np.float32([(0, 0), (0, self.imgCropped.shape[0]), (self.imgCropped.shape[1], self.imgCropped.shape[0]), (self.imgCropped.shape[1], 0)])
		self.imgOutput = []
		self.lutimg = None
		self.points = []

		self.rawImage.resize(1400, 787)

		self.updateUI()

		# calculate input histogram

		fig = Figure((3, 1.5))
		canvas = FigureCanvas(fig)
		ax = fig.add_subplot(111)
		fig.subplots_adjust(0,0,1,1)
		color = ('#1900ff', '#00FB14', '#FB2A1F')
		maxY = 0
		for i, col in enumerate(color):
			histr = cv.calcHist([self.imgOrig], [i], None, [256], [0, 256])
			maxY = max(maxY, max(histr[1:-1]))
			ax.plot(histr, color=col)
		ax.set_xlim([0, 256])
		ax.set_ylim([0, min(maxY,2e5)])
		fig.set_facecolor("#000000")
		ax.axis("off")
		canvas.draw()
		width, height = fig.figbbox.width, fig.figbbox.height
		im = QImage(canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)
		self.inputHistogram.setPixmap(QPixmap(im))

		# move both histograms in the correct place
		# self.update_image_qt(hist, self.inputHistogram)
		self.inputHistogram.move(self.rawImage.pixmap().width() - self.inputHistogram.pixmap().width(), 0)
		self.update_image_qt(np.zeros((150, 300, 3), np.uint8), self.outputHistogram)
		self.outputHistogram.move(self.rawImage.pixmap().width() - self.outputHistogram.pixmap().width(), 170)

		self.updateUI() # yes repeated, but we need to update pixmaps to position the histogram correctly

	def saveLUT(self):
		if self.lutimg is None:
			return
		finalLUT = np.zeros_like(self.lutimg)
		if self.groupBoxAdjustments.isChecked():
			for i in range(3):  # because cv.add doesn't work properly and only affects the first channel
				finalLUT[:, :, i] = cv.add(self.gammaCorrection(self.lutimg)[:, :, i], self.sliderBrightness.sl.value())
			finalLUT[:, :, :] = self.computeToneCurve(finalLUT[:, :, :])
		else:
			for i in range(3):  # because cv.add doesn't work properly and only affects the first channel
				finalLUT[:, :, i] = cv.add(self.lutimg[:, :, i], self.sliderBrightness.sl.value())
			finalLUT[:, :, :] = self.computeToneCurve(finalLUT[:, :, :])

		fname, _ = QFileDialog.getSaveFileName(self,"Save LUT File","","PNG File (*.png);;All Files (*)")
		if fname:
			cv.imwrite(fname,finalLUT)

	def saveCUBE(self):
		fname, _ = QFileDialog.getSaveFileName(self, "Save LUT File", "", "CUBE File (*.cube);;All Files (*)")
		if fname:
			fileTxt = CubeFile(os.path.splitext(fname)[0], self.coes)
			with open(fname, "w") as file:
				file.write(fileTxt)


	def initaliseColourPoints(self):
		self.colourPoints.clear()
		i = 0
		COLS = 4
		ROWS = 6
		BORDER = 4
		widthSpacing = self.imgCropped.shape[1]/COLS - BORDER
		heightSpacing = self.imgCropped.shape[0]/ROWS - BORDER
		# top to bottom, left to right
		for c in self.colourList:
			x = int(math.floor(i/ROWS)*widthSpacing + widthSpacing/2+BORDER*2)
			y = int((i%ROWS)*heightSpacing + heightSpacing/2+BORDER*2)
			self.colourPoints.append((x,y,c))
			i+=1
		self.updateUI()

	def renderOutputHistogram(self):
		if not len(self.img)>1:
			return
		fig = Figure((3, 1.5))
		canvas = FigureCanvas(fig)
		ax = fig.add_subplot(111)
		fig.subplots_adjust(0, 0, 1, 1)
		color = ('#1900ff', '#00FB14', '#FB2A1F')
		maxY = 0
		for i, col in enumerate(color):
			histr = cv.calcHist([self.img], [i], None, [256], [0, 256])
			maxY = max(maxY, max(histr[1:-1]))
			ax.plot(histr, color=col)
		ax.set_xlim([0, 256])
		ax.set_ylim([0, min(maxY,2e5)])
		ax.axis("off")
		fig.set_facecolor("#000000")
		canvas.draw()
		width, height = fig.figbbox.width, fig.figbbox.height
		im = QImage(canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)
		self.outputHistogram.setPixmap(QPixmap(im))

	def overlayColourPoints(self):
		for x,y,c in self.colourPoints:
			cv.circle(self.imgCropped, (x,y), self.DOTSIZE+2, (c[2],c[1],c[0]), -1) # RGB -> BGR
			cv.circle(self.imgCropped, (x,y), 1, (255,255,255), -1)

	def removeDot(self, x, y):
		if (x, y) not in self.points:
			raise IndexError("Point to remove is not in points list!")
		self.points.remove((x, y))

	def renderCrop(self):
		transform = cv.getPerspectiveTransform(np.float32(self.points), self.imgCroppedCorners)
		self.imgCropped[:, :, :] = cv.warpPerspective(self.imgOrig, transform, (self.imgCropped.shape[1], self.imgCropped.shape[0]))

	def updateUI(self):
		if not self.initialised or self.imgOrig is None:
			self.update_image_qt(self.img, self.rawImage)
			self.update_image_qt(self.imgCropped, self.croppedImage)
			return

		if len(self.imgOutput)>1:
			if self.groupBoxAdjustments.isChecked():
				for i in range(3):  # because cv.add doesn't work properly and only affects the first channel
					self.img[:, :, i] = cv.add(self.gammaCorrection(self.imgOutput)[:, :, i], self.sliderBrightness.sl.value())
				self.img[:, :, :] = self.computeToneCurve(self.img[:, :, :])
			else:
				for i in range(3):  # because cv.add doesn't work properly and only affects the first channel
					self.img[:, :, i] = cv.add(self.imgOutput[:, :, i], self.sliderBrightness.sl.value())
				self.img[:, :, :] = self.computeToneCurve(self.img[:, :, :])

			self.renderOutputHistogram()

		# swapped this around so we can use self.img for the histogram rendering
		if self.checkboxShowOutput.isChecked() and len(self.imgOutput)>1:
			pass
		else:
			self.img[:, :, :] = self.imgOrig[:, :, :]

		index = 0
		for x, y in self.points:
			cv.circle(self.img, (x, y), self.DOTSIZE, (0, 255, 0), -1)
			if (x, y) != self.points[0]:
				# if it's not the last point, draw a line to the previous point
				cv.line(self.img, (x, y), self.points[index - 1], (0, 255, 0), 2)
			index += 1
		if len(self.points) == 4:
			# draw the final line
			cv.line(self.img, self.points[0], self.points[-1], (0, 255, 0), 2)

			# render the cropped image
			self.renderCrop()

			# colour the status text
			self.labelPointsStatus.setStyleSheet("QLabel { color: green; }")
		else:
			self.imgCropped = np.zeros((400, 280, 3), np.uint8)
			self.labelPointsStatus.setStyleSheet("QLabel { color: red; }")

		self.labelPointsStatus.setText(str(len(self.points))+"/4 points")

		self.overlayColourPoints()
		self.update_image_qt(self.img, self.rawImage)
		self.update_image_qt(self.imgCropped, self.croppedImage)

	def update_image_qt(self, cv_img, qtObj):
		"""Updates the image_label with a new opencv image"""
		qt_img = self.convert_cv_qt(cv_img, qtObj)
		qtObj.setPixmap(qt_img)

	def convert_cv_qt(self, cv_img, qtObj):
		"""Convert from an opencv image to QPixmap"""
		rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
		h, w, ch = rgb_image.shape
		bytes_per_line = ch * w
		convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
		p = convert_to_Qt_format.scaled(qtObj.size().width(), qtObj.size().height(), Qt.KeepAspectRatio)
		return QPixmap.fromImage(p)

	def getRelativeCoords(self,x,y,qtobj,image):
		# gets the pixel coordinates of an image (pixmap) (with shape) from the global mouse coords and the position of the qtobj
		# I FUCKING HATE PYQT and it's STUPID COORDINATE CRAP
		# p = self.image_label.mapFromGlobal(QPoint(x,y))
		# x,y p.x(), p.y()
		pixmapX,pixmapY = qtobj.x(), qtobj.y()
		if not isinstance(qtobj.parent(), App):
			pixmapX += qtobj.parent().x()
			pixmapY += qtobj.parent().y()
		objOffsetX = (qtobj.width() - qtobj.pixmap().rect().right()) / 2 # assuming centered horizontally
		objOffsetY = (qtobj.height() - qtobj.pixmap().rect().bottom()) / 2 # assuming centered vertically
		if math.floor(objOffsetX)>0:
			# the image must be constrained vertically (i.e. spacing left/right), only adjust X?
			#pixmapX += objOffsetX
			pass # do nothing??? it's fucking inconsistent
		if math.floor(objOffsetY)>0:
			# the image must be constrained horizontally (i.e. spacing up+down), only adjust Y?
			pixmapY += objOffsetY

		newX = (x-pixmapX)*(image.shape[1] / qtobj.pixmap().width())
		newY = (y-pixmapY)*(image.shape[0] / qtobj.pixmap().height())
		return int(newX),int(newY)


	def eventFilter(self, source, event):
		type = event.type()
		if type == QEvent.MouseMove:
			if self.rawImage.underMouse():
				if self.dragingPointIndex < 0:
					return False
				x, y = self.getRelativeCoords(event.x(), event.y(), self.rawImage, self.imgOrig)
				self.points[self.dragingPointIndex] = (x,y)
				self.updateUI()
				return True
			elif self.croppedImage.underMouse():
				if len(self.points)<4:
					return False
				if self.dragingPointIndex < 0:
					return False
				x, y = self.getRelativeCoords(event.x(), event.y(), self.croppedImage, self.imgCropped)
				self.colourPoints[self.dragingPointIndex] = (x, y, self.colourPoints[self.dragingPointIndex][2])
				self.updateUI()
				return True
		elif type == QEvent.MouseButtonPress:
			if self.rawImage.underMouse():
				x, y = self.getRelativeCoords(event.x(), event.y(), self.rawImage, self.imgOrig)
				i = 0
				for pointX, pointY in self.points:
					if distance(x,y, pointX, pointY) < 2 * self.DOTSIZE:
						print("Yoinked point", pointX, pointY, i)
						self.dragingPointIndex = i
					i += 1
				return True
			elif self.croppedImage.underMouse():
				x, y = self.getRelativeCoords(event.x(), event.y(), self.croppedImage, self.imgCropped)
				i=0
				for pointX, pointY,_ in self.colourPoints:
					if distance(x,y, pointX, pointY) < 4 * self.DOTSIZE:
						print("Yoinked colour point", pointX, pointY, i)
						self.dragingPointIndex = i
					i += 1
				return True

		elif type == QEvent.MouseButtonRelease:
			print("Released point", self.dragingPointIndex)
			self.dragingPointIndex = -1
			return True
		elif type == QEvent.MouseButtonDblClick:
			if self.rawImage.underMouse():
				if len(self.points) >= 4:
					return True
				x, y = self.getRelativeCoords(event.x(), event.y(), self.rawImage, self.imgOrig)
				deletedPoint = False
				for pointX, pointY in self.points:
					if distance(x,y, pointX, pointY) < 2 * self.DOTSIZE:
						print("Deleting point", pointX, pointY)
						self.removeDot(pointX, pointY)
						deletedPoint = True
				if not deletedPoint:
					print("Added point", x,y)
					self.points.append((x,y))
				self.updateUI()
				return True

		return False

	def undoPoint(self):
		if len(self.points) > 0:
			self.points.pop()
		self.updateUI()
	def clearPoints(self):
		self.points.clear()
		self.updateUI()

	def keyPressEvent(self, event):
		if event.key() == Qt.Key_Escape:
			self.clearPoints()
		elif event.key() == Qt.Key_Backspace:
			self.undoPoint()

	def doCalibration(self):
		if len(self.points)<4:
			print("Haven't selected a region yet")
			return
		if self.imgOrig is None:
			return # this shouldn't happen, but just in case
		self.buttonDoCalibration.setStyleSheet("background-color:green")
		self.buttonDoCalibration.repaint()

		# make a new LUT image as well
		try:
			lutimg = cv.imread("neutral-lut.png")
			lutimg = cv.cvtColor(lutimg, cv.COLOR_BGR2RGB)
			lutimg = lutimg.astype(np.float32, copy=False)
		except cv2.error:
			dlg = QMessageBox(self)
			dlg.setWindowTitle("Error")
			dlg.setText("Missing neutral-lut.png, please add the base file in the same directory as the executable!")
			dlg.exec()
			return

		#NB everything we do in this bit is in the RGB colourspac
		wrong = []
		right = []
		degreesA = 1  # initial polynomial fitting degrees
		degreesB = 2  # improved 2nd-step polynomial fitting degrees
		poly = {}
		colors = ["Red", "Green", "Blue"]

		self.renderCrop() # we need a fresh copy of the cropped image (not covered in the dots!)
		sourceImg = cv.cvtColor(self.imgCropped, cv.COLOR_BGR2RGB)

		# -------- sample the points --------
		for x, y, col in self.colourPoints:
			# create a mask to the local colour point
			mask = np.zeros(sourceImg.shape[:2], dtype=np.uint8)
			cv.rectangle(mask, (x - self.SAMPLE_REGION, y - self.SAMPLE_REGION), (x + self.SAMPLE_REGION, y + self.SAMPLE_REGION), 255, -1)
			# mask the cropped image to just that colour
			measured = cv.mean(sourceImg, mask=mask)
			wrong.append(np.array(measured[:3], dtype=np.uint8))
			right.append(np.array(col, dtype=np.uint8))
			print("Measured: ", len(wrong), "(", measured, ")")

		# -------- first polyfit (r,g,b individually) --------
		for rgb in range(3):
			www = np.ravel(wrong)[rgb::3]
			rrr = np.ravel(right)[rgb::3]

			poly[rgb] = np.polyfit(www, rrr, degreesA)
			p = np.poly1d(poly[rgb])

			delta = rrr - p(www)
			power = np.power(delta, 2)
			mean = np.mean(power)
			final = np.sqrt(mean)
			print(colors[rgb], "error:", final)
			if (final > 30):
				print("\t ^ High number implies result is not that optimized")

		# now generate more data from the previous polyfit
		grid = np.linspace(0, 255, 2)
		rainbow = np.zeros((len(grid) * len(grid) * len(grid), 3))
		counter = 0
		for r in grid:
			for g in grid:
				for b in grid:
					rainbow[counter, 0] = r
					rainbow[counter, 1] = g
					rainbow[counter, 2] = b
					counter += 1

		wrong0 = np.vstack((wrong, rainbow))

		for rgb in range(3):
			p = np.poly1d(poly[rgb])
			print(colors[rgb], "Pre-Process range:", int(np.min(rainbow[:, rgb])), "-", int(np.max(rainbow[:, rgb])))
			rainbow[:, rgb] = p(rainbow[:, rgb])
			print(colors[rgb], "Post-Process range:", int(np.min(rainbow[:, rgb])), "-", int(np.max(rainbow[:, rgb])))
			print()

		right0 = np.vstack((right, rainbow))

		r1 = np.ravel(wrong0)[0::3].astype(np.float32, copy=False)  # wrong
		g1 = np.ravel(wrong0)[1::3].astype(np.float32, copy=False)
		b1 = np.ravel(wrong0)[2::3].astype(np.float32, copy=False)

		r0 = np.ravel(right0)[0::3].astype(np.float32, copy=False)  # right
		g0 = np.ravel(right0)[1::3].astype(np.float32, copy=False)
		b0 = np.ravel(right0)[2::3].astype(np.float32, copy=False)

		rgb = np.array([r1, g1, b1])
		coes = {}

		# -------- second polyfit (r,g,b affect each other slightly) --------
		def polyfit3d(rgb, pp, x0):
			degrees = [(i, j, k) for i in range(pp) for j in range(pp) for k in range(pp)]  # list of monomials x**i * y**j to use
			matrix = np.stack([np.prod(rgb.T ** d, axis=1) for d in degrees], axis=-1)  # stack monomials like columns
			coeff = np.linalg.lstsq(matrix, x0)[0]  # lstsq returns some additional info we ignore
			# print("Coefficients", coeff)    # in the same order as the monomials listed in "degrees"
			fit = np.dot(matrix, coeff)
			# print(np.sqrt(np.mean((x0-fit)**2)))  ## error
			return coeff

		## PREDICT / SOLVE the function for our input data (getting our target data)
		def poly3d(rgb, coeff, pp):
			degrees = [(i, j, k) for i in range(pp) for j in range(pp) for k in range(pp)]  # list of monomials x**i * y**j to use
			matrix = np.stack([np.prod(rgb.T ** d, axis=1) for d in degrees], axis=-1)  # stack monomials like columns
			fit = np.dot(matrix, coeff)
			return fit

		## Generate and Save the functions; one function for each color type
		coes[0] = polyfit3d(rgb, degreesB, r0)
		coes[1] = polyfit3d(rgb, degreesB, g0)
		coes[2] = polyfit3d(rgb, degreesB, b0)

		# -------- now apply the LUT --------

		def applyLUT(image):
			# assert (image.dtype == "float32")
			for rgb in range(3):
				p = np.poly1d(poly[rgb])
				print("PRE range:", np.min(image[:, :, rgb]), "-", np.max(image[:, :, rgb]))
				image[:, :, rgb] = p(image[:, :, rgb])
				print("POST range:", np.min(image[:, :, rgb]), "-", np.max(image[:, :, rgb]))
			return image

		def applyLUT2(image, pp):
			sss = np.shape(image[:, :, 0])
			rgb = image.reshape(-1, 3).T
			Zr = poly3d(rgb, coes[0], pp).reshape(sss)
			Zg = poly3d(rgb, coes[1], pp).reshape(sss)
			Zb = poly3d(rgb, coes[2], pp).reshape(sss)
			image[:, :, 0] = Zr
			image[:, :, 1] = Zg
			image[:, :, 2] = Zb
			return image

		imgOutput = np.zeros_like(sourceImg, dtype=np.float32)
		imgOutput[:] = sourceImg[:]

		# imgOutput = applyLUT(imgOutput) + mb  ## You can disable applyLUT2 and enable this function instead if you are having problems
		imgOutput = applyLUT2(imgOutput, degreesB)

		print(np.min(imgOutput), np.max(imgOutput))
		if ((np.min(imgOutput) > 25) & (np.max(imgOutput) < 230)):
			print("Consider shooting in LOG or reducing contrast on your camera when recording")
		elif (np.min(imgOutput) > 25):
			print("Consider capturing with reduced exposure to capture more shadow detail")
		elif (np.max(imgOutput) < 230):
			print("Consider capturing with increase exposure to capture more highlight detail")
		elif ((np.min(imgOutput) < -300) or (np.max(sourceImg) > 500)):
			print("Consider shooting with more contrast or less dynamic range to get less colour banding")

		imgFloatified = cv.cvtColor(self.imgOrig, cv.COLOR_BGR2RGB).astype(np.float32,copy=False)
		imgFloatified = applyLUT2(imgFloatified, degreesB)
		imgFloatified[np.where(imgFloatified>255)] = 255
		imgFloatified[np.where(imgFloatified < 0)] = 0
		self.imgOutput = cv.cvtColor(imgFloatified.astype(np.uint8, copy=False), cv.COLOR_RGB2BGR)

		lutimg = applyLUT2(lutimg, degreesB)
		lutimg[np.where(lutimg>255)] = 255
		lutimg[np.where(lutimg<0)] = 0
		self.lutimg = cv.cvtColor(lutimg.astype(np.uint8, copy=False), cv.COLOR_RGB2BGR)
		self.coes = coes

		self.buttonDoCalibration.setStyleSheet("")
		self.updateUI()

	def computeToneCurve(self, src):
		# I can probably totally replicate the lightroom slider set, but more effort
		# (to do this, generate a pair (highest and lowest) curves (cubic for most) for each adjustment range, limit their outputs and then average all curves at all points)

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


		knots = np.array([[0, 0],
					   [variance * shadows / (-10*gradient) + shadow_midpoint + variance, variance * shadows / (10) + shadow_midpoint + variance],
					   [variance * blacks / (5*gradient) + blacks_midpoint + variance, variance * blacks / (5) + blacks_midpoint + variance],
					   [variance * whites / (-5*gradient) + whites_midpoint + variance, variance * whites / (5) + whites_midpoint + variance],
					   [variance * highlights / (10*gradient) + highlights_midpoint + variance, variance * highlights / (10) + highlights_midpoint + variance],
					   [255, 255]])


		p = scipy_bspline(knots, n=256, degree=2, periodic=False)
		xout, yout = p.T

		yout[np.where(yout < 0)] = 0
		yout[np.where(yout > 255)] = 255

		# now we want to fit to our rigid 0-255 int scale
		f = scipy.interpolate.splrep(xout, yout)  # make a spline representation
		xout = np.arange(0, 256)
		yout = scipy.interpolate.splev(xout, f).astype(np.uint8)

		return cv.LUT(src, yout)

	def gammaCorrection(self, src):
		# generate gamma LUT
		gammaTable = []
		highlights = self.sliderGammaHighlights.sl.value() / 100  # 1.0 = no change, greater = lighter
		shadows = self.sliderGammaShadows.sl.value() / 100  # greater = darker
		gamma = self.sliderGamma.sl.value()/100

		# now some value correction and sensitivity adjustment
		if highlights <0.001:
			highlights = 0.001
		if shadows <0.001:
			shadows == 0.001
		if gamma <0.001:
			gamma == 0.001

		highlights = 1+0.1*(highlights-2)**3+0.1*(highlights-2)
		shadows = 1 + 0.1 * (shadows - 2) ** 3 + 0.1 * (shadows - 2)
		gamma = 1+(gamma-1)**3+0.2*(gamma-1)

		for i in range(128):
			value = 128 * (1 - shadows * (2 - 1 / shadows) ** gamma + shadows * (2 + (i - 256) / (128 * shadows)) ** gamma)
			if isinstance(value, complex) or value < 0:
				value = 0

			gammaTable.append(int(value))
		for i in range(128, 256):
			value = 128 * (1 + highlights * (2 - 1 / highlights) ** gamma - highlights * (2 - (i) / (128 * highlights)) ** gamma)
			if isinstance(value, complex) or value > 255:
				value = 255

			gammaTable.append(int(value))

		gammaTable = np.array(gammaTable, np.uint8)
		return cv.LUT(src, gammaTable)

	def closeEvent(self, event):
		event.accept()

if __name__=="__main__":
	app = QApplication(sys.argv)
	a = App()
	a.show()
	sys.exit(app.exec_())