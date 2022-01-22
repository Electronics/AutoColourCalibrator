import math
import cv2 as cv
import numpy as np
import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QEvent, QPoint


def distance(x1,y1,x2,y2):
	return math.sqrt((abs(x1)-abs(x2))**2+(abs(y1)-abs(y2))**2)

class App(QWidget):
	points = []
	dragingPointIndex = -1

	imgOrig = cv.imread("cam5-a.png")
	img = imgOrig.copy()
	imgCropped = np.zeros((400, 280, 3), np.uint8)
	imgCroppedCorners = np.float32([(0, 0), (0, imgCropped.shape[0]), (imgCropped.shape[1], imgCropped.shape[0]), (imgCropped.shape[1], 0)])

	DOTSIZE = 5  # size +/- the center point of a dot
	def __init__(self):
		super().__init__()
		self.setWindowTitle("Colour Calibration Camera Tool")
		# create the label that holds the image
		self.rawImage = QLabel(self)
		self.rawImage.resize(1400, 787)

		self.croppedImage = QLabel(self)
		self.croppedImage.resize(400,300)

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
		# set the vbox layout as the widgets layout
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
		self.buttonResetColourPoints = QPushButton("Reset Colour Points")
		self.buttonResetColourPoints.clicked.connect(lambda:self.initaliseColourPoints())
		vBoxLeft.addWidget(self.buttonResetColourPoints)
		vBoxLeft.addWidget(self.croppedImage)
		self.setLayout(hbox)

		self.setMouseTracking(True)
		self.installEventFilter(self)

		self.updateUI()
		self.initaliseColourPoints()

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
			self.colourPoints.append((x,y,(c[2],c[1],c[0]))) # RGB -> BGR
			i+=1
		self.updateUI()

	def overlayColourPoints(self):
		for x,y,colour in self.colourPoints:
			cv.circle(self.imgCropped, (x,y), self.DOTSIZE+2, colour, -1)
			cv.circle(self.imgCropped, (x,y), 1, (255,255,255), -1)

	def removeDot(self, x, y):
		if (x, y) not in self.points:
			raise IndexError("Point to remove is not in points list!")
		self.points.remove((x, y))

	def updateUI(self):
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
			transform = cv.getPerspectiveTransform(np.float32(self.points), self.imgCroppedCorners)
			self.imgCropped[:, :, :] = cv.warpPerspective(self.imgOrig, transform, (self.imgCropped.shape[1], self.imgCropped.shape[0]))

			# colour the status text
			self.labelPointsStatus.setStyleSheet("QLabel { color: green; }")
		else:
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

	def closeEvent(self, event):
		event.accept()

if __name__=="__main__":
	app = QApplication(sys.argv)
	a = App()
	a.show()
	sys.exit(app.exec_())