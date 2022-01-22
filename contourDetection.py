img = cv.imread("Colorchecker.jpg")
# img = cv.imread("cam1-1.png")
imggs = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
median = cv.medianBlur(img,5)
sharp = cv.filter2D(img,-1,np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
sharpgs = cv.cvtColor(sharp,cv.COLOR_BGR2GRAY)
ret,th = cv.threshold(sharpgs,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

kernel = np.ones((5,5),np.uint8)
opening = cv.morphologyEx(th,cv.MORPH_OPEN, kernel)

contours, hierarchy = cv.findContours(opening,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

# cv.drawContours(img,contours,-1,(0,255,0),3)
for c in contours:
	peri = cv.arcLength(c, True) # find perimeter length (true=enclosed)
	approx = cv.approxPolyDP(c,0.01*peri, True) # approximate the shape further within 4%
	if len(approx)==4:
		M = cv.moments(c)
		cX = int((M["m10"] / M["m00"]))
		cY = int((M["m01"] / M["m00"]))

		(x, y, w, h) = cv.boundingRect(approx)
		ar = w / float(h)
		if ar>=0.95 and ar<=1.05:
			# it's a square
			type="square"
		else:
			# it's a rectangle
			type="rect"

		cv.drawContours(img,[c], -1, (0,255,0), 2)
		cv.putText(img, type, (cX, cY), cv.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)

cv.imshow("wow",img)
cv.waitKey(0)
