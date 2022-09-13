import numpy as np

def CubeFile(title, coEff, domains=3, size=16, degreesPolynomial=2):
	# see https://web.archive.org/web/20201027210201/https://wwwimages2.adobe.com/content/dam/acom/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf
	# coEff is for my polynomial -> cube conversion
	# for domains=3, size is number of points *in each dimension* (i.e. size of 8 is 8 points per r step, then again per g, then again per b etc)

	fileTxt = ""

	fileTxt += f"TITLE \"{title}\"\n\n"
	if domains==1:
		fileTxt += f"LUT_1D_SIZE {size}\n\n"
	elif domains==3:
		fileTxt += f"LUT_3D_SIZE {size}\n\n"
	else:
		print(f"Invalid domains size ({domains}) for cube file!")
		return
	fileTxt += f"DOMAIN_MIN 0.0 0.0 0.0\n"
	fileTxt += f"DOMAIN_MAX 1.0 1.0 1.0\n\n"

	# generate "source image" which is the list of input indexes of the lookup table
	# r is fastest iteration, b slowest
	# assuming equally spaced points
	rgbSource = np.array([(r,g,b) for b in range(size) for g in range(size) for r in range(size)]).T
	# scale to 0-256
	rgbSource = rgbSource * (256 / (size-1))

	def poly3d(rgb, coeff, pp):
		degrees = [(i, j, k) for i in range(pp) for j in range(pp) for k in range(pp)]  # list of monomials x**i * y**j to use
		matrix = np.stack([np.prod(rgb.T ** d, axis=1) for d in degrees], axis=-1)  # stack monomials like columns
		fit = np.dot(matrix, coeff)
		return fit

	Zr = poly3d(rgbSource, coEff[0], degreesPolynomial)/256
	Zg = poly3d(rgbSource, coEff[1], degreesPolynomial)/256
	Zb = poly3d(rgbSource, coEff[2], degreesPolynomial)/256

	for i in range(size**3):
		fileTxt += f"{Zr[i]:.2f} {Zg[i]:.2f} {Zb[i]:.2f}\n"

	return fileTxt

