import cv2 as cv
import numpy as np
import json
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#https://stackoverflow.com/users/12094894/simon-crane
#https://stackoverflow.com/users/4323741/burhan-rashid

#credits to above for this numpy encoder
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv.Canny(image, lower, upper)
	# return the edged image
	return edged

img = cv.imread(r"img.jpeg")

# Convert to graycsale
imgframe_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Blur the image for better edge detection
imgframe_blur = cv.GaussianBlur(imgframe_gray, (3,3), 0) 
# Apply Canny Edge Algo 
img_frame_edged = auto_canny(imgframe_blur)

frame_contours,hierarchies = cv.findContours(img_frame_edged,cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # use CHAIN_APPROX_SIMPLE for more points

@app.route("/")
def index():
    return json.dumps(frame_contours,cls=NumpyEncoder)  

app.run()  


