import cv2 as cv
import numpy as np

def make_coordinates(image,line_paramters):
    """
    we need to place the best lines we got from average_slope_intercept
    but we can't do it without getting the parameters
    """
    slope, intercept = line_paramters
    y1 = image.shape[0]             #height of the image
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)   #from Equation
    x2 = int((y2 - intercept)/slope)  
    return np.array((x1,y1,x2,y2))

def average_slope_intercept(image,lines):
    """
    we need only one line to represent the left and the right lanes
    so we gonna iterate over all the lines points and extract both slope
    and intercept and then average them all to get the best line
    """
    left_fit = []      # Slope -ve
    right_fit = []     # Slope +ve

    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)

        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array((left_line,right_line))

def canny(lane_image):
    """
    Preprocessing the Video to decrease the Noise and
    to capture less details to decrease computations
    """
    gray = cv.cvtColor(lane_image,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),0)
    canny = cv.Canny(blur,50,150)
    return canny

def check_dimensions(Canny):
    """
    Show Image with axis in order to define which part is the
    region of interest we need to focus on
    """
    import matplotlib.pyplot as plt
    plt.imshow(Canny)
    plt.show()

def display_lines (image,lines):
    """
    Each line is a 2D array containing coordinates specify the line's
    parameters as well as the location of the lines with respect to
    the image space,ensuring that they placed in the correct position
    """
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    return line_image

def region_of_interest(image):
    height = image.shape[0]
    #The values depends on the camera position (X,Y)
    triangle = np.array([[(200,height),(1100,height),(550,250)]])
    # Mask is a Black Blank image of the same size as image
    mask = np.zeros_like(image)
    #Fill mask(blank) with triangle(region of interest)
    cv.fillPoly(mask,triangle,255)
    masked_image = cv.bitwise_and(image,mask)
    return masked_image

capture = cv.VideoCapture('test.mp4')
while capture:
    _,frame = capture.read()
    Canny = canny(frame)
    cropped_image = region_of_interest(Canny)
    # check_dimensions(Canny)
    # Numbers here came from experiment
    lines = cv.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    averaged_lines = average_slope_intercept(frame , lines)
    line_image = display_lines(frame,averaged_lines)
    combo_image = cv.addWeighted(frame,0.8,line_image,1,1)
    cv.imshow("result",combo_image)
    if cv.waitKey(1) and 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()