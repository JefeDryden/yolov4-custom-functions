import os
import cv2
import random
import numpy as np
import tensorflow as tf
import pytesseract
from core.utils import read_class_names
from core.config import cfg

anchorClasses = [4,5,6,7,9,10]
playerClasses = [0,1,2,3]

classDict = {
    0: {"Name": "TOR", "Colour": (91, 32, 0)},
    1: {"Name": "LEAFG", "Colour": (255,255,0)},
    2: {"Name": "MTL", "Colour": (45, 30, 175)},
    3: {"Name": "HABSG", "Colour": (255, 0, 255)},
    4: {"Name": "EXCEL", "FixX": 1230, "FixY": 414, "ConstantW": 400, "ConstantH": 850, "Angle": 31.468016, "Width": 0.107031}, #H 3002.674665500596 W 642.3408245746804
    5: {"Name": "ROGERS", "FixX": 220, "FixY": 544, "ConstantW": 420, "ConstantH": 630, "Angle": -21.072726619950554, "Width": 0.123438}, #286.7520440271298 W  553.5512653394878 H
    6: {"Name": "PS5", "FixX": 980, "FixY": 544, "ConstantW": 400 , "ConstantH": 650, "Angle": 17.264518700666944, "Width": 0.126563}, # H 881.5480968376409 W 573.5386922469687
    7: {"Name": "ADIDAS", "FixX": 42, "FixY": 404, "ConstantW": 360, "ConstantH": 610, "Angle": -32.08090010, "Width": 0.082031}, #Got -1137.1887963080944, used same as above , 833.1206812362283 H
    8: {"Name": "SNLOGO", "FixX": 0, "FixY": 0, "ConstantW": 0, "ConstantH": 0, "Angle": 0},
    9: {"Name": "CNTIRE", "FixX": 560, "FixY": 444, "ConstantW": 350, "ConstantH": 500, "Angle": -4.601412235566412, "Width": 0.145312}, # W 361.8850215854191 H  975.3118659647237
    10: {"Name": "CCOLA", "FixX": 710, "FixY": 444, "ConstantW": 350, "ConstantH": 500, "Angle": 0, "Width": 0.215625}, # W 513.4675 H  936.170

}

labelDict = {
    "TOR":0,
    "LEAFG":1,
    "MTL":2,
    "HABSG":3,
    "EXCEL":4,
    "ROGERS":5,
    "PS5":6,
    "ADIDAS":7,
    "SNLOGO":8,
    "CNTIRE":9,
    "CCOLA":10,
}

# function to count objects, can return total classes or count per class
def count_objects(data, by_class = False, allowed_classes = list(read_class_names(cfg.YOLO.CLASSES).values())):
    boxes, scores, classes, num_objects = data

    #create dictionary to hold count of objects
    counts = dict()

    # if by_class = True then count objects per class
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)

        # loop through total number of objects found
        for i in range(num_objects):
            # grab class index and convert into corresponding class name
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
            else:
                continue

    # else count total objects found
    else:
        counts['total object'] = num_objects
    
    return counts

# function for cropping each detection and saving as new image
def crop_objects(img, data, path, allowed_classes):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    #create dictionary to hold count of objects for image name
    counts = dict()
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            # crop detection from image (take an additional 5 pixels around all edges)
            cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
            # construct image name and join it to path for saving crop properly
            img_name = class_name + '_' + str(counts[class_name]) + '.png'
            img_path = os.path.join(path, img_name )
            # save image
            cv2.imwrite(img_path, cropped_img)
        else:
            continue

def draw_location(rink_img, data, pred_homo):
    
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    
    img = cv2.imread(r'/mydrive/images/RinkModelV2Test2.png',1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1280,542))
    
    xcoord_resize = (320/1280)
    ycoord_resize = (320/720)

    x_ratio = (1280/1100)
    y_ratio = (542/466)
    um = 0
    for i in range(num_objects):

        class_index = int(classes[i])
        class_name = class_names[class_index] 
        if labelDict[class_name] in playerClasses:
            xmin, ymin, xmax, ymax = boxes[i]
            x_location = (xmin+xmax)/2
            
            point = np.float32([[[x_location*xcoord_resize, ymax*ycoord_resize]]])
            try: 
                warped_point = cv2.perspectiveTransform(point,pred_homo)
                
                x_warped, y_warped = warped_point[0][0][0]*x_ratio, warped_point[0][0][1]*y_ratio
                
                #drawing
                
                rink_img = cv2.circle(img, (int(x_warped),int(y_warped)), radius=10, color=classDict[labelDict[class_name]]["Colour"], thickness=-1)
                
            except:
                um+=1

    rink_img = img/255.

    return rink_img

            
        
# function to run general Tesseract OCR on any detections 
def ocr(img, data):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    for i in range(num_objects):
        # get class name for detection
        class_index = int(classes[i])
        class_name = class_names[class_index]
        # separate coordinates from box
        xmin, ymin, xmax, ymax = boxes[i]
        # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
        box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
        # grayscale region within bounding box
        gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
        # threshold the image using Otsus method to preprocess for tesseract
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # perform a median blur to smooth image slightly
        blur = cv2.medianBlur(thresh, 3)
        # resize image to double the original size as tesseract does better with certain text size
        blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        # run tesseract and convert image text to string
        try:
            text = pytesseract.image_to_string(blur, config='--psm 11 --oem 3')
            print("Class: {}, Text Extracted: {}".format(class_name, text))
        except: 
            text = None
