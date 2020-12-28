# way to upload images to
# way to save the images
from flask import Flask,render_template,request,url_for
import os,io
import cv2
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image

app =Flask(__name__)




BASE_DIR=os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER= os.path.join(BASE_DIR,'static/images')
new_loc=os.path.join(BASE_DIR,'static/p_images')
prototxt=os.path.join(BASE_DIR,'MobileNetSSD_deploy.prototxt.txt')
model=os.path.join(BASE_DIR,'MobileNetSSD_deploy.caffemodel')
confd=0.2

# defining the classification funtion
def get_class(image_path):
    
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	    "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD
    # implementation)
    img=cv2.imread(image_path)
    
    #image =Image.open(io.BytesIO(image_file))
    image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
	# predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    labels=[]

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	    confidence = detections[0, 0, i, 2]

	    # filter out weak detections by ensuring the `confidence` is
	    # greater than the minimum confidence
	    if confidence > confd:
		# extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
		    idx = int(detections[0, 0, i, 1])
		    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		    (startX, startY, endX, endY) = box.astype("int")
		# display the prediction
		    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		    print("[INFO] {}".format(label))
		    cv2.rectangle(image, (startX, startY), (endX, endY),
			    COLORS[idx], 2)
		    y = startY - 15 if startY - 15 > 15 else startY + 15
		    cv2.putText(image, label, (startX, y),
			    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
		    labels.append(label)
    print(labels)
    
    return(image,labels)



@app.route("/", methods=["GET","POST"])
def upload_predict():
    image_location=''
    if request.method == 'POST':
        image_file=request.files["image"]
    
        if image_file:
            image_location=os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            img=Image.open(image_location)
            resized_image=img.resize((300,300))
            os.remove(image_location)
            resized_image.save(image_location)

            c_image,labels=get_class(image_location)
            d_image=Image.fromarray(c_image, "RGB")
            loc=os.path.join(new_loc,image_file.filename)
            d_image.save(loc)
            text=''
        
            for i in labels:
	            text=text+ i[0:-8]+' + '
            text=text[0:-3]
                
            
            return render_template("index.html",prediction=1,image_name=image_file.filename,labels=labels)

    return render_template("index.html", prediction = 0,image_loc=None)

#if __name__ == "__main__":

app.run()
