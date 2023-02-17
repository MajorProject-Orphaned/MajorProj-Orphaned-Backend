from flask import Flask, request, jsonify
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from deepface import DeepFace
import urllib
from keras_vggface.utils import preprocess_input
import sys


app = Flask(__name__)

try:
    filename = "/usr/local/lib/python3.10/dist-packages/keras_vggface/models.py"
    text = open(filename).read()
    open(filename, "w+").write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))
except:
    print("...")


detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')



def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

def extract_face(filename,detector=detector,required_size=(224, 224)):
	pixels = plt.imread(filename)
	# create the detector, using default weights
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array
    # resize pixels to the model size



#the resnet model represents the face in a 2048-dimension feature space.
def get_embedding(filename,model=model):
    # extract faces
    face = extract_face(filename)
    # convert into an array of samples
    sample = [asarray(face, 'float32')]
    # prepare the face for the model, e.g. center pixels
    sample = preprocess_input(sample, version=2)
    # perform prediction
    yhat = model.predict(sample)
    return yhat



def is_match(file1, file2, ID_embedding, subject_embedding, show_faces,thresh=0.4):
    # calculate distance between embeddings
    score = cosine(ID_embedding, subject_embedding)
    if show_faces==True:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(extract_face(file1))
        ax1.set_title('ID face')
        ax2.imshow(extract_face(file2))
        ax2.set_title('Subject face')

    is_matched = score <= thresh
    result_dic = {"is_matched": is_matched, "score": score}
    return result_dic


@app.route('/faceClassification', methods=['POST'])
def faceClassification():
    if request.method == 'POST':
        data = request.get_json()
        imgUrl1 = data['image1']
        imgUrl2 = data['image2']

        if imgUrl1 is None or imgUrl2 is None:
            return jsonify({"error": "Invalid arguments"}), 400
        
        print(imgUrl1, imgUrl2, "out")
        img1 = url_to_image(imgUrl1)
        img2 = url_to_image(imgUrl2)
        
        img1_embedding = get_embedding(img1)
        img2_embedding = get_embedding(img2)

        matchResult = is_match(img1, img2, img1_embedding, img2_embedding, show_faces=True)
        return jsonify(matchResult)


@app.route('/faceVerification', methods=['POST'])
def faceVerification():
    if request.method == 'POST':
        data = request.get_json()
        img1 = data['image1']
        img2 = data['image2']

        if img1 is None or img2 is None:
            print(img1, img2)
            return jsonify({"error": "Invalid arguments"}), 400
        
        print(img1, img2, "out")
        res = DeepFace.verify(img1, img2, model_name = "VGG-Face", detector_backend="mtcnn")
        return jsonify(res)



if __name__ == '__main__':
    app.run(debug=True)