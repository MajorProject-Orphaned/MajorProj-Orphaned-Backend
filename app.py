from flask import Flask, request, jsonify
from deepface import DeepFace


app = Flask(__name__)


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