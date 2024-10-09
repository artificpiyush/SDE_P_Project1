from flask import Flask, render_template,request,redirect,url_for
import boto3
import cv2
import numpy as np
import tensorflow as tf
# from tensorflow import keras

app = Flask(__name__)

model = tf.keras.models.load_model('path_to_the_model.h5')

class_names = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 
       'Category 5', 'Category 6', 'Category 7', 'Category 8', 
       'Category 9', 'Category 10']

s3 = boto3.client('s3')
model_bucket = 'my-s3-bucket-name'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No File Selected"

    file = request.files['file']

    if file.filename == '':
        return "No File Selected"

    if file:
        # As I'll not be saving the uploaded images
        image_path = f"uploads/{file.filename}"
        # file.save(image_path)

        # Loaded the image
        img = cv2.imread(image_path)

        s3.upload_file(image_path, model_bucket, file.filename)

        # Resized the uploaded image
        img = cv2.resize(img, (224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        predictions = model.predict(img_array)[0] # Array of 10 probabilities

        percentages = [round(p * 100,2) for p in predictions]

        return redirect(url_for('result', percentages=percentages))

# Route to display the classification result
@app.route('/result')
def result():
    percentages = request.args.getlist('percentages', type=float)
    return render_template('result.html', percentages=percentages, class_names=class_names)

# if __name__ == '__main__':
#     app.run(debug=True)



# if __name__ == "__main__":
#   app.run(host = '0.0.0.0', debug = True, port = 5000)

if __name__ == "__main__":
  app.run(host = '0.0.0.0', debug = True, port = 5000)