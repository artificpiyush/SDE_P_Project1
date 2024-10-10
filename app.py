# from flask import Flask, render_template,request,redirect,url_for
# import boto3
# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import load_model

# # model = load_model('CIFAR_piyush.h5')


# # Configuring AWS S3
# s3 = boto3.client('s3')
# model_bucket = 'my-s3-bucket-name'

# allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


# app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB


# model = tf.keras.models.load_model('path_to_the_model.h5')

# # class_names = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 
# #        'Category 5', 'Category 6', 'Category 7', 'Category 8', 
# #        'Category 9', 'Category 10']

# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# # Ensure uploads directory exists
# if not os.path.exists('uploads'):
#     os.makedirs('uploads')


# @app.route('/')
# def home():
#     return render_template('home.html')

# # Helper function to preprocess image
# def prepare_image(image_path):
#     img = cv2.imread(image_path)  # Read image using OpenCV
#     img = cv2.resize(img, (32, 32))  # Resize to the CIFAR input size
#     img = img / 255.0  # Normalize the image
#     img = np.expand_dims(img, axis=0)  # Expand dimensions to fit the model input (1, 32, 32, 3)
#     return img

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return "No File Selected"

#     file = request.files['file']

#     if file.filename == '':
#         return "No File Selected"

#     if file and allowed_file(file.filename):  # Only proceed if the file has a valid extension
#         try:
#             image_path = f"uploads/{file.filename}"
#             file.save(image_path)

#             # Load the image using OpenCV
#             img = cv2.imread(image_path)

#             # Upload image to S3
#             try:
#                 s3.upload_file(image_path, model_bucket, file.filename)
#             except Exception as e:
#                 return f"Error uploading to S3: {e}"

#             # Resize the image for prediction
#             img = cv2.resize(img, (224, 224))
#             img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

#             predictions = model.predict(img_array)[0]  # Predict and get probabilities
#             percentages = [round(p * 100, 2) for p in predictions]

#             # Cleanup: Remove uploaded image after processing
#             os.remove(image_path)

#             return redirect(url_for('result', percentages=percentages))
#         except Exception as e:
#             return f"An error occurred: {e}"
#     else:
#         return "Invalid file format. Please upload a valid image."


# # Route to display the classification result
# @app.route('/result')
# def result():
#     percentages = request.args.getlist('percentages', type=float)
#     class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#     return render_template('result.html', percentages=percentages, class_names=class_names)


# if __name__ == "__main__":
#   app.run(debug = True, port = 5000)




#  JUST FOR LOCAL SERVER NOT AWS



from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore


allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

app = Flask(__name__)
model = load_model('cifar.piyush.h5')  # Loading my model

# Ensure uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def home():
    return render_template('home.html')

# Helper function to preprocess image
def prepare_image(image_path):
    img = cv2.imread(image_path)  # Read image using OpenCV
    img = cv2.resize(img, (32, 32))  # Resize to the CIFAR input size
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Expand dimensions to fit the model input (1, 32, 32, 3)
    return img


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No File Selected"

    file = request.files['file']

    if file.filename == '':
        return "No File Selected"

    if file and allowed_file(file.filename):  # Only proceeds if the file has a valid extension
        try:
            image_path = f"uploads/{file.filename}"
            file.save(image_path)

            # Preprocessing
            img = prepare_image(image_path)

            # Getting Predictions
            predictions = model.predict(img)
            percentages = [round(p * 100, 2) for p in predictions[0]]  # Convert to percentages
            
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            os.remove(image_path)  

            # Redirect to result page with probabilities and class names
            return redirect(url_for('result', percentages=percentages, class_names=class_names))
        except Exception as e:
            return f"An error occurred: {e}"
    else:
        return "Invalid file format. Please upload a valid image."

# Route to display the classification result
@app.route('/result')
def result():
    percentages = request.args.getlist('percentages', type=float)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return render_template('result.html', percentages=percentages, class_names=class_names)

if __name__ == "__main__":
  app.run(debug = True, port = 5000)