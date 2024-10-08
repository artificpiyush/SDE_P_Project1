from flask import Flask, render_template,request,redirect,url_for
import boto3
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('home.html')


if __name__ == "__main__":
  app.run(host = '0.0.0.0', debug = True, port = 5000)