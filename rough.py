from flask import Flask
app = Flask(__name__)
@app.route('/', methods = ['POST'])
def hello():
    return "heloooo beeee"

if __name__ == "__main__":
    app.run(port = 5050,debug = True)
