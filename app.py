
from flask import Flask, render_template
import model
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
   midi_path = model.run2()
   return midi_path

if __name__ == '__main__':
    app.run(debug=True)
