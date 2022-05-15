# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, render_template, request
import pickle 
import pandas as pd
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
app = Flask(__name__,template_folder='template')


z=SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        #access the data from form
        text1 = request.form["text1"]
        x1=z.encode(text1)
        
        text2 = request.form["text2"]
        x2=z.encode(text2)
        def cosine_similarity(list_1, list_2):
            cos_sim = dot(list_1, list_2) / (norm(list_1) * norm(list_2))
            return cos_sim
        output= cosine_similarity(x1,x2)
        return render_template("index.html", prediction_text='Your sentence similarity is {}'.format(output))
if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
   