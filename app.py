from flask import Flask, request, redirect, url_for, flash, jsonify
import pickle
import json
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import ocrspace
from googletrans import Translator
import requests



app = Flask(__name__)
with open('tokenizer.pickle', 'rb') as handle:
	tokenizer = pickle.load(handle)

model = keras.models.load_model('model.h5')

@app.route('/', methods=['GET'])
def makecalc():
	#data = request.get_json()
	translator = Translator()
	#img_url=data["image"]
	img_url=str(request.args['image'])
	api = ocrspace.API('415485422988957', ocrspace.Language.English)
	nudepost = requests.post(
		"https://api.deepai.org/api/nsfw-detector",
		files={
			'image': img_url,
			},
		headers={'api-key': '9db2c733-b156-4efd-9dda-9f84901562ea'}
	)
	nuderesponse=nudepost.json()
	if(nuderesponse['output']['nsfw_score']>=0.5):
		return jsonify("OFFENSIVE")
	extractedInformation=api.ocr_url(img_url)
	extractedInformation=extractedInformation.replace('\r', '').replace('\n',' ')
	extractedInformation=translator.translate(extractedInformation).text
	sequences=tokenizer.texts_to_sequences(extractedInformation)
	sequence=pad_sequences(sequences,maxlen=66,padding="pre")
	prediction =model.predict(sequence)
	if(prediction[0][0]>=0.5):
		return jsonify("OFFENSIVE")
	else:
		return jsonify("NORMAL")
	app.run()
