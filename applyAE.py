#example for applying the trained AEs on wang simplicity dataset

import numpy as np
from numpy import genfromtxt

from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import regularizers
from keras.models import model_from_json

def saveModel(model, json_filename, weights_filename):
	# serialize model to JSON
	_json = model.to_json()
	with open(json_filename, "w") as json_file:
	    json_file.write(_json)
	#
	model.save_weights(weights_filename)
	print("Saved model to disk")

def loadModel(json_filename, weights_filename):
	json_file = open(json_filename, 'r')
	_json = json_file.read()
	json_file.close()
	model = model_from_json(_json)
	# load weights into new model
	model.load_weights(weights_filename)
	print("Loaded model from disk")
	return model



descr = genfromtxt('data/wang-extracted.txt', delimiter='\t');
selector = [x for x in range(descr.shape[1]) if x != 0] 
descr = descr[:, selector]

import pandas as pd
df = pd.read_csv('data/wang-extracted.txt')

import csv
fnames = []
with open('data/wang-extracted.txt', 'rb') as csvfile:
	rdr = csv.reader(csvfile, delimiter='\t', quotechar='|')
	for row in rdr:
		 fnames = np.append(fnames, [ row[0] ])

fnames = fnames[1:1001]
f = open('res10k-wang/fnames', mode='w')
for i in range(1,len(fnames)):
	f.write(fnames[i].strip()+'\n')


m = loadModel("models/encoder_256_shallow.json", "models/encoder_256_shallow.h5")
res_256 = m.predict(descr)
np.savetxt('res10k-wang/res_256_shallow.csv', res_256, delimiter='\t')

m = loadModel("models/encoder_128_shallow.json", "models/encoder_128_shallow.h5")
res_128 = m.predict(descr)
np.savetxt('res10k-wang/res_128_shallow.csv', res_128, delimiter='\t')

m = loadModel("models/encoder_64_shallow.json", "models/encoder_64_shallow.h5")
res_64 = m.predict(descr)
np.savetxt('res10k-wang/res_64_shallow.csv', res_64, delimiter='\t')

m = loadModel("models/encoder_32_shallow.json", "models/encoder_32_shallow.h5")
res_32 = m.predict(descr)
np.savetxt('res10k-wang/res_32_shallow.csv', res_32, delimiter='\t')



m = loadModel("models/encoder_1024.json", "models/encoder_1024.h5")
res_1024 = m.predict(descr)
np.savetxt('res10k-wang/res_1024.csv', res_1024, delimiter='\t')



m = loadModel("models/encoder_512.json", "models/encoder_512.h5")
res_512 = m.predict(descr)
np.savetxt('res10k-wang/res_512.csv', res_512, delimiter='\t')

m = loadModel("models/encoder_256.json", "models/encoder_256.h5")
res_256 = m.predict(descr)
np.savetxt('res10k-wang/res_256.csv', res_256, delimiter='\t')

m = loadModel("models/encoder_128.json", "models/encoder_128.h5")
res_128 = m.predict(descr)
np.savetxt('res10k-wang/res_128.csv', res_128, delimiter='\t')

m = loadModel("models/encoder_64.json", "models/encoder_64.h5")
res_64 = m.predict(descr)
np.savetxt('res10k-wang/res_64.csv', res_64, delimiter='\t')

m = loadModel("models/encoder_32.json", "models/encoder_32.h5")
res_32 = m.predict(descr)
np.savetxt('res10k-wang/res_32.csv', res_32, delimiter='\t')



####

m = loadModel("models/encoder_16.json", "models/encoder_16.h5")
res_16 = m.predict(descr)
np.savetxt('res_16.csv', res_16, delimiter='\t')

m = loadModel("models/encoder_8.json", "models/encoder_8.h5")
res_8 = m.predict(descr)
np.savetxt('res_8.csv', res_8, delimiter='\t')

m = loadModel("models/encoder_4.json", "models/encoder_4.h5")
res_4 = m.predict(descr)
np.savetxt('res_4.csv', res_4, delimiter='\t')

