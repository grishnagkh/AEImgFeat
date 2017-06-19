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


descr = genfromtxt('data/10k-extracted.txt', delimiter='\t');
selector = [x for x in range(descr.shape[1]) if x != 0] 
descr = descr[:, selector]

train_mask = np.ones(len(descr), dtype=bool)
train_mask[0::20] = False
test_mask = np.invert(train_mask)



input_dim =  descr.shape[1]
latent_dim = [512, 256, 128, 64, 32, 16, 8, 4]



##shallow 1024

input_descr=Input(shape=(input_dim,))
encoded = Dense(1024 , activation='relu'
	#, activity_regularizer=regularizers.activity_l1(10e-5)
	)(input_descr)
	
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoenc = Model(input=input_descr, output=decoded)

enc = Model(input=input_descr, output=encoded)

enc_in = Input(shape=(1024,))
dec_l = autoenc.layers[-1]
dec = Model(input=enc_in, output=dec_l(enc_in))

autoenc.compile(optimizer='adam', loss='mae')


history = autoenc.fit(descr[train_mask], descr[train_mask],
                nb_epoch=350,
                batch_size=950,
                shuffle=True,
                validation_data=(descr[test_mask], descr[test_mask])                
)
np.savetxt("models/train_loss_encoder_1024.log", history.history.get("loss"), delimiter="\t")
np.savetxt("models/val_loss_encoder_1024.log", history.history.get("val_loss"), delimiter="\t")
saveModel(enc, "models/encoder_1024.json", "models/encoder_1024.h5")





## train layerwise


#	## layer 1

input_descr0=Input(shape=(input_dim,))
encoded0 = Dense(latent_dim[0] , activation='relu'
	#, activity_regularizer=regularizers.activity_l1(10e-5)
	)(input_descr0)
	
decoded0 = Dense(input_dim, activation='sigmoid')(encoded0)
autoencoder0 = Model(input=input_descr0, output=decoded0)

encoder0 = Model(input=input_descr0, output=encoded0)

encoded_input0 = Input(shape=(latent_dim[0],))
decoder_layer0 = autoencoder0.layers[-1]
decoder0 = Model(input=encoded_input0, output=decoder_layer0(encoded_input0))

autoencoder0.compile(optimizer='adam', loss='mae')


history = autoencoder0.fit(descr[train_mask], descr[train_mask],
                nb_epoch=1000,
                batch_size=950,
                shuffle=True,
                validation_data=(descr[test_mask], descr[test_mask])                
)
np.savetxt("models/train_loss_encoder_512.log", history.history.get("loss"), delimiter="\t")
np.savetxt("models/val_loss_encoder_512.log", history.history.get("val_loss"), delimiter="\t")
saveModel(encoder0, "models/encoder_512.json", "models/encoder_512.h5")

#	# layer 2

input_descr1=Input(shape=(latent_dim[0],))
encoded1 = Dense(latent_dim[1] , activation='relu'
	#, activity_regularizer=regularizers.activity_l1(10e-5)
	)(input_descr1)
decoded1 = Dense(latent_dim[0], activation='sigmoid')(encoded1)
autoencoder1 = Model(input=input_descr1, output=decoded1)

encoder1 = Model(input=input_descr1, output=encoded1)

encoded_input1 = Input(shape=(latent_dim[1],))
decoder_layer1 = autoencoder1.layers[-1]
decoder1 = Model(input=encoded_input1, output=decoder_layer1(encoded_input1))

autoencoder1.compile(optimizer='adam', loss='mae')


train1 = encoder0.predict(descr[train_mask]);
test1 = encoder0.predict(descr[test_mask]);

autoencoder1.fit(train1, train1,
                nb_epoch=1000,
                batch_size=950,
                shuffle=True,
                validation_data=(test1,test1)    )

#	#finetune weights together...

autoencoder00 = Sequential()
autoencoder00.add(encoder0)
autoencoder00.add(encoder1)
autoencoder00.add(decoder1)
autoencoder00.add(decoder0)
autoencoder00.compile(optimizer='adam', loss='mae')

history = autoencoder00.fit(descr[train_mask], descr[train_mask],
                nb_epoch=350,
                batch_size=950,
                shuffle=True,
                validation_data=(descr[test_mask], descr[test_mask])
		)

encoder00 = Sequential()
encoder00.add(encoder0)
encoder00.add(encoder1)

saveModel(encoder00, "models/encoder_256.json", "models/encoder_256.h5")
np.savetxt("models/train_loss_encoder_256.log", history.history.get("loss"), delimiter="\t")
np.savetxt("models/val_loss_encoder_256.log", history.history.get("val_loss"), delimiter="\t")

#	layer 3

input_descr2=Input(shape=(latent_dim[1],))
encoded2 = Dense(latent_dim[2] , activation='relu')(input_descr2)
decoded2 = Dense(latent_dim[1], activation='sigmoid')(encoded2)
autoencoder2 = Model(input=input_descr2, output=decoded2)

encoder2 = Model(input=input_descr2, output=encoded2)

encoded_input2 = Input(shape=(latent_dim[2],))
decoder_layer2 = autoencoder2.layers[-1]
decoder2 = Model(input=encoded_input2, output=decoder_layer2(encoded_input2))

autoencoder2.compile(optimizer='adam', loss='mae')


train2 = encoder00.predict(descr[train_mask]);
test2 = encoder00.predict(descr[test_mask]);

autoencoder2.fit(train2, train2,
                nb_epoch=500,
                batch_size=950,
                shuffle=True,
                validation_data=(test2,test2)    )

#fine tuning 

autoencoder01 = Sequential()
autoencoder01.add(encoder0)
autoencoder01.add(encoder1)
autoencoder01.add(encoder2)
autoencoder01.add(decoder2)
autoencoder01.add(decoder1)
autoencoder01.add(decoder0)
autoencoder01.compile(optimizer='adam', loss='mae')

history = autoencoder01.fit(descr[train_mask], descr[train_mask],
                nb_epoch=350,#* 1,
                batch_size=950,
                shuffle=True,
                validation_data=(descr[test_mask], descr[test_mask])
		)

encoder01 = Sequential()
encoder01.add(encoder0)
encoder01.add(encoder1)
encoder01.add(encoder2)

np.savetxt("models/train_loss_encoder_128.log", history.history.get("loss"), delimiter="\t")
np.savetxt("models/val_loss_encoder_128.log", history.history.get("val_loss"), delimiter="\t")
saveModel(encoder01, "models/encoder_128.json", "models/encoder_128.h5")



#	layer 4
layer = 3

input_descr3=Input(shape=(latent_dim[layer-1],))
encoded3 = Dense(latent_dim[layer] , activation='relu')(input_descr3)
decoded3 = Dense(latent_dim[layer-1], activation='sigmoid')(encoded3)
autoencoder3 = Model(input=input_descr3, output=decoded3)

encoder3 = Model(input=input_descr3, output=encoded3)

encoded_input3 = Input(shape=(latent_dim[layer],))
decoder_layer3 = autoencoder3.layers[-1]
decoder3 = Model(input=encoded_input3, output=decoder_layer3(encoded_input3))

autoencoder3.compile(optimizer='adam', loss='mae')


train3 = encoder01.predict(descr[train_mask]);
test3 = encoder01.predict(descr[test_mask]);

autoencoder3.fit(train3, train3,
                nb_epoch=1000,
                batch_size=950,
                shuffle=True,
                validation_data=(test3,test3)    )

#fine tuning 

autoencoder02 = Sequential()
autoencoder02.add(encoder0)
autoencoder02.add(encoder1)
autoencoder02.add(encoder2)
autoencoder02.add(encoder3)
autoencoder02.add(decoder3)
autoencoder02.add(decoder2)
autoencoder02.add(decoder1)
autoencoder02.add(decoder0)
autoencoder02.compile(optimizer='adam', loss='mae')

history = autoencoder02.fit(descr[train_mask], descr[train_mask],
                nb_epoch=350,
                batch_size=950,
                shuffle=True,
                validation_data=(descr[test_mask], descr[test_mask])
		)

encoder02 = Sequential()
encoder02.add(encoder0)
encoder02.add(encoder1)
encoder02.add(encoder2)
encoder02.add(encoder3)

saveModel(encoder02, "models/encoder_64.json", "models/encoder_64.h5")
np.savetxt("models/train_loss_encoder_64.log", history.history.get("loss"), delimiter="\t")
np.savetxt("models/val_loss_encoder_64.log", history.history.get("val_loss"), delimiter="\t")

#	layer 5
layer = 4

input_descr4=Input(shape=(latent_dim[layer-1],))
encoded4 = Dense(latent_dim[layer] , activation='relu')(input_descr4)
decoded4 = Dense(latent_dim[layer-1], activation='sigmoid')(encoded4)
autoencoder4 = Model(input=input_descr4, output=decoded4)

encoder4 = Model(input=input_descr4, output=encoded4)

encoded_input4 = Input(shape=(latent_dim[layer],))
decoder_layer4 = autoencoder4.layers[-1]
decoder4 = Model(input=encoded_input4, output=decoder_layer4(encoded_input4))

autoencoder4.compile(optimizer='adam', loss='mae')


train4 = encoder02.predict(descr[train_mask]);
test4 = encoder02.predict(descr[test_mask]);

autoencoder4.fit(train4, train4,
                nb_epoch=1000,
                batch_size=950,
                shuffle=True,
                validation_data=(test4,test4)    )
                
                
#fine tuning 

autoencoder03 = Sequential()
autoencoder03.add(encoder0)
autoencoder03.add(encoder1)
autoencoder03.add(encoder2)
autoencoder03.add(encoder3)
autoencoder03.add(encoder4)
autoencoder03.add(decoder4)
autoencoder03.add(decoder3)
autoencoder03.add(decoder2)
autoencoder03.add(decoder1)
autoencoder03.add(decoder0)

autoencoder03.compile(optimizer='adam', loss='mae')

history = autoencoder03.fit(descr[train_mask], descr[train_mask],
                nb_epoch=1000,
                batch_size=950,
                shuffle=True,
                validation_data=(descr[test_mask], descr[test_mask])
		)

encoder03 = Sequential()
encoder03.add(encoder0)
encoder03.add(encoder1)
encoder03.add(encoder2)
encoder03.add(encoder3)
encoder03.add(encoder4)

np.savetxt("models/train_loss_encoder_32.log", history.history.get("loss"), delimiter="\t")
np.savetxt("models/val_loss_encoder_32.log", history.history.get("val_loss"), delimiter="\t")
saveModel(encoder03, "models/encoder_32.json", "models/encoder_32.h5")






#####

#	layer 6

layer = 5

input_descr5=Input(shape=(latent_dim[layer-1],))
encoded5 = Dense(latent_dim[layer] , activation='relu')(input_descr5)
decoded5 = Dense(latent_dim[layer-1], activation='sigmoid')(encoded5)
autoencoder5 = Model(input=input_descr5, output=decoded5)

encoder5 = Model(input=input_descr5, output=encoded5)

encoded_input5 = Input(shape=(latent_dim[layer],))
decoder_layer5 = autoencoder5.layers[-1]
decoder5 = Model(input=encoded_input5, output=decoder_layer5(encoded_input5))

autoencoder5.compile(optimizer='adam', loss='mae')


train5 = encoder03.predict(descr[train_mask]);
test5 = encoder03.predict(descr[test_mask]);

autoencoder5.fit(train5, train5,
                nb_epoch=2000,
                batch_size=950,
                shuffle=True,
                validation_data=(test5,test5)    )
                
                
#fine tuning 

autoencoder04 = Sequential()
autoencoder04.add(encoder0)
autoencoder04.add(encoder1)
autoencoder04.add(encoder2)
autoencoder04.add(encoder3)
autoencoder04.add(encoder4)
autoencoder04.add(encoder5)
autoencoder04.add(decoder5)
autoencoder04.add(decoder4)
autoencoder04.add(decoder3)
autoencoder04.add(decoder2)
autoencoder04.add(decoder1)
autoencoder04.add(decoder0)

autoencoder04.compile(optimizer='adam', loss='mae')

history = autoencoder04.fit(descr[train_mask], descr[train_mask],
                nb_epoch=2000,
                batch_size=950,
                shuffle=True,
                validation_data=(descr[test_mask], descr[test_mask])
		)

encoder04 = Sequential()
encoder04.add(encoder0)
encoder04.add(encoder1)
encoder04.add(encoder2)
encoder04.add(encoder3)
encoder04.add(encoder4)
encoder04.add(encoder5)

saveModel(encoder04, "models/encoder_16.json", "models/encoder_16.h5")
np.savetxt("models/loss_encoder_16.log", history.history.get("loss"), delimiter="\t")

#	layer 7


layer = 6

input_descr6=Input(shape=(latent_dim[layer-1],))
encoded6 = Dense(latent_dim[layer] , activation='relu')(input_descr6)
decoded6 = Dense(latent_dim[layer-1], activation='sigmoid')(encoded6)
autoencoder6 = Model(input=input_descr6, output=decoded6)

encoder6 = Model(input=input_descr6, output=encoded6)

encoded_input6 = Input(shape=(latent_dim[layer],))
decoder_layer6 = autoencoder6.layers[-1]
decoder6 = Model(input=encoded_input6, output=decoder_layer6(encoded_input6))

autoencoder6.compile(optimizer='adam', loss='mae')


train6 = encoder04.predict(descr[train_mask]);
test6 = encoder04.predict(descr[test_mask]);

autoencoder6.fit(train6, train6,
                nb_epoch=4000,
                batch_size=950,
                shuffle=True,
                validation_data=(test6,test6)    )
                
                
#fine tuning 

autoencoder05 = Sequential()
autoencoder05.add(encoder0)
autoencoder05.add(encoder1)
autoencoder05.add(encoder2)
autoencoder05.add(encoder3)
autoencoder05.add(encoder4)
autoencoder05.add(encoder5)
autoencoder05.add(encoder6)
autoencoder05.add(decoder6)
autoencoder05.add(decoder5)
autoencoder05.add(decoder4)
autoencoder05.add(decoder3)
autoencoder05.add(decoder2)
autoencoder05.add(decoder1)
autoencoder05.add(decoder0)

autoencoder05.compile(optimizer='adam', loss='mae')

history = autoencoder05.fit(descr[train_mask], descr[train_mask],
                nb_epoch=1000,
                batch_size=950,
                shuffle=True,
                validation_data=(descr[test_mask], descr[test_mask])
		)

encoder05 = Sequential()
encoder05.add(encoder0)
encoder05.add(encoder1)
encoder05.add(encoder2)
encoder05.add(encoder3)
encoder05.add(encoder4)
encoder05.add(encoder5)
encoder05.add(encoder6)

saveModel(encoder05, "models/encoder_8.json", "models/encoder_8.h5")
np.savetxt("models/loss_encoder_16.log", history.history.get("loss"), delimiter="\t")

#	layer 8


layer = 7

input_descr7=Input(shape=(latent_dim[layer-1],))
encoded7 = Dense(latent_dim[layer] , activation='relu')(input_descr7)
decoded7 = Dense(latent_dim[layer-1], activation='sigmoid')(encoded7)
autoencoder7 = Model(input=input_descr7, output=decoded7)

encoder7 = Model(input=input_descr7, output=encoded7)

encoded_input7 = Input(shape=(latent_dim[layer],))
decoder_layer7 = autoencoder7.layers[-1]
decoder7 = Model(input=encoded_input7, output=decoder_layer7(encoded_input7))

autoencoder7.compile(optimizer='adam', loss='mae')


train7 = encoder05.predict(descr[train_mask]);
test7 = encoder05.predict(descr[test_mask]);

autoencoder7.fit(train7, train7,
                nb_epoch=2000,
                batch_size=950,
                shuffle=True,
                validation_data=(test7,test7)    )
                
                
#fine tuning 

autoencoder06 = Sequential()
autoencoder06.add(encoder0)
autoencoder06.add(encoder1)
autoencoder06.add(encoder2)
autoencoder06.add(encoder3)
autoencoder06.add(encoder4)
autoencoder06.add(encoder5)
autoencoder06.add(encoder6)
autoencoder06.add(encoder7)
autoencoder06.add(decoder7)
autoencoder06.add(decoder6)
autoencoder06.add(decoder5)
autoencoder06.add(decoder4)
autoencoder06.add(decoder3)
autoencoder06.add(decoder2)
autoencoder06.add(decoder1)
autoencoder06.add(decoder0)

autoencoder06.compile(optimizer='adam', loss='mae')

history = autoencoder06.fit(descr[train_mask], descr[train_mask],
                nb_epoch=1000,
                batch_size=950,
                shuffle=True,
                validation_data=(descr[test_mask], descr[test_mask])
		)

encoder06 = Sequential()
encoder06.add(encoder0)
encoder06.add(encoder1)
encoder06.add(encoder2)
encoder06.add(encoder3)
encoder06.add(encoder4)
encoder06.add(encoder5)
encoder06.add(encoder6)
encoder06.add(encoder7)

saveModel(encoder06, "models/encoder_4.json", "models/encoder_4.h5")
np.savetxt("models/loss_encoder_16.log", history.history.get("loss"), delimiter="\t")



####### shallow models #####



####shallow 256
input_descr=Input(shape=(input_dim,))
encoded = Dense(256 , activation='relu'
	#, activity_regularizer=regularizers.activity_l1(10e-5)
	)(input_descr)
	
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoenc = Model(input=input_descr, output=decoded)

enc = Model(input=input_descr, output=encoded)

enc_in = Input(shape=(256,))
dec_l = autoenc.layers[-1]
dec = Model(input=enc_in, output=dec_l(enc_in))

autoenc.compile(optimizer='adam', loss='mae')


history = autoenc.fit(descr[train_mask], descr[train_mask],
                nb_epoch=200,
                batch_size=950,
                shuffle=True,
                validation_data=(descr[test_mask], descr[test_mask])                
)
np.savetxt("models/train_loss_encoder_256_shallow.log", history.history.get("loss"), delimiter="\t")
np.savetxt("models/val_loss_encoder_256.log_shallow", history.history.get("val_loss"), delimiter="\t")
saveModel(enc, "models/encoder_256_shallow.json", "models/encoder_256_shallow.h5")



####shallow 128
input_descr=Input(shape=(input_dim,))
encoded = Dense(128 , activation='relu'
	#, activity_regularizer=regularizers.activity_l1(10e-5)
	)(input_descr)
	
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoenc = Model(input=input_descr, output=decoded)

enc = Model(input=input_descr, output=encoded)

enc_in = Input(shape=(128,))
dec_l = autoenc.layers[-1]
dec = Model(input=enc_in, output=dec_l(enc_in))

autoenc.compile(optimizer='adam', loss='mae')


history = autoenc.fit(descr[train_mask], descr[train_mask],
                nb_epoch=200,
                batch_size=950,
                shuffle=True,
                validation_data=(descr[test_mask], descr[test_mask])                
)
np.savetxt("models/train_loss_encoder_128_shallow.log", history.history.get("loss"), delimiter="\t")
np.savetxt("models/val_loss_encoder_128_shallow.log", history.history.get("val_loss"), delimiter="\t")
saveModel(enc, "models/encoder_128_shallow.json", "models/encoder_128_shallow.h5")



####shallow 64
input_descr=Input(shape=(input_dim,))
encoded = Dense(64 , activation='relu'
	#, activity_regularizer=regularizers.activity_l1(10e-5)
	)(input_descr)
	
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoenc = Model(input=input_descr, output=decoded)

enc = Model(input=input_descr, output=encoded)

enc_in = Input(shape=(64,))
dec_l = autoenc.layers[-1]
dec = Model(input=enc_in, output=dec_l(enc_in))

autoenc.compile(optimizer='adam', loss='mae')


history = autoenc.fit(descr[train_mask], descr[train_mask],
                nb_epoch=200,
                batch_size=950,
                shuffle=True,
                validation_data=(descr[test_mask], descr[test_mask])                
)
np.savetxt("models/train_loss_encoder_64_shallow.log", history.history.get("loss"), delimiter="\t")
np.savetxt("models/val_loss_encoder_64_shallow.log", history.history.get("val_loss"), delimiter="\t")
saveModel(enc, "models/encoder_64_shallow.json", "models/encoder_64_shallow.h5")


####shallow 32
input_descr=Input(shape=(input_dim,))
encoded = Dense(32 , activation='relu'
	#, activity_regularizer=regularizers.activity_l1(10e-5)
	)(input_descr)
	
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoenc = Model(input=input_descr, output=decoded)

enc = Model(input=input_descr, output=encoded)

enc_in = Input(shape=(32,))
dec_l = autoenc.layers[-1]
dec = Model(input=enc_in, output=dec_l(enc_in))

autoenc.compile(optimizer='adam', loss='mae')


history = autoenc.fit(descr[train_mask], descr[train_mask],
                nb_epoch=200,
                batch_size=950,
                shuffle=True,
                validation_data=(descr[test_mask], descr[test_mask])                
)
np.savetxt("models/train_loss_encoder_32_shallow.log", history.history.get("loss"), delimiter="\t")
np.savetxt("models/val_loss_encoder_32_shallow.log", history.history.get("val_loss"), delimiter="\t")
saveModel(enc, "models/encoder_32_shallow.json", "models/encoder_32_shallow.h5")



