'''
 * llewyn 17.03.09
 * 3 Layers Neural Network using restaurant sales data
'''
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout
from keras.layers import advanced_activations
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

import preprocess

'''
 * Set up Numbers
'''
BATCH_SIZE = 10
FEATURE_NUM = 3
LABEL_NUM = 1
HIDDEN1_SIZE = 500
HIDDEN2_SIZE = 200
HIDDEN3_SIZE = 70
HIDDEN4_SIZE = 20
OUTPUT = 4
MAX_RANGE = 10000

'''
 * Get data from preprocess.py
 * The type of data is DataFrame
'''
pre = preprocess.preprocess_data()
dataframe = pre.get_data()

'''
----------------------------- Preprocessing -----------------------------
 * 'sales' will be label, 'vacation', 'temp', 'weekday' will be features
 * vacation : 1 , semester : 0
 * monday : 0 ~ sunday : 6
 * sales is divided by 0% ~ 25% : 0 , 25% ~ 50% : 1 , 50% ~ 75% : 2 , 75% ~ 100% : 3
 * Make dataframe to list in order to insert to 'train_test_split function'
 * In tensorflow, I have to use tf.one_hot but it is easy to use to_categorical in keras
'''
label_list = dataframe['sales'].values.tolist()
label = np.transpose([label_list])
categorical_labels = to_categorical(label, nb_classes=4)

vacation = dataframe['vacation'].values.tolist()
temp = dataframe['temp'].values.tolist()
weekday = dataframe['weekday'].values.tolist()
features = np.transpose([vacation, temp, weekday])

X_train, X_test, y_train, y_test = train_test_split(features, categorical_labels, test_size=0.2, random_state=40)

'''
----------------------- Modeling ----------------------
 * Keras is very comfortable. Just set up input_dimenstion at the first layer
  and don't need to do it after that.
 * All weights are initialized to 'uniform'
 * W_regularizer are in 2nd and 4th layer
 * 2nd Layer's Activation function is 'ELU'
 * 3rd Layer's Activation function is 'ELU'
 * 4th Layer's Activation function is 'ELU'
 * 5th Layer's Activation function is 'softmax'
 * There are 1 Dropout function between 2nd and 3rd layer
 * Loss function is 'categorical_crossentropy'
 * Optimizer function is 'Adam'
'''
model = Sequential()
model.add(Dense(HIDDEN1_SIZE, input_dim = FEATURE_NUM, init='uniform'))

model.add(Dense(HIDDEN2_SIZE, W_regularizer=l2(0.01), init='uniform'))
model.add(advanced_activations.ELU(alpha=1.0))

model.add(Dense(HIDDEN3_SIZE, init='uniform'))
model.add(advanced_activations.ELU(alpha=1.0))
model.add(Dropout(0.6))

model.add(Dense(HIDDEN4_SIZE, W_regularizer=l2(0.001), init='uniform'))
model.add(advanced_activations.ELU(alpha=1.0))

model.add(Dense(OUTPUT, init='uniform', activation='softmax'))

'''
 * tensorboard and checkpoints saver callbacks
 * Keras Tensorboard graph is not prettier than original Tensorflow graph, but much simple.
 * Checkpoint callback shows improved weights on the output console.
'''
checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

Adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])
model.fit(X_train, y_train, nb_epoch=MAX_RANGE, batch_size=100, validation_data=(X_test, y_test), callbacks=[checkpointer, tensorboard])

