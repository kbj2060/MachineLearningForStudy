'''
 * llewyn 17.03.02
 * 3 Layers Neural Network using Stanford-Tutorial's heart.csv data
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from keras.callbacks import ModelCheckpoint, TensorBoard

'''
 * Set up Paths and Numbers
'''
DATA_PATH = os.getcwd() + '/data/heart.csv'
FEATURE_NUM = 9
BATCH_SIZE = 50
HIDDEN1_SIZE = 32
HIDDEN2_SIZE = 13
OUTPUT = 1
MAX_RANGE = 100

'''
-------------------- Preprocessing -------------------
 * Read csv data and split to label and features
 * Change 'Present' in 'famhist' feature to 1.0 float
 --> I think there is another way to do that.
 * Make dataframe to list in order to insert to train_test_split function.
'''
features_str = []
feature_cols = [ i for i in range(9) ]
label_col = [ 9 ]
features_df = pd.read_csv(DATA_PATH, header=0, usecols = feature_cols)
label_df = pd.read_csv(DATA_PATH, header=0, usecols = label_col)

for string in features_df['famhist']:
    if string == 'Present':
        features_str.append(1.0)
    else:
        features_str.append(0.0)

del features_df['famhist']
features_df['famhist'] = features_str

features = features_df.values.tolist()
label = label_df.values.tolist()

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=40)

'''
----------------------- Modeling ----------------------
 * Keras is very comfortable. Just set up input_dimenstion at the first layer
  and don't need to do it after that.
 * All weights are initialized to 'uniform'
 * 1st Layer's Activation function is 'sigmoid'
 * 2nd Layer's Activation function is 'relu'
 * 3rd Layer's(Output Layer) Activation function is 'sigmoid'
 * There are two Dropout function
 * Loss function is 'binary_crossentropy'
 * Optimizer function is 'Adam'
'''
model = Sequential()
model.add(Dense(HIDDEN1_SIZE, input_dim=FEATURE_NUM, init='uniform', activation='sigmoid'))
model.add(Dropout(0.6))
model.add(Dense(HIDDEN2_SIZE, init='uniform', activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(OUTPUT, init='uniform', activation = 'sigmoid'))

'''
 * tensorboard and checkpoints saver callbacks
 * Keras Tensorboard graph is not prettier than original Tensorflow graph, but much simple.
 * Checkpoint callback shows improved weights on the output console.
'''
checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X_train, y_train, nb_epoch=MAX_RANGE, batch_size=50, validation_data=(X_test, y_test), callbacks=[checkpointer, tensorboard])
