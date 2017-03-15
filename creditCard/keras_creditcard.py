'''
 * llewyn 17.03.02
 * 3 Layers Neural Network using creditcard.csv data in kaggle
'''
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import advanced_activations


'''
-------------------- Preprocessing -------------------
 * Read csv data and split to label and features
 * Make dataframe to list in order to insert to train_test_split function.
'''

'''
 * bring the csv file to the dataframe variable without header
'''
dataframe = pd.read_csv('creditcard.csv', header=0)
df_cp = dataframe.copy()

'''
 * classify label and features
 * make label to one_hot using to_categorical in keras
'''
label = df_cp['Class'].values.tolist()
del df_cp['Class']
features = df_cp.values.tolist()
categorical_labels = to_categorical(label, 2)

'''
 * split train and test data using scikit learn
'''
X_train, X_test, y_train, y_test = train_test_split(features, categorical_labels, test_size=0.2, random_state=40)


'''
----------------------- Modeling ----------------------
 * Keras is very comfortable. Just set up input_dimenstion at the first layer
  and don't need to do it after that.
 * All weights are initialized to 'uniform'
 * 1st Layer's Activation function is 'ELU'
 * 2nd Layer's Activation function is 'ELU'
 * 3rd Layer's(Output Layer) Activation function is 'softmax'
 * There are two Dropout function
 * Loss function is 'binary_crossentropy'
 * Optimizer function is 'Adam'
'''

'''
 * define numbers for deep learning
'''
FEATURE_NUM = 30
CLASSES = 2
HIDDEN1_SIZE = 100
HIDDEN2_SIZE = 50
MAX_RANGE = 1000

model = Sequential()

model.add(Dense(HIDDEN1_SIZE, input_dim=FEATURE_NUM, init='uniform'))

model.add(advanced_activations.ELU(alpha=1.0))
model.add(Dropout(0.6))
model.add(Dense(HIDDEN2_SIZE, init='uniform'))

model.add(advanced_activations.ELU(alpha=1.0))
model.add(Dropout(0.6))
model.add(Dense(CLASSES, init='uniform', activation='softmax'))

'''
 * tensorboard and checkpoints saver callbacks
 * Keras Tensorboard graph is not prettier than original Tensorflow graph, but much easier to use.
'''
checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5",
                               verbose=1,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)
model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          nb_epoch =MAX_RANGE,
          batch_size=1000,
          validation_data=(X_test, y_test),
          callbacks=[checkpointer, tensorboard])
