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
train = pd.read_csv('train.csv', header=0)
test = pd.read_csv('test.csv', header=0)
test_sample = pd.read_csv('sample_submission.csv', header=0)

del train['Id']
del test['Id']
del test_sample['Id']

tr_cp = train.copy()
te_cp = test.copy()

'''
 * classify label and features
 * make label to one_hot using to_categorical in keras
'''
y_train= tr_cp['SalePrice'].values.tolist()
del tr_cp['SalePrice']
X_train = tr_cp.values.tolist()

X_test = te_cp.values.tolist()
y_test = test_sample.values.tolist()


'''
----------------------- Modeling ----------------------
 * Keras is very comfortable. Just set up input_dimenstion at the first layer
  and don't need to do it after that.
 * All weights are initialized to 'uniform'
 * 1st Layer's Activation function is 'ELU'
 * 2nd Layer's Activation function is 'ELU'
 * 3rd Layer's(Output Layer) Activation function is 'softmax'
 * There are two Dropout function
 * Loss function is 'mean_squared_error'
 * Optimizer function is 'Adam'
'''

'''
 * define numbers for deep learning
'''
FEATURE_NUM = 36
CLASSES = 1
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
model.add(Dense(CLASSES, init='uniform', activation='relu'))

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
model.compile(loss='mean_squared_error',
              optimizer='Adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          nb_epoch =MAX_RANGE,
          batch_size=500,
          validation_data=(X_test, y_test),
          callbacks=[checkpointer, tensorboard])
