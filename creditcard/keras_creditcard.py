import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import advanced_activations


dataframe = pd.read_csv('creditcard.csv', header=0)
df_cp = dataframe.copy()

label = df_cp['Class'].values.tolist()
del df_cp['Class']
features = df_cp.values.tolist()
categorical_labels = to_categorical(label, 2)

X_train, X_test, y_train, y_test = train_test_split(features, categorical_labels, test_size=0.2, random_state=40)

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
          epochs =MAX_RANGE,
          batch_size=1000,
          validation_data=(X_test, y_test),
          callbacks=[checkpointer, tensorboard])
