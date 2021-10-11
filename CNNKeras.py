import tensorflow as tf
import keras
from tensorflow.keras import layers, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
# import matplotlib.pyplot as plt
import DatasetLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

input_width = 16  # width of the image in pixels
input_height = 16  # height of the image in pixels
class_output = 4  # number of possible classifications for the problem


def prep_submissions(preds_array):
    preds_df = pd.DataFrame(preds_array)
    predicted_labels = preds_df.idxmax(axis=1)
    return predicted_labels

def cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3),
                            activation='relu',
                            padding='same',
                            input_shape=(input_width, input_height, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3),
                            activation='relu',
                            padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(class_output, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model_checkpoint = ModelCheckpoint('cf_cnn_1.h5', verbose=1, save_best_only=True)


n_folds = 5
EPOCHS = 5
BATCH_SIZE = 64
model_history = []

for fold in range(n_folds):
    print("Training on Fold: ", fold+1)

    train_data = DatasetLoader.get_data_train_complete(fold=fold)
    train_x = train_data[0].reshape(train_data[0].shape[0], input_width, input_height, 1)
    train_y = train_data[1]

    labels = np.array(prep_submissions(train_y).tolist())
    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))

    model = cnn_model()
    history = model.fit(train_x, train_y,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint])

    model_history.append(history)

    test_data = DatasetLoader.get_data_test(fold=fold)
    test_x = test_data[0].reshape(test_data[0].shape[0], input_width, input_height, 1)
    test_y = test_data[1]

    print(model.evaluate(test_x, test_y))
    test_preds = model.predict(test_x)

    categ_test_y = prep_submissions(test_y)
    categ_test_pred = prep_submissions(test_preds)

    print(classification_report(categ_test_y, categ_test_pred))
    conf_matx = confusion_matrix(categ_test_y, categ_test_pred)
    print(conf_matx)


# plt.title('Train Accuracy vs Val Accuracy')
# plt.plot(model_history[0].history['acc'], label='Train Accuracy Fold 1', color='black')
# plt.plot(model_history[0].history['val_acc'], label='Val Accuracy Fold 1', color='black', linestyle="dashdot")
# plt.plot(model_history[1].history['acc'], label='Train Accuracy Fold 2', color='red', )
# plt.plot(model_history[1].history['val_acc'], label='Val Accuracy Fold 2', color='red', linestyle="dashdot")
# plt.plot(model_history[2].history['acc'], label='Train Accuracy Fold 3', color='green', )
# plt.plot(model_history[2].history['val_acc'], label='Val Accuracy Fold 3', color='green', linestyle="dashdot")
# plt.legend()
# plt.show()



