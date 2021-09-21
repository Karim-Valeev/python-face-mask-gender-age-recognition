import keras.metrics
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
from keras import backend as K
from sklearn.model_selection import train_test_split


def train_models():
    data = pd.read_csv(r'dataset/age_gender.csv')

    data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype="float32"))

    print('Total rows: {}'.format(len(data)))
    print('Total columns: {}'.format(len(data.columns)))

    # normalizing pixels data
    data['pixels'] = data['pixels'].apply(lambda x: x / 255)

    # calculating distributions
    age_dist = data['age'].value_counts()
    # ethnicity_dist = data['ethnicity'].value_counts()
    gender_dist = data['gender'].value_counts().rename(index={0: 'Male', 1: 'Female'})

    # def ditribution_plot(x, y, name):
    #     fig = go.Figure([
    #         go.Bar(x=x, y=y)
    #     ])
    #
    #     fig.update_layout(title_text=name)
    #     fig.show()

    # ditribution_plot(x=age_dist.index, y=age_dist.values, name='Age Distribution')

    def recall(y_true, y_pred):
        y_true = K.ones_like(y_true)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_m = true_positives / (all_positives + K.epsilon())
        return recall_m

    def precision(y_true, y_pred):
        y_true = K.ones_like(y_true)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_m = true_positives / (predicted_positives + K.epsilon())
        return precision_m

    def f1_score(y_true, y_pred):
        precision_m = precision(y_true, y_pred)
        recall_m = recall(y_true, y_pred)
        return 2 * ((precision_m * recall_m) / (precision_m + recall_m + K.epsilon()))

    X = np.array(data['pixels'].tolist())

    # Converting pixels from 1D to 3D
    X = X.reshape(X.shape[0], 48, 48, 1)

    y = data['gender']

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.22, random_state=37
    # )
    #
    # model = tf.keras.Sequential([
    #     L.InputLayer(input_shape=(48, 48, 1)),
    #     L.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    #     L.BatchNormalization(),
    #     L.MaxPooling2D((2, 2)),
    #     L.Conv2D(64, (3, 3), activation='relu'),
    #     L.MaxPooling2D((2, 2)),
    #     L.Flatten(),
    #     L.Dense(64, activation='relu'),
    #     L.Dropout(rate=0.5),
    #     L.Dense(1, activation='sigmoid')
    # ])
    #
    # model.compile(optimizer='sgd',
    #               loss=tf.keras.losses.BinaryCrossentropy(),
    #               metrics=['accuracy'])
    #
    #
    # ## Stop training when validation loss reach 0.2700
    # class myCallback(tf.keras.callbacks.Callback):
    #     def on_epoch_end(self, epoch, logs={}):
    #         if (logs.get('val_loss') < 0.2700):
    #             print("\nReached 0.2700 val_loss so cancelling training!")
    #             self.model.stop_training = True
    #
    #
    #
    # callback = myCallback()
    #
    # print(model.summary())
    #
    # history = model.fit(
    #     X_train, y_train, epochs=20, validation_split=0.1, batch_size=64, callbacks=[callback]
    # )
    #
    # loss, acc = model.evaluate(X_test, y_test, verbose=0)
    # print('Test loss: {}'.format(loss))
    # print('Test Accuracy: {}'.format(acc))
    #
    # model.save("gender_recognise.model", save_format="h5")

    y = data['age']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=37
    )

    model = tf.keras.Sequential([
        L.InputLayer(input_shape=(48, 48, 1)),
        L.Conv2D(128, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        L.MaxPooling2D((2, 2)),
        L.BatchNormalization(),
        L.Conv2D(64, (3, 3), activation='relu'),
        L.MaxPooling2D((2, 2)),
        L.BatchNormalization(),
        L.Conv2D(32, (3, 3), activation='relu'),
        L.MaxPooling2D((2, 2)),
        L.BatchNormalization(),
        L.Flatten(),
        L.Dense(256, activation='relu'),
        L.Dense(128, activation='relu'),
        L.Dense(1, activation='relu'),
    ])

    sgd = tf.keras.optimizers.SGD(momentum=0.9)

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae', precision, recall])

    # Stop training when validation loss reach 110
    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_loss') < 110):
                print("\nReached 110 val_loss so cancelling training!")
                self.model.stop_training = True

    callback = MyCallback()

    print(model.summary())

    # model.save()

    history = model.fit(
        X_train, y_train, epochs=20, validation_split=0.1, batch_size=64, callbacks=[callback]
    )

    mse, mae, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    print('Test Mean squared error: {}'.format(mse))
    print('Test Mean absolute error: {}'.format(mae))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))

    model.save("age_recognise.model", save_format="h5")


if __name__ == '__main__':
    train_models()

