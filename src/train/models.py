import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class CNNModel:
    def __init__(self, num_channels=3, num_classes=2) -> None:
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.model = self.create()

    def create(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32,
                    (3, 3),
                    padding="same",
                    activation="relu",
                    input_shape=(50, 50, self.num_channels),
                ),
                tf.keras.layers.MaxPooling2D(strides=2),
                tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                tf.keras.layers.MaxPooling2D((3, 3), strides=2),
                tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                tf.keras.layers.MaxPooling2D((3, 3), strides=2),
                tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                tf.keras.layers.MaxPooling2D((3, 3), strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(1, activation="softmax"),
            ]
        )

        print("------------------------------------")
        print("----- Details about current model:")
        print(model.summary())
        print("------------------------------------")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(self, train_it, val_it, epochs, batch_size):
        train_it_steps = train_it.samples // batch_size
        validation_steps = val_it.samples // batch_size
        history = self.model.fit_generator(
            train_it,
            steps_per_epoch=train_it_steps,
            validation_data=val_it,
            validation_steps=validation_steps,
            epochs=epochs,
        )
        self.history = history
        return history.history

    def save(self, path):
        self.model.save(path)

    def save_history_visualization(self):
        if not self.history:
            return
        # accuracy
        plt.plot(self.history.history["accuracy"])
        plt.plot(self.history.history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.savefig("results_xray/model_acc.png")
        plt.close()

        # loss
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.savefig("results_xray/model_loss.png")
        plt.close()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def predict(self, test_images_paths):
        test_images = []
        for img_path in test_images_paths:
            test_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            test_img_size = cv2.resize(
                test_img, (50, 50), interpolation=cv2.INTER_LINEAR
            )
            test_images.append(test_img_size)
        test_images = np.array(test_images)
        pred_list = self.model.predict(test_images)
        res = [i.argmax() for i in pred_list]
        return res

    def predict_files(self, files):
        test_images = []
        for file in files:
            np_arr = np.fromstring(file.file.read(), np.uint8)
            img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            test_img_size = cv2.resize(img_np, (50, 50), interpolation=cv2.INTER_LINEAR)
            test_images.append(test_img_size)
        test_images = np.array(test_images)
        pred_list = self.model.predict(test_images)
        res = [i.argmax() for i in pred_list]
        return res

    def evaluate(self, test_it):
        loss = self.model.evaluate_generator(test_it, steps=24)
        return loss
