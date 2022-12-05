import tensorflow as tf
import matplotlib.pyplot as plt


class CNNModel:
    def __init__(self) -> None:
        self.model = self.create()

    def create(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32,
                    (3, 3),
                    padding="same",
                    activation="relu",
                    input_shape=(50, 50, 3),
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
                tf.keras.layers.Dense(2, activation="softmax"),
            ]
        )

        # print(model.summary())
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(self, X_train, y_train, X_test, y_test):
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=75,
        )
        self.history = history
        return history.history

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