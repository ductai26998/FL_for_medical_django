import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (AveragePooling2D, Dense, Dropout, Flatten,
                                     Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

matplotlib.use("Agg")


class CNNModel:
    def __init__(self, num_channels=3, num_classes=2) -> None:
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.model = self.create()

    def create(self):
        baseModel = VGG16(
            weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=(224, 224, 3)),
        )
        # Insert dropout in VGG16
        layer_1 = baseModel.layers[-3]
        layer_2 = baseModel.layers[-2]
        layer_3 = baseModel.layers[-1]

        # Create the dropout layers
        dropout1 = Dropout(0.85)
        dropout2 = Dropout(0.85)

        # Reconnect the layers
        x = dropout1(layer_1.output)
        x = layer_2(x)
        x = dropout2(x)
        x = layer_3(x)

        headModel = x
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(512, activation="relu")(headModel)
        headModel = Dense(256, activation="relu")(headModel)
        headModel = Dense(64, activation="relu")(headModel)
        headModel = Dense(self.num_classes, activation="softmax")(headModel)

        model = Model(inputs=baseModel.input, outputs=headModel)

        print("------------------------------------")
        print("----- Details about current model:")
        print(model.summary())
        print("------------------------------------")

        opt = SGD(learning_rate=1e-4, momentum=0.9)
        model.compile(
            loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
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
                test_img, (224, 224), interpolation=cv2.INTER_LINEAR
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
            test_img_size = cv2.resize(
                img_np, (224, 224), interpolation=cv2.INTER_LINEAR
            )
            test_images.append(test_img_size)
        test_images = np.array(test_images)
        pred_list = self.model.predict(test_images)
        pred_list = self.model.predict(test_images)
        res = [i.argmax() for i in pred_list]
        print(res)
        return res

    def evaluate(self, test_it, batch_size):
        steps = test_it.samples // batch_size
        loss = self.model.evaluate_generator(test_it, steps=steps)
        return loss
