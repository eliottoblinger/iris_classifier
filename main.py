import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

IRIS_NAMES = [
    "Setosa",
    "Versicolor",
    "Virginica"
]

def prepare_data(path):
    dataset = np.loadtxt(path, delimiter=',', skiprows=1)

    X = dataset[:, 0:4]
    y = dataset[:, 4]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(model = Sequential()):
    model.add(Dense(12, input_shape=(4,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=150, batch_size=10)

def test_model(model, X_test, y_test):
    predictions = (model.predict(X_test) > 0.7).astype(int)

    for i in range(len(predictions)):
        result = "Not found"
        if 1 in list(predictions[i]):
            result = IRIS_NAMES[list(predictions[i]).index(1)]

        print(f"Data : {X_test[i]} => {result} (Expected {IRIS_NAMES[int(y_test[i])]})")

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_data("iris_dataset.csv")

    model = build_model()

    train_model(model, X_train, y_train)

    test_model(model, X_test, y_test)

