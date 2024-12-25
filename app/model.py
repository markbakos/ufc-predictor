from tabnanny import verbose

import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers , models, callbacks

from sklearn.metrics import classification_report, confusion_matrix

def create_model(input_dim):

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(16, activation='relu'),
        layers.Dropout(0.1),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=50, batch_size=32):

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val,y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    return history

def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"\nTest Accuracy: {accuracy:.4f}")

    y_pred = model.predict(x_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    print("\nClassification Reposrt:")
    print(classification_report(y_test, y_pred_binary))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))

if __name__ == "__main__":
    from main import load_and_preprocess_data, prepare_model_data

    fights_df = load_and_preprocess_data("../data/complete_ufc_data.csv")
    x_train, x_val, x_test, y_train, y_val, y_test, scaler, features = prepare_model_data(fights_df)

    model = create_model(input_dim=x_train.shape[1])
    history = train_model(model, x_train, y_train, x_val, y_val)

    evaluate_model(model, x_test, y_test)




