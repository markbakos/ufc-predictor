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

def plot_training_history(history):
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,4))

    ax1.plot(history.history['accuracy'], label='Training accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Training loss')
    ax2.plot(history.history['val_loss'], label='Validation loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('loss')
    ax2.legend()

    plt.tight_layout()
    plt.show()

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

    plot_training_history(history)

    evaluate_model(model, x_test, y_test)




