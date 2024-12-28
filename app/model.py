from tabnanny import verbose
from pathlib import Path
import seaborn as sb
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers , models, callbacks, regularizers, optimizers
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

def create_model(input_dim):

    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    input_tensor = x
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    input_tensor = layers.Dense(64)(input_tensor)
    x = layers.Add()([x, input_tensor])
    x = layers.Activation('relu')(x)

    input_tensor = x
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    input_tensor = layers.Dense(32)(input_tensor)
    x = layers.Add()([x, input_tensor])
    x = layers.Activation('relu')(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=100, batch_size=64):

    def lr_schedule(epoch):
        initial_lr = 0.001
        drop = 0.5
        epochs_drop = 10.0
        lr = initial_lr * np.power(drop, np.floor((1+epoch)/epochs_drop))
        return lr

    callbacks_list = [
        callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=0.00001
        ),
        callbacks.LearningRateScheduler(lr_schedule)
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val,y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )

    return history

def save_model(model, scaler, features, model_dir="models", version="v1"):
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / f"ufc_model_{version}.keras"
    model.save(model_path)

    joblib.dump(scaler, model_dir / f"scaler_{version}.joblib")
    joblib.dump(features, model_dir / f"features_{version}.joblib")

    print(f"Model and components saved to {model_dir}")


def load_model(model_dir="models", version="v1"):
    script_dir = Path(__file__).parent
    model_dir = script_dir / model_dir

    model_path = model_dir / f"ufc_model_{version}.keras"
    model = keras.models.load_model(model_path)

    scaler = joblib.load(model_dir / f"scaler_{version}.joblib")
    features = joblib.load(model_dir / f"features_{version}.joblib")

    return model, scaler, features

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

    y_pred = (model.predict(x_test) > 0.5).astype(int)
    matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sb.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['True Negative', 'True Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
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

    fights_df = load_and_preprocess_data("data/complete_ufc_data.csv")
    x_train, x_val, x_test, y_train, y_val, y_test, scaler, features = prepare_model_data(fights_df)

    model = create_model(input_dim=x_train.shape[1])
    history = train_model(model, x_train, y_train, x_val, y_val)

    plot_training_history(history)

    save_model(model, scaler, features)
    evaluate_model(model, x_test, y_test)




