from models import resnet50_custom, load_data
import tensorflow as tf

def train():
    model = resnet50_custom()
    train_dataset, val_dataset, test_dataset = load_data()
    callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=6,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

    history = model.fit(train_dataset, 
                                epochs = 100, 
                                validation_data = val_dataset,
                                callbacks=[callback])

    loss, acc = model.evaluate(test_dataset)
    model.save('Resnet50_stack_f182.h5')
