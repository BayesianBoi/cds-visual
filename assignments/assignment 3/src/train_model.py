from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(train_generator, test_generator, categories):
    """
    Training a vgg16 model on the given training data. Been playing around with this over a couple of days and this setup was the highest performing
    """

    # loading the model with imageNen without the top classification layer
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # unfreezing the top layers of the model to allow for more fine tuning. somewhat arbitrary with the 8 layers but was best performing in accuracy 
    for layer in base_model.layers[-8:]:
        layer.trainable = True

    # using custom layers. been tinkering a lot with this
    x = base_model.output
    x = Flatten()(x) # flattens the base model output into 1d instead of 2D
    x = Dense(512, activation="relu")(x) # using 512 layers with relu activation
    x = Dropout(0.5)(x) # 50% regularization 
    predictions = Dense(len(categories), activation="softmax")(x) # 10 different possible outcomes using softmax 

    model = Model(inputs=base_model.input, outputs=predictions) # combining base model with the added layers

    model.compile(optimizer=Adam(learning_rate=0.00001), loss="categorical_crossentropy", metrics=["accuracy"]) # compiling the model with adam optimiser and a very low learning rate using 

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True) # to prevent overfitting; uses loss in validation to stop earlier
    model_checkpoint = ModelCheckpoint("out/best_model.keras", monitor="val_loss", save_best_only=True) # saves the best fitting model if early stop. Wanted to upload it to Github but it is sadly too big (even when compressed)

    # training the model
    history = model.fit(
        train_generator,
        epochs=50, # 50 seems to be a good balance
        validation_data=test_generator,
        callbacks=[early_stopping, model_checkpoint]
    )

    return model, history
