# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score


# Defining dataset directory
baseDir = "dataset/data_rgb"
trainDir = baseDir

# Setting image size, epochs, and batch size
imgSize = (224, 224)
epochs = 5
batchSize = 32

# Creating training data generator with augmentation and MobileNetV2 preprocessing
trainDataGen = ImageDataGenerator(
    
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# Creating validation data generator with MobileNetV2 preprocessing and no augmentation
valDataGen = ImageDataGenerator(
   
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# Loading training data from directory
trainGenerator = trainDataGen.flow_from_directory(
    
    directory=trainDir,
    target_size=imgSize,
    batch_size=batchSize,
    class_mode="binary",
    shuffle=True,
    subset='training'
)

# Loading validation data from directory
valGenerator = valDataGen.flow_from_directory(
    
    directory=trainDir,
    target_size=imgSize,
    batch_size=batchSize,
    class_mode="binary",
    shuffle=False,
    subset='validation'
)

# Printing class indices mapping for reference
print("Class Mapping:", trainGenerator.class_indices)

# Loading MobileNetV2 base model without top layers, using pretrained imagenet weights
baseModel = MobileNetV2(input_shape=(*imgSize, 3), include_top=False, weights='imagenet')

# Freezing base model layers to train only new classification head initially
baseModel.trainable = False

# Creating custom classification head on top of base model
inputs = Input(shape=(*imgSize, 3))

x = baseModel(inputs, training=False)

x = GlobalAveragePooling2D()(x)

x = Dropout(0.4)(x)

outputs = Dense(1, activation='sigmoid')(x)

# Building full model
model = Model(inputs, outputs)

# Compiling model with Adam optimizer and binary crossentropy loss
model.compile(
    
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Printing model summary
model.summary()

# Setting callbacks for checkpoint saving, early stopping and learning rate reduction
checkpointCb = ModelCheckpoint(
    
    "best_mobilenetv2_model.keras",
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

earlyStopCb = EarlyStopping(
    
    monitor="val_accuracy",
    patience=3,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

lrScheduler = ReduceLROnPlateau(
    
    monitor="val_accuracy",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1,
    mode="max"
)

# Training the model with generators and callbacks
history = model.fit(
    
    trainGenerator,
    validation_data=valGenerator,
    epochs=epochs,
    batch_size=batchSize,
    callbacks=[checkpointCb, earlyStopCb, lrScheduler]
)

# Plotting training and validation accuracy over epochs
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training History")
plt.show()

# Predicting on validation set
valPred = model.predict(valGenerator)
valPredLabels = (valPred > 0.5).astype("int32")
valTrueLabels = valGenerator.classes

# Printing classification report for validation results
print("Classification Report:")
print(classification_report(valTrueLabels, valPredLabels))

# Printing precision, recall, and F1 score
print("Precision:", precision_score(valTrueLabels, valPredLabels))
print("Recall:", recall_score(valTrueLabels, valPredLabels))
print("F1 Score:", f1_score(valTrueLabels, valPredLabels))

# Plotting confusion matrix heatmap
cm = confusion_matrix(valTrueLabels, valPredLabels)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=trainGenerator.class_indices.keys(),
            yticklabels=trainGenerator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()