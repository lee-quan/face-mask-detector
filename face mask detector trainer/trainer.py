#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as ts
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import session_info


# In[2]:


session_info.show()


# In[3]:


INIT_LR = 1e-4  # Initial Learning Rate
EPOCHS = 15    # How many runs of trainings
BS = 64         # Batch Size of the training
DIRECTORY = r"dataset"
CATEGORIES = ["with_mask", "without_mask"]
data = []   #Image arrays are appended in it
labels = [] #Appends image labels

data_dir = pathlib.Path("dataset")
image_countTotal = len(list(data_dir.glob('*/*')))
image_countMask = len(list(data_dir.glob('with_mask/*')))
image_countWithoutMask = len(list(data_dir.glob('without_mask/*')))
print("Total = ",image_countTotal, ", with_mask = ", image_countMask, ", without_mask = ",image_countWithoutMask )


# In[4]:


print("[INFO] loading images")
for category in CATEGORIES:
    print(datetime.now()," - Loading "+category+"")
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)  #specific path of one img
        # LOAD IMAGE AND CONVERT SIZE
        image = load_img(img_path,target_size=(256,256))
        image = img_to_array(image) #KERAS: convert image to array
        image = preprocess_input(image) #MobileNetV2

        # Append Image to data list
        data.append(image)

        # Appends labels to label list
        labels.append(category)
    print(datetime.now()," - Loaded "+category+"")


# In[5]:


# Encoding the labels as 0 and 1
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)


# In[6]:


(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size = 0.20, stratify=labels, random_state=53)


# In[7]:


# Construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
    )


# In[8]:


# load the MobileNetV2 network, ensuring head fc layer sets are left off.
# imagenet is used as the predefined weights for images
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(256, 256, 3)))


# In[9]:


# construct the head of the model that will be placed on top of the base mmodel
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel) #non linear, for images
headModel = Dropout(0.3)(headModel) # Dropout rate, careful of overfitting
headModel = Dense(2, activation="softmax")(headModel) #number of categories


# In[10]:


# place head FC model on top of base model
# This is the ACTUAL model we will train
model = Model(inputs=baseModel.input, outputs=headModel)


# In[11]:


# loop over all layers in the base model and freeze them so they will not
# be updated during the first training
for layer in baseModel.layers:
    layer.trainable = False


# In[12]:


# Compile Model
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])


# In[13]:


# Train the head of the NN
print("[INFO] Training head!")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX,testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)


# In[14]:


# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the label with
# corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs, 
                            target_names=lb.classes_))


# In[15]:


# Plot training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

plt


# In[16]:


print("[INFO] Generating confusion matrix...")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools


# In[17]:


cm = confusion_matrix(y_true=testY.argmax(axis=1), y_pred = predIdxs)


# In[18]:


cm


# In[26]:


# Function to plot confusion matrix, do not import the plot_confusion_matrix from sklearn.metrics because 
# we do not have estimator here.
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation="nearest",cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalised Confusion Matrix")
    else:
        print("Confusion Matrix without Normalization")
        
    print(cm)
    
    thres = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment="center", color="white" if cm[i,j] > thres else "black")
        
    plt.tight_layout()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")


# In[27]:


plot_confusion_matrix(cm=cm, classes=CATEGORIES, title="Confusion Matrix" )
plt.savefig("confusion_matrix.png")


# In[21]:


#Easier way of plotting confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CATEGORIES)
disp.plot(cmap=plt.cm.Blues)


# In[24]:


# SERIALIZE model to the disk
print("[INFO] saving mask detector model...")
model.save("../facemask_detector.model", save_format="h5")


# In[ ]:




