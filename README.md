# Vietnamese Plate License Recognition 

This model is implemented with Haar Cascade and CNN to detect and recognise one-line plate number in car.

## Installation

This project is written by Google Colab.
Using library: OpenCV, Keras and Scikit-Learn

## Train and Validation Data
Include 6 similar fonts to real plate (Cause font for plate number is national secret, can not be public). 31 class of characters (digits from 0 to 9, 21 letters ABCDEFGHKLMNPRSTUVXYZ)
After that, generate them to 744 images by:
```python
train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1) 
```

## Main tasks:

**Step 1:**  Detect location of plate number by algorithm Haar Cascade

Firstly, download CascadeClassifier file for Vietnamese Plate License [here](https://drive.google.com/file/d/1kcg_3WVxyei4BdrViPV7SFyOSgP_AxRo/view?usp=share_link).
Loads the data required for detecting the license plates from cascade classifier.
```python
plate_cascade = cv2.CascadeClassifier('/content/drive/MyDrive/ImageProcessing/vn_license_plate.xml') 

#detects number plates and returns the coordinates and dimensions of detected license plate's contours.
plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.2, minNeighbors = 7)
```

**Step 2:** Match contours to license plate or character template

Using function *cv2.findContours* to draw contours for image
Choose 12 biggest contours then find possible character segmentation by conditions:

> Width > lower_width
> Width < upper_width
> Height > lower_height
> Height < upper_height

**Step 3:** Using CNN to classify characters

```python
model = Sequential()
model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=[custom_f1score])
```

**Step 4:** Predicting the output

Return class of character to get index in dictionary
```python
for i, ch in enumerate(char): #iterating over the characters
        ... #preparing image for the model
        y_ = model.predict(img, verbose=0) #predicting the class
        character = dic[y_.argmax()] 
```

# Result 
Accuracy = 94,62 %

Validation accuracy = 99,48 %

![alt text](https://github.com/khanhnd0908/VNPlateNumber/blob/main/Picture2.png)
