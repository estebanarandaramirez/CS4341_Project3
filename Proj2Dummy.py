import numpy as np
import csv
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

np.random.seed(1)

images = np.load('images.npy')
images = images.reshape(6500, 784)

labels = np.load('labels.npy')
encodedLabels = np.zeros((6500,10))
encodedLabels[np.arange(6500), labels]=1

pixels = 784
numbers = 10
trainingImages = np.zeros([5200, pixels])
testImages = np.zeros([1300, pixels])
trainingLabels = np.zeros([5200, numbers])
testLabels = np.zeros([1300, numbers])
for i in range(6500):
    if i < 5200:
        trainingImages[i] = images[i]
        trainingLabels[i] = encodedLabels[i]
    else:
        testImages[i-5200] = images[i]
        testLabels[i-5200] = encodedLabels[i]

model = Sequential()
model.add(Dense(50, input_dim=pixels, activation='relu'))
for i in range (9):
    model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

sgd = optimizers.SGD(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit(trainingImages, trainingLabels, validation_split=0.16, batch_size=512, epochs=500, verbose=2)

prediction = model.predict(testImages)
confMatrix = confusion_matrix(testLabels.argmax(axis=1), prediction.argmax(axis=1))
print(confMatrix)

evaluation = model.evaluate(testImages, testLabels, verbose=2)
print("Baseline Error for Test: %.2f%%" % (100-evaluation[1]*100))

with open('Task2Part1.csv', 'w', newline='') as writeFile:
   writer = csv.writer(writeFile)
   writer.writerow(
       ['epoch', 'acc', 'loss', 'val_acc', 'val_loss'])
   for epoch in history.epoch:
       writer.writerow([epoch, history.history['acc'][epoch], history.history['loss'][epoch], history.history['val_acc'][epoch], history.history['val_loss'][epoch]])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()