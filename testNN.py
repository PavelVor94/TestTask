from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from imutils import paths
import numpy as np
import random
import cv2
import os


# изображения для проверки модели: 1-Пчела ; 2-Оса ; 3-Оса 4-Пчела
TEST_IMAGES = ['test.jpg' , 'test2.jpg' , 'test3.jpg' , 'test4.jpg']


data = []
labels = []

imagePaths = sorted(list(paths.list_images('./data')))
random.seed(42)
random.shuffle(imagePaths)

# цикл по изображениям
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 24)).flatten()
	data.append(image)
	label = imagePath.split(os.path.sep)[1]
	labels.append(label)
	print(len(data))

# переводим изображения и метки в машиночитаемый вид
data = np.array(data) / 255.0
labels = np.array(labels)

# переводим метки в числовой вид
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# разделяем данные на обучающий набор и тестовый
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25)

# создаем модель

model = svm.SVC(kernel='poly')
model.fit(trainX,trainY.ravel())




# проходим по проверочным изображениям и выводим их с надписями к какому классу относиться объект и с какой вероятностью
for test_img in TEST_IMAGES:
	image_test = cv2.imread(test_img)
	output = image_test.copy()
	image_test = cv2.resize(image_test, (32, 24)).flatten()
	image_test = image_test.astype("float") / 255.0
	image_test = image_test.reshape((1, image.shape[0]))

	preds = model.predict(image_test)
	print(preds)



	label = lb.classes_[preds]

	text = "{}".format(label)
	cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	cv2.imshow("Image", output)
	cv2.waitKey(0)



