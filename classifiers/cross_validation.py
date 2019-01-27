import random
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import centroid_classifier as my_centroid
import knn_clasifier as my_knn


def pickDataClass(filename, class_ids):
    """
    :param filename: char_string specifing the data file to read. For example, 'ATNT_face_image.txt'
    :param class_ids: array that contains the classes to be pick. For example: (3, 5, 8, 9)
    :return:an multi-dimension array or a file, containing the data (both attribute vectors and class labels)
           of the selected classes
    """
    input_data = []
    file_name = open(filename, 'r')

    for i in filename:
        input_data.append(file_name.readline())  # removed rstrip('\n')

    with open(filename, 'r') as f:
        input_data = f.readlines()

    data = []
    for i in range(len(input_data)):
        data.append(input_data[i].split(','))

    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = int(data[i][j])

    # Transposing matrix
    images =[[row[i] for row in data] for i in range(len(data[0]))]

    # Get class labels and attribute values separately
    class_labels=[]
    attributes = []
    for i in range(len(images)):
        if images[i][0] in class_ids:
            class_labels.append(images[i][0])
            attributes.insert(i,images[i][1:])

    # Create 2d list with only the data of classes required with class label
    images = [img for img in images if img[0] in class_ids]

    return images


def get_accuracy(predicted_classes, test_label):
    """
    :param predicted_classes: list of integer list representing training data instances
    :param test_label: list of integer list representing test data instances
    :return: accuracy percentage of the test data
    """
    correct_prediction = 0

    for i in range(len(test_label)):
        if test_label[i] == predicted_classes[i]:
            correct_prediction+=1

    accuracy = (correct_prediction/len(test_label))*100

    return accuracy


def letter_2_digit_convert(str):
    """
    :param str: string representing classes
    :return: corresponding classes in numbers
    """
    classes =[ord(i)-ord('A')+1 for i in str.upper()]
    return classes


def cross_validation(all_images, fold, classifier='knn'):
    all_accuracy = []
    print("{} classifier has been selected".format(classifier))

    data_length = len(all_images)

    # generates 'data_length' numbers of randoms numbers between 0 to (data_length-1)
    random_numbers = random.sample(range(0, data_length), data_length)

    # shuffles the input images
    random_images =[]
    for r in random_numbers:
        random_images.append(all_images[r])

    images = random_images

    foldList = [math.floor(data_length / fold) for i in range(fold)]

    for i in range(data_length % fold):
        foldList[i] += 1

    start = 0
    end = foldList[0]

    for f in range(len(foldList)):
        trainX = []
        testX = []
        trainY = []
        testY = []

        test_data = images[start:end]
        train_data = images[0:start]

        for i in images[end:data_length]:
            train_data.append(i)

        for i in test_data:
            testY.append(i[0])
            testX.append(i)

        for i in train_data:
            trainY.append(i[0])
            trainX.append(i)

        start = start + foldList[f]
        if f == len(foldList)-1:
            end = foldList[f]
        else:
            end = end + foldList[f+1]
        trainX = np.array(trainX)
        trainY = np.array(trainY)
        testX = np.array(testX)
        testY = np.array(testY)

        if classifier == 'knn':
            prediction = my_knn.knn_classifier(trainX, testX, 5)
            accuracy = get_accuracy(prediction, testY)
            print("My KNN fold accuracy: {:.2f}".format(accuracy))

        if classifier == 'centroid':
            prediction = my_centroid.centroid_method(trainX, testX)
            accuracy = get_accuracy(prediction, testY)
            print("My Centroid fold accuracy: {:.2f}".format(accuracy))

        if classifier == 'svm':
            linearsvm = LinearSVC(random_state=0).fit(trainX, trainY)
            accuracy = linearsvm.score(testX, testY)*100
            print("SVM classifier accuracy: {:.3f}".format(accuracy))

        if classifier == 'linearRegression':
            regressor = LinearRegression()
            regressor.fit(trainX, trainY)
            y_pred = regressor.predict(testX)
            prediction = [round(i) for i in y_pred]
            accuracy = get_accuracy(prediction,testY)
            print("Linear Regression classifier accuracy: {:.3f}".format(accuracy))

        all_accuracy.append(accuracy)

    return sum(all_accuracy)/len(all_accuracy), all_accuracy


def show_plot(accuracy_lst):

    fold_number = range(len(accuracy_lst))

    #plt.scatter(fold_number, accuracy_lst, label="test accuracy")
    plt.plot(fold_number, accuracy_lst, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("split_number")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Selecting the classes based on individual's name and student ID
    name = "KNBR"  # (A) Your FirstName (KN) and Your FamilyName (BR)
    utaID = '5716'  # UTA ID - 5715
    input_classes = letter_2_digit_convert(name)

    for i in utaID:
        input_classes.append(int(i))
    print('Input Classes:', input_classes)

    # Activate below line for using ATNTFaceImages
    # images = pickDataClass('ATNTFaceImages400.txt',input_classes)
    images = pickDataClass('HandWrittenLetters.txt', input_classes)

    fold = 5
    classifiers = ['knn','centroid','svm','linearRegression']

    for classifier in classifiers:
        avg_accuracy, accuracy_list = cross_validation(images, fold, classifier=classifier)
        print('Overall Cross-Validation Accuracy: ', avg_accuracy)

    print(accuracy_list)