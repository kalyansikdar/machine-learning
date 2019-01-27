import random
from itertools import groupby
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid


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
        input_data.append(file_name.readline()) #removed rstrip('\n')

    with open(filename, 'r') as f:
        input_data = f.readlines()

    data = []
    for i in range(len(input_data)):
        data.append(input_data[i].split(','))

    # Transposing matrix
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = int(data[i][j])

    images =[[row[i] for row in data] for i in range(len(data[0]))]

    # Get class labels and attribute values separately
    class_labels=[]
    attributes = []
    for i in range(len(images)):
        if images[i][0] in class_ids:
            class_labels.append(images[i][0])
            attributes.insert(i,images[i][1:])

    #Create 2d list with only the data of classes required with class label
    images = [img for img in images if img[0] in class_ids]

    return images


def splitData2TestTrain(data, number_per_class=None,  test_instances=None):
    """
    :param data: char_string specifing the data file to read. This can also be an array containing input data.
  number_per_class: number of data instances in each class (we assume every class has the same number of data instances)
  test_instances: the data instances in each class to be used as test data.
    :param number_per_class: number of data instances in each class (we assume every class has the same number of data instances)
    :param test_instances: the data instances in each class to be used as test data.
                  We assume that the remaining data instances in each class (after the test data instances are taken out)
                  will be training_instances
    :return: Training_attributeVector(trainX), Training_labels(trainY), Test_attributeVectors(testX), Test_labels(testY)
  The data should easily feed into a classifier.
    """
    trainX=[]
    testX=[]
    trainY=[]
    testY=[]
    testX_file=[]
    trainX_file = []

    img ={}

    # Creates dictionary with each class as keys and the array of images of that class as values
    # segregates the data as per classes (class, [[data][data][data]])
    for key,group in groupby(data, lambda x:x[0]):
        img[key]=list(group)

    # print the classes and amount of images in that class
    for k,v in img.items():
        print(k, len(v))

    # randomly selects "len(v) / 2" number of images and saves into testX and rest into trainX
    # and corresponding labels into testY and trainY
    if not test_instances:
        random_number = random.sample(range(0, len(v) - 1), int(len(v) / 2))
        print('random', random_number)
        for k, v in img.items():
            for i in range(len(v)):
                if i in random_number:
                    # testX.append(v[i][1:])
                    testX.append(v[i])
                    testY.append(k)
                else:
                    # trainX.append(v[i][1:]) #removing the class label
                    trainX.append(v[i])
                    trainY.append(k)

    else:
        for k,v in img.items():
            for i in range(test_instances,number_per_class):
                testX.append(v[i])
                testX_file.append(v[i][1:])
                testY.append(k)
            for i in range(0,test_instances):
                trainX.append(v[i])
                trainX_file.append(v[i][1:])
                trainY.append(k)

    print('Total number of images to train: ', len(trainX))
    print('Total number of images to test: ', len(testX))

    with open('trainX.txt', 'w') as f:
        f.write(np.array2string(np.array(trainX_file), separator=', '))
    with open('trainY.txt', 'w') as f:
        f.write(np.array2string(np.array(trainY), separator=', '))
    with open('testX.txt', 'w') as f:
        f.write(np.array2string(np.array(testX_file), separator=', '))
    with open('testY.txt', 'w') as f:
        f.write(np.array2string(np.array(testY), separator=', '))

    return np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)


def find_distance(train_instance, test_instance):
    """
    :param train_instance: list of integers representing each train image instances
    :param test_instance: list of integers representing each test image instances
    :return: Euclidean distance between train and test instance
    """
    dist = np.linalg.norm(np.array(train_instance) - np.array(test_instance))

    return round(dist,4)


def predict_class(centroid_data, test_instance):
    """
    :param train_data: List of list of integers
    :param test_instance: List of list of integers
    :return: list of neighbours of the train instances
    """
    distances = []

    # for each centroid data, calculates the distance from each test instances
    for centroid_instance in centroid_data:
        dist=find_distance(centroid_instance[1:], test_instance[1:])
        distances.append((centroid_instance[0],dist))

    # sorts the distances from centroids in ascending order
    sorted_distances = sorted(distances, key=lambda v: v[1])

    predicted_class, dist = sorted_distances[0]

    return predicted_class


def find_centroid(train_data):
    """
    :param train_data: list of integer list representing train data
    :return: centroid classes representing each classes
    """
    dict = {}
    for i in train_data:
        if i[0] not in dict.keys():
            dict[i[0]] = [i[1:]]
        else:
            dict[i[0]].append(i[1:])

    centroids = {}
    # sums the same image features of each classes and divides by length of data to
    # find mean of the features which represent features of the centroid
    for i in dict.keys():
        centroids[i] = [round(sum(x) / len(x)) for x in zip(*dict[i])]  # need to divide the features with total

    cent = []
    x=0
    for k,v in centroids.items():
        cent.insert(x,[k])
        for i in v:
            cent[x].append(i)
        x+=1

    return cent


def digit_2_letter_convert(str):
    """
    :param str: string representing classes
    :return: corresponding classes in numbers
    """
    classes=[]
    for i in str:
        a = int(i)+64
        classes.append(chr(a))

    return classes


def centroid_method(train_data, test_data):
    """
    :param train_data: list of integer list representing training data instances
    :param test_data: list of integer list representing test data instances
    :return: list of predicted class labels
    """
    predicted_classes = []
    centroids = find_centroid(train_data)

    for test_instance in test_data:
        predicted_classes.append(predict_class(centroids, test_instance))

    return predicted_classes


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

    calculated_accuracy = (correct_prediction/len(test_label))*100

    return calculated_accuracy


def letter_2_digit_convert(str):
    """
    :param str: string representing classes
    :return: corresponding classes in numbers
    """
    classes =[ord(i)-ord('A')+1 for i in str.upper()]
    return classes


if __name__ == '__main__':
    name = "KNBR"  # (A) Your FirstName (KN) and Your FamilyName (BR)
    utaID = '5716'  # UTA ID - 5715
    input_classes = letter_2_digit_convert(name)
    for i in utaID:
        input_classes.append(int(i))
    print(input_classes)

    # images = pickDataClass('ATNTFaceImages400.txt',input_classes)
    data = pickDataClass('HandWrittenLetters.txt', input_classes)
    trainX, trainY, testX, testY = splitData2TestTrain(data, number_per_class=39, test_instances=25)
    # trainX, trainY, testX, testY = splitData2TestTrain(images, number_per_class=39)

    # using centroid method from sklearn library
    centroid=NearestCentroid()
    centroid.fit(trainX, trainY)
    print("Test set predictions:\n{}".format(centroid.predict(testX)))
    print("Sklearn Centroid Test set accuracy: {:.2f}".format(centroid.score(testX, testY)*100))

    # using my nearest centroid method
    prediction = centroid_method(trainX, testX)
    print('Prediction',prediction)

    predicted_class = digit_2_letter_convert(prediction)
    print(predicted_class)

    accuracy = get_accuracy(prediction, testY)
    print("My Centroid classifier accuracy: {:.2f}".format(accuracy))
