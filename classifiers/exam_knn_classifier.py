import random
from itertools import groupby
import numpy as np
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


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
        input_data.append(file_name.readline())

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

    #Create 2d list with only the data of classes required with class label
    images = [img for img in images if img[0] in class_ids]

    return images


def splitData2TestTrain(images, number_per_class=None,  test_instances=26):
    """
    :param filename: char_string specifing the data file to read. This can also be an array containing input data.
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
    testY_file=[]
    trainY_file=[]

    random_number=[]
    img ={}
    # Creates dictionary with each class as keys and the array of images of that class as values
    # segregates the data as per classes (class, [[data][data][data]])
    for key,group in groupby(images, lambda x:x[0]):
        img[key]=list(group)

    # print the classes and amount of images in that class
    for k,v in img.items():
        print(k, len(v))

    # randomly selects "test_instances" number of images and saves into testX and rest into trainX
    # and corresponding labels into testY and trainY
    if number_per_class == None:
        for k, v in img.items():
            if len(v) > 1:
                random_number = random.sample(range(0, len(v) - 1), test_instances)
                print('random', random_number)
            for i in range(len(v)):
                if i in random_number:
                    #testX.append(v[i][1:])
                    testX.append(v[i])
                    testY.append(k)
                else:
                    #trainX.append(v[i][1:]) #removing the class label
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
    :param train_instance: list of integers representing train instance
    :param test_instance: list of integers representing test instance
    :return: Euclidean distance between trains and test instance
    """
    dist = np.linalg.norm(np.array(train_instance) - np.array(test_instance))
    #print('distance',round(dist,4))
    return round(dist,4)


def find_neighbours(train_data, test_instance, k=0):
    """
    :param train_data: List of integer list representing train data
    :param test_instance: Integer list representing a test data instance
    :param k: number of neighbours, defaulted to 0
    :return: list of neighbours of the train instances
    """
    distances = []
    dist = []
    neighbours = []
    # finds distance of the test instance from each train instances and stores in a list
    for train_instance in train_data:
        dist=find_distance(train_instance[1:], test_instance[1:])
        distances.append((train_instance[0],dist))
        #distances.append(('A', dist))

    # sorts all the distances in ascending order
    sorted_distances = sorted(distances, key=lambda v: v[1])
    if k!= 0: # selects first k distances to consider only the k neighbors
        sorted_distances = sorted_distances[0:k]

    # stores only the labels as neighbors
    print(sorted_distances)
    neighbours = [i[0] for i in sorted_distances]
    #neighbours = [sum(i)/len(sorted_distances) for i in sorted_distances]
    print("neightbors", neighbours)
    return neighbours


def predict_class(neighbours):
    """
    :param neighbours: list of class labels of the neighbors
    :return: the class label which features maximum number of times
    """
    class_count= {}
    # finds frequency mapping of classes
    for i in neighbours:
        if i in class_count.keys():
            class_count[i] = class_count[i] + 1
        else:
            class_count[i] = 1

    return max(class_count.items(), key=operator.itemgetter(1))[0]


def knn_classifier(train_data, test_data, k):
    """
    :param train_data: Input train data. List of list of integers
    :param test_data: Input test data. List of integers
    :param k: number of neighbours
    :return: list of predicted class of the test data
    """
    neighbours = []
    predicted_class = []
    # find neighbours for each test data instances
    for test_instance in test_data:
        # each test instance's neighbors are calculated based on each train_data instances, hence, whole train_data
        # is used as parameters
        neighbours = find_neighbours(train_data, test_instance, k)

        prediction = predict_class(neighbours)

        predicted_class.append(prediction)

    return predicted_class


def get_accuracy(predicted_classes, test_label):
    """
    :param trainX: list of integer list representing training data instances
    :param testX: list of integer list representing test data instances
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

def digit_2_letter_convert(str):
    """
    :param str: string representing classes
    :return: corresponding classes in numbers
    """
    #classes =[chr(i)+ ord('A')+1 for i in str]
    classes=[]
    for i in str:
        a = int(i)+64
        classes.append(chr(a))
    #classes = [chr(i) for i in str]
    # for i in classes:
    #     x.append(unichr(1 + ord(u'\u0B85')))
    #
    # print(x)
    return classes


def main():
    k = 3

    name = "KNBR" # (A) Your FirstName (KN) and Your FamilyName (BR)
    utaID = '5716' # UTA ID - 5715
    input_classes = letter_2_digit_convert(name)
    print(input_classes)
    for i in utaID:
        input_classes.append(int(i))
    print(input_classes)
    #input_classes = [i for i in range(41)]
    #images = pickDataClass('trainDataXY.txt', [1,2,3,4,5])
    # images = pickDataClass('testDataXY.txt', [1,2,3,4,5])
    #images = pickDataClass('ATNTFaceImages400.txt',input_classes)
    # images = pickDataClass('ATNTFaceImages400.txt',letter_2_digit_convert("ABCDE"))
    #images = pickDataClass('HandWrittenLetters.txt', letter_2_digit_convert("ABCDE"))
    images = pickDataClass('HandWrittenLetters.txt', input_classes)

    # given_splits = [(5, 34), (10, 29), (15, 24), (20, 19), (25, 14), (30, 9), (35, 4)]
    # for i in given_splits:
    #     trainX, trainY, testX, testY = splitData2TestTrain(images, number_per_class=, test_instances=i[0])

    trainX, trainY, testX, testY = splitData2TestTrain(images, number_per_class=39, test_instances=25)

    # using knn classifier from sklearn library
    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(trainX,trainY)
    # print("Test set predictions:\n{}".format(knn.predict(testX)))
    # print("Sklearn KNN Test set accuracy: {:.2f}".format(knn.score(testX, testY)))

    # using own knn
    prediction = knn_classifier(trainX, testX, k)
    print(prediction)
    predicted_class = digit_2_letter_convert(prediction)
    print(predicted_class)


    accuracy = get_accuracy(prediction, testY)
    print("My KNN Test set accuracy: {:.2f}".format(accuracy))

    train_class = np.array(trainY)
    train_feature = np.array(trainX)
    test_class = np.array(testY)
    test_feature = np.array(testX)

    linearsvm = LinearSVC(random_state=0).fit(train_feature, train_class)
    print("SVM Classifier accuracy: {:.3f}".format((linearsvm.score(test_feature, test_class)) * 100))

main()