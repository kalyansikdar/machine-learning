from itertools import groupby
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from munkres import Munkres, print_matrix


def pickDataClass(filename, class_ids):
    """
    Extracts the data from file without using any external library - numpy or pandas
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


def splitData2DataLabel(images, number_per_class=None):
    """
    Splits the data into feature and label
    :param number_per_class: number of data instances in each class (we assume every class has the same number of data instances)
    :return: Training_attributeVector(trainX), Training_labels(trainY), Test_attributeVectors(testX), Test_labels(testY)
    The data should easily feed into a classifier.
    """
    trainX=[]
    trainY=[]
    trainX_file = []

    img ={}

    # Creates dictionary with each class as keys and the array of images of that class as values
    # segregates the data as per classes (class, [[data][data][data]])
    for key,group in groupby(images, lambda x:x[0]):
        img[key]=list(group)

    # print the classes and amount of images in that class
    # for k,v in img.items():
    #     print(k, len(v))

    for k,v in img.items():
        for i in range(0,number_per_class):
            trainX.append(v[i])
            trainX_file.append(v[i][1:])
            trainY.append(k)

    print('Total number of images to classify: ', len(trainX))
    print('Total number of labels: ', len(trainY))
    print('Distinct labels: ', set(trainY))

    with open('trainX.txt', 'w') as f:
        f.write(np.array2string(np.array(trainX_file), separator=', '))
    with open('trainY.txt', 'w') as f:
        f.write(np.array2string(np.array(trainY), separator=', '))

    return np.array(trainX), np.array(trainY)


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
    classes=[]
    for i in str:
        a = int(i)+64
        classes.append(chr(a))

    return classes


def get_accuracy(conf_matrix):
    """
    Calculates the accuracy of the confusion matrix by considering the number of elements at (i,j) position as correct
    and rest as incorrect
    :param conf_matrix: 2D array
    :return: accuracy percentage
    """
    correct_prediction = 0
    total = 0
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            if i==j:
                correct_prediction+=abs(conf_matrix[i][j])
            total += abs(conf_matrix[i][j])

    print('Number of Correct Prediction: ', correct_prediction)
    print('Total Count of Labels: ', total)
    acc_score = (correct_prediction/total)*100
    return acc_score


def save_file(filename,data):
    """
    Saves the data in a file
    :param filename: The target file name
    :param data: data to be saved in the file
    """
    np.savetxt(filename, np.array(data), fmt="%.0f", delimiter=",")


def main():
    k = 10 # number of neighborsletter_2_digit_convert("ABCDE")
    print('Value of K: ', k)
    #given_classes = 'CF'
    name = "SIKALY" # (A) Your FirstName (KN) and Your FamilyName (BR)
    utaID = '5726' # UTA ID - 5715
    utaID_letter = digit_2_letter_convert(utaID)
    letter_input = []
    for i in name:
        letter_input.append(i)
    for i in utaID_letter:
        letter_input.append(i)

    input_classes = letter_2_digit_convert(name)
    for i in utaID:
        input_classes.append(int(i))

    #input_classes = [i+1 for i in range(start,end)]
    print('Input Classes:', input_classes)

    # images = pickDataClass('ATNTFaceImages400.txt', input_classes)
    images = pickDataClass('HandWrittenLetters.txt', input_classes)

    trainX, trainY = splitData2DataLabel(images, number_per_class=39)

    save_file('trainX.txt', trainX)
    save_file('trainY.txt', trainY)
    print('Shape of TrainX: ', np.array(trainX).shape)

    # Applying k-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(trainX)

    y_pred = kmeans.predict(trainX)
    print("K-Means Prediction: ", y_pred)
    y_prediction = [i+1 for i in y_pred]  # as the predicted classes should be assigned from 1

    print("Modified K-Means Prediction: ", y_prediction)

    distinct_train_y = list(set(trainY))
    for (index, pred) in enumerate(y_prediction):
        y_prediction[index] = distinct_train_y[pred - 1]

    print ('Mapped Distinct Prediction:',set(y_prediction))
    print('Distinct trainY: ', set(trainY))
    print('Length Prediction:', len(y_prediction))
    print('Length trainY: ', len(trainY))

    conf_matrix = confusion_matrix(y_true=trainY, labels=list(set(trainY)), y_pred=y_prediction)
    print(conf_matrix.shape)
    print_matrix(conf_matrix,"Confusion Matrix")

    kmeans_accuracy = accuracy_score(trainY, y_prediction)*100
    print ('K-Means Accuracy:', kmeans_accuracy)

    print("My Accuracy {}%" .format(get_accuracy(conf_matrix)))

    new_cm = []
    for row in conf_matrix:
        new_row = [-i for i in row]
        new_cm.append(new_row)

    # Applying Munkre's algorithm
    m = Munkres()
    indexes = m.compute(new_cm)
    print('Index Calculated by Munkres Algorithm:',indexes)

    # Column wise organization
    final_matrix = np.zeros([k,k])

    for i in indexes:
        final_matrix[:,i[0]] = np.array(new_cm)[:,i[1]]

    final_cm = []
    for row in final_matrix:
        new_row = [-i for i in row]
        final_cm.append(new_row)

    print_matrix(final_cm)

    print("My Accuracy {}%".format(get_accuracy(final_cm)))


main()