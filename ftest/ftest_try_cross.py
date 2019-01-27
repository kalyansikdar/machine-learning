import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC


def pickDataClass(filename, class_ids):
    """
    :param filename: char_string specifing the data file to read. For example, 'ATNT_face_image.txt'
    :param class_ids: array that contains the classes to be pick. For example: (3, 5, 8, 9)
    :return:an multi-dimension array or a file, containing the data (both attribute vectors and class labels)
           of the selected classes
    """
    input_data = np.genfromtxt(filename, delimiter=',')
    data_input = input_data.T

    input_data = []
    for i in data_input:
        if i[0] in class_ids:
            input_data.append(i)

    input_data = np.array(input_data).T
    print(input_data.shape)

    return input_data


def letter_2_digit_convert(str):
    """
    :param str: string representing classes
    :return: corresponding classes in numbers
    """
    classes =[ord(i)-ord('A')+1 for i in str.upper()]
    return classes


def get_mean(data):
    #return np.average(data)
    return np.mean(data)

def get_variance(data):
    return np.var(data, ddof=1)

def calculate_pool_variance(data):
    sum = 0
    n = 0
    k = 0
    for key in data:
        sum += (data[key]['length'] - 1) * data[key]['variance']
        n += data[key]['length']
        k += 1
    return sum/(n-k)

def calculate_f_statistic(data, g_avg, pool_variance):
    sum = 0
    k = 0
    for key in data:
        sum += data[key]['length']*(data[key]['average'] - g_avg)**2
        k += 1
    if k-1 == 0 or pool_variance == 0:
          return(np.inf)
    return (sum/(k-1))/pool_variance


def get_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            row = line[:-1].split(',')
            data.append(row)
    return data

def select_feature(n, f_scores):
    return sorted(range(len(f_scores)), key=lambda i: f_scores[i])[-n:]


def f_test(data):
    labels = np.array(data[0],dtype='float')
    features = np.array(data[1:],dtype='float')

    class_ids = set(labels)
    print(class_ids)

    f_test_scores = []

    for row in features:
        row_summary = {}
        for id in class_ids:
            each_class_data = []
            row_summary[id] = {}
            for i in range(len(row)):
                if labels[i] == id:
                    each_class_data.append(row[i])

            row_summary[id]['length'] = len(each_class_data)
            row_summary[id]['average'] = np.mean(each_class_data)
            row_summary[id]['variance'] = get_variance(each_class_data)

        pool_variance = calculate_pool_variance(row_summary)
        row_mean = get_mean(row)
        f_score = calculate_f_statistic(row_summary, row_mean, pool_variance)
        f_test_scores.append(f_score)

    print('f test scores:', f_test_scores)

    top_feature_numbers = select_feature(100, f_test_scores)
    print('top feature scores:', top_feature_numbers)
    top_feature_scores=[]
    for i in top_feature_numbers:
        top_feature_scores.append([f_test_scores[i],i])

    top_feature_scores = sorted(top_feature_scores, key=lambda x:x[0], reverse=True)
    print('top feature:', top_feature_scores)
    return top_feature_scores,f_test_scores


def print_scores(top_feature_numbers):
        for i in top_feature_numbers:
            print(i)


def generate_testdata(filename, class_ids):
    test_input = np.genfromtxt(filename, delimiter=',')
    testX = []
    for i in range(len(test_input)):
        if i in class_ids:
            testX.append(test_input[i])

    print('Shape of test file:', np.array(testX).shape)
    testX = np.array(testX).T  # transposing the test input so each row represent a data
    print('Shape of test data after transposing:', testX.shape)
    return testX


def format_linear_reg_pred(predictions, labels):
    for (index, value) in enumerate(predictions):
        min_distance = 99999
        new_label = round(value)
        for label in labels:
            distance = abs(value - label)
            if min_distance > distance:
                min_distance = distance
                new_label = label
        predictions[index] = new_label

    return predictions

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


def save_file(filename,data):
    np.savetxt(filename, np.array(data), fmt="%.0f", delimiter=",")


def main():
    name = "KNBRX"  # (A) Your FirstName (KN) and Your FamilyName (BR)
    utaID = '56718'  # UTA ID - 5715
    input_classes = letter_2_digit_convert(name)
    for i in utaID:
        input_classes.append(int(i))

    # input_classes = letter_2_digit_convert(given_classes)
    input_classes = [i+1 for i in range(4)]
    print(input_classes)

    input_data = pickDataClass('GenomeTrainXY.txt', input_classes)
    # images = pickDataClass('GenomeTrainXY.txt', letter_2_digit_convert("ABCDE"))
    # input_data = pickDataClass('HandWrittenLetters.txt', input_classes)

    # print('Input from main:', input_data)
    features, scores = f_test(input_data)
    # print('Inside Main')
    print_scores(features)

    selected_rows = [i[1] for i in features]
    print('Selected Rows:', selected_rows)

    j = 0
    trainX = []
    # create trainX by selecting rows from input data : shape - 100x40
    for i in range(len(input_data)):
        if i in selected_rows:
            trainX.append(input_data[i+1]) # as input data has labels in the first row

    trainY = input_data[0]

    print('Shape of trainY', np.array(trainY).shape)
    print('Shape of trainX', np.array(trainX).shape)

    trainXY = np.vstack((trainY, trainX))
    print('Shape of trainXY', trainXY.shape)

    save_file('selected_train.txt', trainXY)

    # train data label and data, transposing so that each row represent each class
    train_data = np.vstack((trainY, trainX)).T
    print('Shape of selected trainXY', train_data.shape)
    # get label and features
    trainY = train_data[:,0] # holds all labels
    trainX = train_data[:,1:] # holds all data, 40 rows, each rows 100 features

    print('Length of trainY', len(trainY))
    print('Length of trainX', len(trainX))
    print('Length of each trainX', len(trainX[1]))

    #testX = generate_testdata('TestHandWrittenLetters.txt', selected_rows)
    testX = generate_testdata('GenomeTestX.txt', selected_rows)

    # using knn classifier from sklearn library
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(trainX, trainY)
    knn_pred = knn.predict(testX)
    knn_pred_letters = digit_2_letter_convert(knn_pred)
    print("KNN Test set predictions:\n{}".format(knn.predict(testX)))
    np.savetxt('knn_prediction.txt', knn_pred_letters, fmt="%s", delimiter=",")
    #print("Sklearn KNN Test set accuracy: {:.2f}".format(knn.score(testX, testY)))

    # using centroid method from sklearn library
    centroid=NearestCentroid(metric='minkowski')
    centroid.fit(trainX, trainY)
    centroid_pred = centroid.predict(testX)
    print("Centroid Test set predictions:\n{}".format(centroid_pred))
    centroid_pred_letter = digit_2_letter_convert(centroid_pred)
    np.savetxt('centroid_prediction.txt', centroid_pred_letter, fmt="%s", delimiter=",")

    # print('Classifier SVM selected')
    linearsvm = LinearSVC(random_state=0).fit(trainX, trainY)
    svm_pred = linearsvm.predict(testX)
    svm_pred_letters = digit_2_letter_convert(svm_pred)
    print("Linear SVC Test set predictions:\n{}".format(linearsvm.predict(testX)))
    np.savetxt('svm_prediction.txt', svm_pred_letters, fmt="%s", delimiter=",")

    # print('Linear Regression Selected')
    regressor = LinearRegression()
    regressor.fit(trainX, trainY)
    linear_reg_pred = format_linear_reg_pred(regressor.predict(testX), trainY)
    print("Linear Regression Test set predictions:\n{}".format(linear_reg_pred))
    linear_reg_pred_letter = digit_2_letter_convert(linear_reg_pred)
    np.savetxt('linear_reg_prediction.txt', linear_reg_pred_letter, fmt="%s", delimiter=",")


main()