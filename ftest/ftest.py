import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from munkres import Munkres, print_matrix


def pickDataClass(filename, class_ids):
    """
    Formats the input data
    :param filename: char_string specifing the data file to read. For example, 'ATNT_face_image.txt'
    :param class_ids: array that contains the classes to be pick. For example: (3, 5, 8, 9)
    :return: input data in numpy array format
    """
    input_data = np.genfromtxt(filename, delimiter=',')
    data_input = input_data.T

    input_data = []
    for i in data_input:
        if i[0] in class_ids:
            input_data.append(i)

    input_data = np.array(input_data).T
    print('Shape of input data:', input_data.shape)

    return input_data


def letter_2_digit_convert(str):
    """
    Converts letters into corresponding digits
    :param str: string representing classes
    :return: corresponding classes in numbers
    """
    classes =[ord(i)-ord('A')+1 for i in str.upper()]
    return classes


def get_mean(data):
    """
    Calculates average of the input data
    :param data: list holding multiple values
    :return: mean of the input data
    """
    return np.mean(data)

def get_variance(data):
    """
    Calculates variance of the input data
    :param data: list holding multiple values
    :return: variance of the input data
    """
    return np.var(data, ddof=1)

def calculate_pool_variance(data):
    """
    Calculates the pooled variance of the whole data
    :param data: input data
    :return: pooled variance of the data
    """
    sum = 0
    n = 0
    k = 0
    for key in data:
        sum += (data[key]['length'] - 1) * data[key]['variance']
        n += data[key]['length']
        k += 1
    return sum/(n-k)

def calculate_f_statistic(data, g_avg, pool_variance):
    """
    Calculates f-stats of the input data
    :param data: summary of each data row(features)
    :param g_avg: average of the whole feature across class
    :param pool_variance: pooled variance
    :return: f-statistics
    """
    sum = 0
    k = 0
    for key in data:
        sum += data[key]['length']*(data[key]['average'] - g_avg)**2
        k += 1
    if k-1 == 0 or pool_variance == 0:
          return(np.inf)
    return (sum/(k-1))/pool_variance


def select_feature(n, f_scores):
    """
    Selects top n number of f_scores
    :param n: integer denoting top value
    :param f_scores: all f-test scores
    :return: 2D list holding top 100 f-test scores and corresponding row number
    """
    return sorted(range(len(f_scores)), key=lambda i: f_scores[i])[-n:]


def get_ftest_score(data):
    """
    Calculates the f-test score of each of the rows of the input data
    :param data: 2D array
    :return: top 100 f-test scores and corresponding row number
    :return: all f-test scores
    """
    labels = np.array(data[0],dtype='float')
    features = np.array(data[1:],dtype='float')

    print('Shape of label input: ', labels.shape)
    print('Shape of feature input: ', features.shape)

    class_ids = set(labels)

    f_test_scores = []

    for row in features:
        row_summary = {}
        for cls in class_ids:
            each_class_data = []
            row_summary[cls] = {}
            for i in range(len(row)):
                if labels[i] == cls:
                    each_class_data.append(row[i])

            row_summary[cls]['length'] = len(each_class_data)
            row_summary[cls]['average'] = np.mean(each_class_data)
            row_summary[cls]['variance'] = get_variance(each_class_data)

        pool_variance = calculate_pool_variance(row_summary)
        row_mean = get_mean(row)
        f_score = calculate_f_statistic(row_summary, row_mean, pool_variance)
        f_test_scores.append(f_score)

    print('All f-test scores:', f_test_scores)

    # get top 100 features based on top f-test score
    top_feature_numbers = select_feature(100, f_test_scores)
    print('Top 100 feature Rows:', top_feature_numbers)
    top_feature_scores=[]
    for i in top_feature_numbers:
        top_feature_scores.append([f_test_scores[i],i])

    # top_feature_scores = (sorted(top_feature_scores)[::-1])

    top_feature_scores = sorted(top_feature_scores, key=lambda x:x[0], reverse=True)

    return top_feature_scores,f_test_scores


def print_scores(top_feature_numbers):
        for i in top_feature_numbers:
            print(i)


def format_linear_reg_pred(predictions, labels):
    """
    Provides absolute labels for linear Regression Prediction
    """
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
    """
    Saves numpy array in a file in a particular format
    :param filename: name of the file
    :param data: numpy array data
    """
    np.savetxt(filename, np.array(data), fmt="%.0f", delimiter=",")

def get_accuracy(confusion_matrix):
    """
    Calculates accuracy of the confusion matrix
    :param confusion_matrix: input confusion matrix as 2D numpy array
    :return: Calculated accuracy score
    """
    correct_prediction = 0
    total = 0
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            if i==j:
                correct_prediction+=abs(confusion_matrix[i][j])
            total += abs(confusion_matrix[i][j])

    print('Number of Correct Prediction: ', correct_prediction)
    print('Total Count of Labels: ', total)
    acc_score = (correct_prediction/total)*100

    return acc_score

def apply_kmeans(trainX, trainY, k):
    """
    Applies k-means clustering algorithm on trainX data
    :param trainX: input feature data
    :param trainY: input label data
    :param k: Number of clusters to be generated
    :return: confusion matrix based on the clustering
    """
    print('Selected K = {} for Kmeans Clustering'.format(k))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(trainX)

    y_pred = kmeans.predict(trainX)
    print("K-Means Prediction: ", y_pred)
    y_prediction = [i + 1 for i in y_pred]

    print("Modified K-Means Prediction: ", y_prediction)

    distinct_train_y = list(set(trainY))
    for (index, pred) in enumerate(y_prediction):
        y_prediction[index] = distinct_train_y[pred - 1]

    print('Mapped Distinct Prediction:', set(y_prediction))
    print('Distinct trainY: ', set(trainY))

    conf_matrix = confusion_matrix(y_true=trainY, labels=list(set(trainY)), y_pred=y_prediction)

    kmeans_accuracy = accuracy_score(trainY, y_prediction) * 100
    print('K-Means Accuracy:', kmeans_accuracy)

    return conf_matrix

def apply_munkrees(conf_matrix, k):
    """
    Applies munkress algorithm to permute the confusion matrix using bipartite graph matching
    :param conf_matrix: input 2D array confusion matrix
    :param k: number of clusters
    :return: permuted confusion matrix
    """
    new_cm = []
    for row in conf_matrix:
        new_row = [-i for i in row]
        new_cm.append(new_row)

    # Applying Munkre's algorithm
    m = Munkres()
    indexes = m.compute(new_cm)
    # print_matrix(new_cm, msg='Find Lowest cost through this matrix:')
    print('Index Calculated by Munkres Algorithm:', indexes)

    # Columnwise bi-partite matching
    final_matrix = np.zeros([k, k])
    for i in indexes:
        final_matrix[:, i[0]] = np.array(new_cm)[:, i[1]]

    final_cm = []
    for row in final_matrix:
        new_row = [-i for i in row]
        final_cm.append(new_row)

    return final_cm


def main():
    k = 10
    name = "SIKALY"  # (A) Your FirstName (KN) and Your FamilyName (BR)
    utaID = '5726'  # UTA ID - 5715
    input_classes = letter_2_digit_convert(name)
    for i in utaID:
        input_classes.append(int(i))

    utaID_letter = digit_2_letter_convert(utaID)
    letter_input = []
    for i in name:
        letter_input.append(i)
    for i in utaID_letter:
        letter_input.append(i)

    print('Input Letters:', letter_input)
    print('Input Classes:', input_classes)

    # input_data = pickDataClass('GenomeTrainXY.txt', input_classes)
    input_data = pickDataClass('HandWrittenLetters.txt', input_classes)
    save_file('input.txt', input_data)

    # get f-test score
    features, scores = get_ftest_score(input_data)

    # print top 100 features and their f-test score
    print_scores(features)

    # get the features with top 100 f-test scores
    selected_rows = [i[1] for i in features]

    j = 0
    trainX = []

    # create trainX by selecting rows from input data : shape - 100x40
    for i in range(len(input_data)):
        if i in selected_rows:
            trainX.append(input_data[i+1]) # as input data has labels in the first row

    trainY = input_data[0]

    print('Shape of trainY', np.array(trainY).shape)
    print('Shape of trainX', np.array(trainX).shape)

    save_file('trainX.txt',trainX)

    trainXY = np.vstack((trainY, trainX))
    print('Shape of trainXY', trainXY.shape)

    save_file('selected_train.txt', trainXY)

    # train data label and data, transposing so that each row represent each class
    train_data = np.vstack((trainY, trainX)).T
    print('Shape of selected trainXY', train_data.shape)

    # get label and features
    trainY = train_data[:,0]  # holds all labels
    trainX = train_data[:,1:]  # holds all data, 40 rows, each rows 100 features

    print('Shape of trainX: ', trainX.shape)

    save_file('trainX_for_keans.txt', trainX)

    # Applying k-means clustering
    confusion_matrix = apply_kmeans(trainX, trainY, k=10)
    print_matrix(confusion_matrix, 'Initial Confusion Matrix')
    print("My Accuracy {}%".format(get_accuracy(confusion_matrix)))

    # Apply Munkres Algorithm to use optimal bipartite matching
    permuted_confusion_matrix = apply_munkrees(confusion_matrix, k=10)
    print_matrix(permuted_confusion_matrix, 'Permuted Confusion Matrix')

    print("My Accuracy {}%".format(get_accuracy(permuted_confusion_matrix)))


main()