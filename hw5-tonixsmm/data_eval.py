"""Machine learning algorithm evaluation functions. 

NAME: Tony Nguyen
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_learn import *
from data_util import *
from random import randint


def holdout(table, test_set_size):
    """Partitions the table into a training and test set using the holdout method. 

    Args:
        table: The table to partition.
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """
    test_set = DataTable(table.columns())
    training_set = table.copy()

    for i in range(test_set_size):
        rand_row = randint(0, training_set.row_count()-1)
        test_set.append(training_set[rand_row].values())
        del training_set[rand_row]

    return (training_set, test_set)


def knn_eval(train, test, vote_fun, k, label_col, numeric_cols, nominal_cols=[]):
    """Returns a confusion matrix resulting from running knn over the
    given test set. 

    Args:
        train: The training set.
        test: The test set.
        vote_fun: The function to use for selecting labels.
        k: The k nearest neighbors for knn.
        label_col: The column to use for predictions. 
        numeric_cols: The columns compared using Euclidean distance.
        nominal_cols: The nominal columns for knn (match or no match).

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the given voting function returns multiple labels, the
        first such label is selected.

    """
    # Create an empty confusion matrix
    unique_labels = set(train.get_column_data(label_col))
    num_labels = len(unique_labels)
    confusion_matrix = [[0] * num_labels for _ in range(num_labels)]

    # Separate the label column from the test set
    test_copy = test.copy()
    test_label = DataTable.get_column_data(test_copy, label_col)
    test_copy.drop([label_col])

    # Iterate through the test set
    for i in range(test_copy.row_count()):
        # Run knn on the test instance
        knn_result = knn(train, test_copy[i], k, numeric_cols, nominal_cols)
        scores, instances = handle_knn_dict(knn_result)
        
        scaled_scores = [0 - score for score in scores]
        
        # Use the vote_fun to get the predicted label
        predicted_label = vote_fun(instances, scaled_scores, label_col)
        
        # Update the confusion matrix
        actual_index = list(unique_labels).index(test_label[i])
        
        # Gets the first label if mutiple returns
        predicted_index = list(unique_labels).index(predicted_label[0]) 
        confusion_matrix[actual_index][predicted_index] += 1

    # Create a data table for the confusion matrix
    columns = ['actual'] + list(unique_labels) + ['total']
    result = DataTable(columns)
    
    for i in range(num_labels):
        actual = list(unique_labels)[i]

        # Get the total count for the row
        sum_row = 0
        for j in confusion_matrix[i]:
            sum_row += j

        row_data = [actual] + confusion_matrix[i] + [sum_row]
        result.append(row_data)

    # Get the total count for the column
    append_row = ['total']
    for i in result.columns():
        sum_col = 0
        if i != 'actual':
            for j in result.get_column_data(i):
                sum_col += j
            append_row.append(sum_col)
    result.append(append_row)

    return result


def handle_knn_dict(knn_result):
    distance = []
    intansces = []
    for item in knn_result:
        for instance in knn_result[item]:
            distance.append(item)
            intansces.append(instance)

    return (distance, intansces)


def find_metrics_from_confusion_matrix(confusion_matrix, label):
    """Given a confusion matrix and a label, returns a dictionary of metrics
    including true positives, true negatives, false positives, false negatives,
    positive predictions, negative predictions, actual positives, actual negatives,
    and total positives.

    Parameters:
        confusion_matrix (DataTable): A confusion matrix.
        label (int): The label to calculate metrics for.

    Returns:
        dict: A dictionary of metrics.
    """
    matrix = confusion_matrix.copy()
    labels = matrix.columns()
    labels.remove('actual')

    if 'total' in labels:
        labels.remove('total')

    if matrix[matrix.row_count() - 1]['actual'] == 'total':
        del matrix[matrix.row_count() - 1]

    metrics_dict = {'tp': 0, 'tn': 0,
                    'fp': 0, 'fn': 0,
                    'p_pred': 0, 'n_pred': 0,
                    'p_actual': 0,'n_actual': 0,
                    'total_p': 0}

    for row in range(matrix.row_count()):
        for col in labels:
            if label == matrix[row]['actual'] and label == col:
                metrics_dict['tp'] += matrix[row][col]
            elif label != matrix[row]['actual'] and label != col:
                metrics_dict['tn'] += matrix[row][col]
            elif label != matrix[row]['actual'] and label == col:
                metrics_dict['fp'] += matrix[row][col]
            elif label == matrix[row]['actual'] and label != col:
                metrics_dict['fn'] += matrix[row][col]

        if confusion_matrix[row]['actual'] == 'total':
            pass

    metrics_dict['p_pred'] = metrics_dict['tp'] + metrics_dict['fp']
    metrics_dict['n_pred'] = metrics_dict['tn'] + metrics_dict['fn']
    metrics_dict['p_actual'] = metrics_dict['tp'] + metrics_dict['fn']
    metrics_dict['n_actual'] = metrics_dict['tn'] + metrics_dict['fp']
    metrics_dict['total_p'] = metrics_dict['p_actual'] + metrics_dict['n_actual']

    return metrics_dict


def accuracy(confusion_matrix, label):
    """Returns the accuracy for the given label from the confusion matrix.
    
    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the accuracy of.

    """
    metrics = find_metrics_from_confusion_matrix(confusion_matrix, label)
    
    return (metrics['tp'] + metrics['tn']) / metrics['total_p']


def precision(confusion_matrix, label):
    """Returns the precision for the given label from the confusion
    matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the precision of.

    """
    metrics = find_metrics_from_confusion_matrix(confusion_matrix, label)

    return (metrics['tp'] / metrics['p_pred']) if metrics['p_pred'] != 0 else -1


def recall(confusion_matrix, label): 
    """Returns the recall for the given label from the confusion matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the recall of.

    """
    metrics = find_metrics_from_confusion_matrix(confusion_matrix, label)

    return (metrics['tp'] / metrics['p_actual']) if metrics['p_actual'] != 0 else -1

