"""Machine learning algorithm evaluation functions. 

NAME: Tony Nguyen
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_learn import *
from data_util import *
from random import randint


#----------------------------------------------------------------------
# HW-6
#----------------------------------------------------------------------

def stratify(table, label_column, k):
    """Returns a list of k stratified folds as data tables from the given
    data table based on the label column.

    Args:
        table: The data table to partition.
        label_column: The column to use for the label. 
        k: The number of folds to return. 
    
    Return:
        folds (list): A list of k data tables.

    Note: Does not randomly select instances for the folds, and
        instead produces folds in order of the instances given in the
        table.

    """
    filtered_dict = {}
    folds = []
    unique_label = set(table.get_column_data(label_column))

    for label in unique_label:
        filtered_dict[label] = filter(table, [label_column], [label])

    for i in range(k):
        folds.append(DataTable(table.columns()))

    for label in filtered_dict:
        for i in range(filtered_dict[label].row_count()):
            folds[i % k].append(filtered_dict[label][i].values())
    return folds

def union_all(tables):
    """Returns a table containing all instances in the given list of data
    tables.

    Args:
        tables: A list of data tables. 

    Notes: Returns a new data table that contains all the instances of
       the first table in tables, followed by all the instances in the
       second table in tables, and so on. The tables must have the
       exact same columns, and the list of tables must be non-empty.

    """
    # Error checking
    if not len(tables):
        raise ValueError("List of tables is empty")
    
    for item in tables:
        if item.columns() != tables[0].columns():
            raise ValueError(f'Tables index {item} do not have the same columns')
        
    # Joining tables
    result = DataTable(tables[0].columns())

    for table in tables:
        for row in range(table.row_count()):
            result.append(table[row].values())

    return result


def naive_bayes_eval(train, test, label_col, continuous_cols, categorical_cols=[]):
    """Evaluates the given test set using naive bayes over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       continuous_cols: The continuous columns (estimated via PDF)
       categorical_cols: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    # Create an empty confusion matrix
    unique_labels = set(train.get_column_data(label_col))
    num_labels = len(unique_labels)
    confusion_matrix = [[0] * num_labels for _ in range(num_labels)]

    # Separate the label column from the test set
    test_copy = test.copy()
    test_label = test_copy.get_column_data(label_col)
    test_copy.drop([label_col])

    # Iterate through the test set
    for i in range(test_copy.row_count()):
        # Run knn on the test instance
        naive_bayes_result = naive_bayes(train, test_copy[i], label_col, continuous_cols, categorical_cols)
        
        # Update the confusion matrix
        actual_index = list(unique_labels).index(test_label[i])
        
        # Gets the first label if mutiple returns
        predicted_index = list(unique_labels).index(naive_bayes_result[0][0]) 
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


def naive_bayes_stratified(table, k_folds, label_col, cont_cols, cat_cols=[]):
    """Evaluates naive bayes over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        cont_cols: The continuous columns for naive bayes. 
        cat_cols: The categorical columns for naive bayes. 

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    folds = stratify(table, label_col, k_folds)
    confusion_matrix = None
  
    for fold in folds:
        train = union_all([folds[i] for i in range(k_folds) if i != folds.index(fold)])

        if confusion_matrix is None:
            confusion_matrix = naive_bayes_eval(train, fold, label_col, cont_cols, cat_cols)
        else:
            concat_two_confusion_matrices(confusion_matrix, naive_bayes_eval(train, fold, 
                                                label_col, cont_cols, cat_cols))

    return confusion_matrix


def knn_stratified(table, k_folds, label_col, vote_fun, k, num_cols, nom_cols=[]):
    """Evaluates knn over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        vote_fun: The voting function to use with knn.
        num_cols: The numeric columns for knn.
        nom_cols: The nominal columns for knn.

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    folds = stratify(table, label_col, k_folds)
    confusion_matrix = None
  
    for fold in folds:
        train = union_all([folds[i] for i in range(k_folds) if i != folds.index(fold)])

        if confusion_matrix is None:
            confusion_matrix = knn_eval(train, fold, vote_fun, k, label_col, num_cols, nom_cols)
        else:
            concat_two_confusion_matrices(confusion_matrix, knn_eval(train, fold, vote_fun, k, 
                                                label_col, num_cols, nom_cols))

    return confusion_matrix


def concat_two_confusion_matrices(matrix1, matrix2):
    """Returns a new confusion matrix that is the result of concatenating
    the two given confusion matrices.

    Args:
        matrix1: The first confusion matrix.
        matrix2: The second confusion matrix.

    Notes: 
        The two matrices must have the same columns, and the
            'actual' column is used as the key for concatenation.
        The two matrices must have the same labels.

    """
    # Error checking
    if matrix1.columns() != matrix2.columns():
        raise ValueError("The two matrices must have the same columns")
    
    if matrix1.row_count() != matrix2.row_count():
        raise ValueError("The two matrices must have the same number of rows")
    
    # Concatenating matrices
    for row in range(matrix1.row_count()):
        for col in range(len(matrix1.columns())):
            if matrix1.columns()[col] != 'actual':
                matrix1[row][matrix1.columns()[col]] += matrix2[row][matrix2.columns()[col]]

    return matrix1


#----------------------------------------------------------------------
# HW-5
#----------------------------------------------------------------------


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





