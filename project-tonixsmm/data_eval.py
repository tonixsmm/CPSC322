"""Machine learning algorithm evaluation functions. 

NAME: Tony Nguyen
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_util import *
from data_learn import *
from random import randint


#----------------------------------------------------------------------
# HW-8
#----------------------------------------------------------------------


def run_standard_eval_metrices(confusion_matrix):
    """Return the standard evaluation metrics given a confusion matrix.
    This includes the macro accuracy, precision, recall, and F1-score.

    Args:
        confusion_matrix (DataTable): The confusion matrix

    Returns:
        tuple: The standard evaluation metrics (accuracy, precision, recall, F1-score)
    """
    label_list = confusion_matrix.get_column_data('actual')
    if 'total' in label_list:
        label_list.remove('total')

    # Find non-empty labels
    sum_of_actual_labels = confusion_matrix.get_column_data('total')
    non_empty_label = 0
    for i in sum_of_actual_labels:
        if i != 0:
            non_empty_label += 1

    if confusion_matrix[confusion_matrix.row_count() - 1]['actual'] == 'total':
        non_empty_label -= 1

    # Accuracy
    total_accuracy = 0
    for i in label_list: 
        num = accuracy(confusion_matrix, i)
        if num != -1:
            total_accuracy += num
    acc = total_accuracy / non_empty_label

    # Precision
    total_precision = 0
    denum = non_empty_label
    for i in label_list:
        num = precision(confusion_matrix, i)
        if num != -1:
            total_precision += num
        else:
            non_empty_label -= 1
    precision_m = total_precision / denum

    # Recall
    total_recall = 0
    denum = non_empty_label
    for i in label_list: # labels are 1, 2, 3
        num = recall(confusion_matrix, i)
        if num != -1:
            total_recall += num
        else:
            denum -= 1
    recall_m = total_recall / non_empty_label

    # F-score
    f_score = 2 * precision_m * recall_m / (precision_m + recall_m)

    return (acc, precision_m, recall_m, f_score)


def bootstrap(table): 
    """Creates a training and testing set using the bootstrap method.

    Args: 
        table: The table to create the train and test sets from.

    Returns: The pair (training_set, testing_set)

    """
    table_copy = table.copy()
    train = DataTable(table.columns())
    test = DataTable(table.columns())

    for i in range(table.row_count()):
        rand_row = randint(0, table.row_count()-1)
        train.append(table_copy[rand_row].values())

    for i in range(table_copy.row_count()):
        if table_copy[i] not in train:
            test.append(table_copy[i].values())

    return (train, test)


def stratified_holdout(table, label_col, test_set_size):
    """Partitions the table into a training and test set using the holdout
    method such that the test set has a similar distribution as the
    table.

    Args:
        table: The table to partition.
        label_col (str): The column with the class labels. 
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """
    table_copy = table.copy()
    test_set = DataTable(table.columns())
    train_set = DataTable(table.columns())
    unique_labels = get_unique_ordered_list(table.get_column_data(label_col))

    partitioned_table = partition(table_copy, [label_col])
    label_distribution = [partitioned_table[label].row_count() / table.row_count() 
                         for label in range(len(unique_labels))]
    label_size = [int(test_set_size * label_distribution[label]) for label in range(len(unique_labels))]

    if test_set_size != 0:
        for label in range(len(unique_labels)):
            for i in range(label_size[label]):
                rand_row = randint(0, partitioned_table[label].row_count()-1)
                test_set.append(partitioned_table[label][rand_row].values())
                del partitioned_table[label][rand_row]

    train_set = union_all(partitioned_table)

    return (train_set, test_set)


def tdidt_eval_with_tree(dt_root, test, label_col, labels):
    """Evaluates the given test set using tdidt over the decision tree, 
        returning a corresponding confusion matrix.

    Args:
       dt_root (AttributeNode | LeafNode): The decision tree to use.
       test (DataTable): The testing data set.
       label_col (str): The column being predicted.
       labels (list): The list of class labels

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    """
    # Create an empty confusion matrix
    labels = list(set(labels))
    num_labels = len(labels)
    confusion_matrix = [[0] * num_labels for _ in range(num_labels)]

    # Separate the label column from the test set
    test_copy = test.copy()
    test_label = test_copy.get_column_data(label_col)
    test_copy.drop([label_col])

    # Iterate through the test set
    for i in range(test_copy.row_count()):
        # Run tdidt evaluation on the test instance
        tdidt_result = tdidt_predict(dt_root, test_copy[i])
        
        if tdidt_result is not None:
            # Update the confusion matrix
            actual_index = labels.index(test_label[i])
            
            # Gets the first label if mutiple returns
            predicted_index = labels.index(tdidt_result[0]) 
            confusion_matrix[actual_index][predicted_index] += 1

    # Create a data table for the confusion matrix
    columns = ['actual'] + labels + ['total']
    result = DataTable(columns)
    
    for i in range(num_labels):
        actual = labels[i]

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


def random_forest(table, remainder, F, M, N, label_col, columns, cfmatrix=None):
    """Returns a random forest build from the given table. 
    
    Args:
        table: The original table for cleaning up the decision tree.
        remainder: The table to use to build the random forest.
        F: The subset of columns to use for each classifier.
        M: The number of unique accuracy values to return.
        N: The total number of decision trees to build initially.
        label_col: The column with the class labels.
        columns: The categorical columns used for building the forest.

    Returns: A list of (at most) M pairs (tree, accuracy) consisting
        of the "best" decision trees and their corresponding accuracy
        values. The exact number of trees (pairs) returned depends on
        the other parameters (F, M, N, and so on).

    """
    # Error checking
    if M > N:
        raise ValueError('M cannot be greater than N')
    
    # Create N bootstrap samples
    bootstrap_samples = []

    for i in range(N):
        bootstrap_samples.append(bootstrap(remainder))

    # Create decision trees for each bootstrap sample
    tree_list = []

    for train, validation in bootstrap_samples:
        tree = tdidt_F(train, label_col, F, columns)
        tree = resolve_attribute_values(tree, table)
        tree = resolve_leaf_nodes(tree)
        tree_list.append(tree)

    # Evaluate each decision tree
    cmatrix_list = []
    label_values = list(set(table.get_column_data(label_col)))

    for tree in range(len(tree_list)):
        cmatrix_list.append(tdidt_eval_with_tree(tree_list[tree], bootstrap_samples[tree][1],
                                                 label_col, label_values))

    # Calculate the accuracy for each decision tree
    accuracy_list = []
    return_cfmatrix = {}

    for matrix in range(len(cmatrix_list)):
        # Find non-empty labels
        sum_of_actual_labels = cmatrix_list[matrix].get_column_data('total')
        non_empty_label = 0
        for i in sum_of_actual_labels:
            if i != 0:
                non_empty_label += 1

        if cmatrix_list[matrix][cmatrix_list[matrix].row_count() - 1]['actual'] == 'total':
            non_empty_label -= 1

        # Accuracy
        total_accuracy = 0
        for i in label_values: 
            num = accuracy(cmatrix_list[matrix], i)
            if num != -1:
                total_accuracy += num
        if non_empty_label == 0:
            acc = 0
        else:
            acc = total_accuracy / non_empty_label

        accuracy_list.append((tree_list[matrix], acc))
        return_cfmatrix[str((tree_list[matrix], acc))] = cmatrix_list[matrix]

    # Find the M best decision trees
    accuracy_list.sort(key=lambda x: x[1], reverse=True)

    final_cfmatrix = []
    for i in range(len(accuracy_list)):
        final_cfmatrix.append(return_cfmatrix[str(accuracy_list[i])])

    if len(accuracy_list) > M:
        accuracy_list = accuracy_list[:M]
        final_cfmatrix = final_cfmatrix[:M]

    if cfmatrix is not None:
        return accuracy_list, final_cfmatrix
    else:
        return accuracy_list


def produce_track_record(cfm_forest, label_values):
    """Return the track record of the given random forest using the given test set.

    Args:
        rforest (DataTable): The pair of tuple (tree, accuracy) consisting of 
            the "best" decision trees and their corresponding accuracy values.
        test (DataTable): The test set to use for evaluation.
        label_values (list): The list of class labels.

    Returns: A table of the track record of the given random forest using the given test set.

    """
    # Evaluate each decision tree
    result = DataTable(['classifier', 'prediction'] + list(label_values))
    
    for item in range(len(cfm_forest)):        
        labels = cfm_forest[item].columns()
        labels.remove('actual')

        if 'total' in labels:
            labels.remove('total')
        
        for col in labels:
            col_val = []
            for row in range(cfm_forest[item].row_count()):
                col_val.append(cfm_forest[item][row][col])

            for i in range(len(col_val) - 1):
                if col_val[len(col_val) - 1] != 0:
                    col_val[i] = col_val[i] / col_val[len(col_val) - 1]

            col_val = col_val[:len(col_val) - 1]
            result.append([item, col] + col_val)

    return result


def perform_track_record_voting(trees, cmatrix_list, test_instance):
    """Returns the voting result of the given random forest using track history

    Args:
        trees (list): The list of trees.
        cmatrix_list (DataTable): A table of the track record of the given random forest
        test_instance (DataRow): The test instance to use for evaluation.
    
    Return: The predicted class label

    Note: The trees and cmatrix_list must have the same length.
    """
    # Create a list of predictions
    predictions = []
    for i in range(len(trees)):
        pred = tdidt_predict(trees[i][0], test_instance)
        #tdidt_predict returns a list of tuples of predictions and their probabilities

        if pred is not None:
            predictions.append(pred)

    # Create a list of votes
    votes = []
    for i in range(len(predictions)):
        for j in range(cmatrix_list.row_count()):
            if cmatrix_list[j]['classifier'] == i and cmatrix_list[j]['prediction'] == predictions[i][0]:
                votes.append(cmatrix_list[j].values())
    
    # Perform the voting
    vote_table = DataTable(columns=cmatrix_list.columns(), data=votes)
    total_vote = []
    for col in vote_table.columns():
        if col != 'classifier' and col != 'prediction':
            vote_per_col = 0
            for row in range(vote_table.row_count()):
                vote_per_col += vote_table[row][col]
            total_vote.append(vote_per_col)

    # Get the label with the highest vote
    max_vote = max(total_vote)
    max_vote_index = total_vote.index(max_vote)

    return vote_table.columns()[max_vote_index + 2] # +2 to account for classifier and prediction columns


def random_forest_eval(table, train, test, F, M, N, label_col, columns):
    """Builds a random forest and evaluate's it given a training and
    testing set.

    Args: 
        table: The initial table.
        train: The training set from the initial table.
        test: The testing set from the initial table.
        F: Number of features (columns) to select.
        M: Number of trees to include in random forest.
        N: Number of trees to initially generate.
        label_col: The column with class labels. 
        columns: The categorical columns to use for classification.

    Returns: A confusion matrix containing the results. 

    Notes: Assumes weighted voting (based on each tree's accuracy) is
        used to select predicted label for each test row.

    """
    # Create an empty confusion matrix
    unique_labels = set(table.get_column_data(label_col))
    num_labels = len(unique_labels)
    confusion_matrix = [[0] * num_labels for _ in range(num_labels)]

    # Separate the label column from the test set
    test_copy = test.copy()
    test_label = test_copy.get_column_data(label_col)
    test_copy.drop([label_col])

    # Construct and clean the Random Forest
    random_forest_result, cfm_result = random_forest(table, train, F, M, N, label_col, columns, cfmatrix=True)
    cfm_matrices = produce_track_record(cfm_result, list(unique_labels))

    # Iterate through the test set
    for i in range(test_copy.row_count()):
        # Run run the evaluation on the test instance
        prediction_result = perform_track_record_voting(random_forest_result, cfm_matrices, test_copy[i])
        
        if prediction_result is not None:
            # Update the confusion matrix
            actual_index = list(unique_labels).index(test_label[i])
            
            # Gets the first label if mutiple returns
            predicted_index = list(unique_labels).index(prediction_result) 
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


#----------------------------------------------------------------------
# HW-7
#----------------------------------------------------------------------


def tdidt_eval(train, test, label_col, columns):
    """Evaluates the given test set using tdidt over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       columns: The categorical columns

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

    # Construct and clean the Decision Tree
    tree = tdidt(train, label_col, columns)
    tree = resolve_attribute_values(tree, train)
    tree = resolve_leaf_nodes(tree)

    # Iterate through the test set
    for i in range(test_copy.row_count()):
        # Run tdidt evaluation on the test instance
        tdidt_result = tdidt_predict(tree, test_copy[i])
        
        if tdidt_result is not None:
            # Update the confusion matrix
            actual_index = list(unique_labels).index(test_label[i])
            
            # Gets the first label if mutiple returns
            predicted_index = list(unique_labels).index(tdidt_result[0]) 
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


def tdidt_stratified(table, k_folds, label_col, columns):
    """Evaluates tdidt prediction approach over the table using stratified
    k-fold cross validation, returning a single confusion matrix of
    the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        columns: The categorical columns for tdidt. 

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    folds = stratify(table, label_col, k_folds)
    confusion_matrix = None
  
    for fold in folds:
        train = union_all([folds[i] for i in range(k_folds) if i != folds.index(fold)])

        if confusion_matrix is None:
            confusion_matrix = tdidt_eval(train, fold, label_col, columns)
        else:
            concat_two_confusion_matrices(confusion_matrix, tdidt_eval(train, fold, 
                                                label_col, columns))

    return confusion_matrix


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
    
    return (metrics['tp'] + metrics['tn']) / metrics['total_p'] if metrics['total_p'] != 0 else 0


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

