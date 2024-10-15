"""Machine learning algorithm implementations.

NAME: Tony Nguyen
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import DataTable, DataRow


#----------------------------------------------------------------------
# HW-5
#----------------------------------------------------------------------


def knn(table, instance, k, numerical_columns, nominal_columns=[]):
    """Returns the k closest distance values and corresponding table
    instances forming the nearest neighbors of the given instance. 

    Args:
        table: The data table whose instances form the nearest neighbors.
        instance: The instance to find neighbors for in the table.
        k: The number of closest distances to return.
        numerical_columns: The numerical columns to use for comparison.
        nominal_columns: The nominal columns to use for comparison (if any).

    Returns: A dictionary with k key-value pairs, where the keys are
        the distances and the values are the corresponding rows.

    Notes: 
        The numerical and nominal columns must be disjoint. 
        The numerical and nominal columns must be valid for the table.
        The resulting score is a combined distance without the final
        square root applied.

    """
    # Error Checking
    if any(x not in table.columns() for x in numerical_columns):
        raise ValueError("Numerical columns must be valid for the table.")
    
    if not len(nominal_columns) and any(x not in table.columns() for x in nominal_columns):
        raise ValueError("Nominal columns must be valid for the table.")
    
    if len(set(numerical_columns).intersection(set(nominal_columns))) > 0:
        raise ValueError("Numerical and nominal columns must be disjoint.")

    # Handle numerical columns. Assuming the instance is valid for the table.
    distance_dict = {}

    # Dropping label column
    col_for_compare = numerical_columns + nominal_columns
    table_without_label = table.copy()
    table_without_label.drop([x for x in table.columns() if x not in col_for_compare])

    if len(numerical_columns):
        if len(nominal_columns):
            table_without_nominal = table_without_label.copy()
            table_without_nominal.drop(nominal_columns)

        for row in range(table_without_label.row_count()):
            calculated_dist = 0
            # Calculating Eucledian Distance
            if len(nominal_columns):
                calculated_dist = calculate_eucledian_distance(instance, table_without_nominal[row], 
                                                                square_root_result=False)
                
                # Handle nominal columns
                unmatch_count = 0
                for col in nominal_columns:
                    if instance[col] != table_without_label[row][col]:
                        unmatch_count += 1
                calculated_dist += unmatch_count

            else:
                calculated_dist = calculate_eucledian_distance(instance, table_without_label[row], 
                                                                square_root_result=False)

            # Assign to distance dictionary
            if calculated_dist not in distance_dict.keys():
                distance_dict[calculated_dist] = [table[row]]
            else:
                distance_dict[calculated_dist] += [table[row]]
    
    # Handle nominal columns when there is no numerical columns
    else:
        for row in range(table_without_label.row_count()):
            unmatch_count = 0
            for col in nominal_columns:
                if instance[col] != table_without_label[row][col]:
                    unmatch_count += 1
            if unmatch_count not in distance_dict.keys():
                distance_dict[unmatch_count] = [table[row]]
            else:
                distance_dict[unmatch_count] += [table[row]]

    # Output handling & Neighbor selection
    distance_dict_sorted = sort_knn_distance_dict(distance_dict)
    return_dict = {}
    unique_distance = []

    for key in distance_dict_sorted:
        if len(set(unique_distance)) != k:
            return_dict[key] = distance_dict[key]
            unique_distance.append(key)
    
    return return_dict


def sort_knn_distance_dict(input, ascending=True):
    """Sorts a dictionary of k-nearest neighbor distances in ascending or descending order 
    based on the value of the 'ascending' parameter.

    Args:
        input (dict): A dictionary of k-nearest neighbor distances.
        ascending (bool): A boolean value indicating whether to sort the dictionary 
            in ascending (True) or descending (False) order.

    Returns:
        dict: A sorted dictionary of k-nearest neighbor distances.
    """
    dict = {}
    if ascending == True:
        sorted_key = sorted(input)
    else:
        sorted_key = sorted(input, reverse=True)

    for item in sorted_key:
        dict[item] = input[item]

    return dict


def calculate_eucledian_distance(base_list, target_list, square_root_result=True):
    """Calculates the Euclidean distance between two lists of numbers.

    Args:
        base_list (DataRow): The first list of numbers.
        target_list (DataRow): The second list of numbers.
        square_root_result (bool): Whether to return the square root of the distance or not.
            Default is True.

    Returns:
        float: The Euclidean distance between the two lists of numbers.

    Notes:
        Assuming two lists are valid for each other, meaning they have the same format
            and share the same column length.
    """
    distance = 0

    for i in range(len(target_list.values())):
        distance += (target_list.values()[i] - base_list.values()[i]) ** 2

    return distance ** 1/2 if square_root_result==True else distance


def majority_vote(instances, scores, labeled_column):
    """Returns the labels in the given instances that occur the most.

    Args:
        instances: A list of instance rows.
        labeled_column: The column holding the class labels.

    Returns: A list of the labels that occur the most in the given
    instances.

    """
    return_list = []
    unique_dict = {}
    for i in instances:
        return_list.append(i[labeled_column])

    for i in return_list:
        if i not in unique_dict.keys():
            unique_dict[i] = 1
        else:
            unique_dict[i] += 1

    max_value = max(unique_dict.values())
    return [key for key, value in unique_dict.items() if value == max_value]


def weighted_vote(instances, scores, labeled_column):
    """Returns the labels in the given instances with the largest total
    sum of corresponding scores.

    Args:
        instances: The list of instance rows.
        scores: The corresponding scores for each instance.
        labeled_column: The column with class labels.

    """
    return_list = []
    unique_dict = {}
    for i in instances:
        return_list.append(i[labeled_column])
    
    for i in range(len(return_list)):
        if return_list[i] not in unique_dict.keys():
            unique_dict[return_list[i]] = scores[i]
        else:
            unique_dict[return_list[i]] += scores[i]

    max_value = max(unique_dict.values())

    return [key for key, value in unique_dict.items() if value == max_value]

