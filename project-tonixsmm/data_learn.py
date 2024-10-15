"""Machine learning algorithm implementations.

NAME: Tony Nguyen
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_util import *
from decision_tree import *

from random import randint
import math


#----------------------------------------------------------------------
# HW-8
#----------------------------------------------------------------------


def random_subset(F, columns):
    """Returns F unique column names from the given list of columns. The
    column names are selected randomly from the given names.

    Args: 
        F (int): The number of columns to return.
        columns (list): The columns to select F column names from.

    Notes: If F is greater or equal to the number of names in columns,
       then the columns list is just returned.

    """
    if F >= len(columns):
        return columns
    else:
        random_list = []
        while len(random_list) < F:
            random_num = randint(0, len(columns) - 1)
            if columns[random_num] not in random_list:
                random_list.append(columns[random_num])
        return random_list
    

def tdidt_F(table, label_col, F, columns): 
    """Returns an initial decision tree for the table using information
    gain, selecting a random subset of size F of the columns for
    attribute selection. If fewer than F columns remain, all columns
    are used in attribute selection.

    Args:
        table: The table to build the decision tree from. 
        label_col: The column containing class labels. 
        F: The number of columns to randomly subselect
        columns: The categorical columns. 

    Notes: The resulting tree can contain multiple leaf nodes for a
        particular attribute value, may have attributes without all
        possible attribute values, and in general is not pruned.

    """
    # Error checking
    if label_col not in table.columns():
        raise ValueError("Label column must be valid for the table.")
    if any(x not in table.columns() for x in columns):
        raise ValueError("Categorical columns must be valid for the table.")

    # Base case
    # 1. Check if table is empty (if so, then return None)
    if table.row_count() == 0:
        return None
    # 2. Check if table instances are all of the same class, return single leaf node (calls same_class() function)
    if same_class(table, label_col):
        return build_leaves(table, label_col)
    # 3. If no more attributes to partition on, return leaves from current partition (calls build_leaves function)
    if len(columns) == 0:
        return build_leaves(table, label_col)
    
    # Build an AttributeNode
    # 4. Choose a random subset of F columns
    random_columns = random_subset(F, columns)

    # 4. Calculate the entropy of the table (e_new)
    entropy = calc_e_new(table, label_col, random_columns)

    # 5. Find the lowest entropy value and corresponding column
    lowest_entropy = sorted(list(entropy.keys()))[0]
    lowest_entropy_col = entropy[lowest_entropy][0]

    # 6. Partition the table on the column
    partitioned_table = partition(table, [lowest_entropy_col])
    node = AttributeNode(lowest_entropy_col, {})
    for table in partitioned_table:
        new_columns = [col for col in columns if col != lowest_entropy_col]
        node.values[table[0][lowest_entropy_col]] = tdidt(table, label_col, new_columns)

    return node


def closest_centroid(centroids, row, columns):
    """Given k centroids and a row, finds the centroid that the row is
    closest to.

    Args:
        centroids (list): The list of rows serving as cluster centroids.
        row (list): The row to find closest centroid to.
        columns: The numerical columns to calculate distance from. 
    
    Returns: The index of the centroid the row is closest to. 

    Notes: Uses Euclidean distance (without the sqrt) and assumes
        there is at least one centroid.

    """
    distance_list = []
        
    for centroid in centroids:
        dist = 0
        for col in columns:
            dist += (centroid[col] - row[col]) ** 2

        distance_list.append(dist)

    for key, value in enumerate(distance_list):
        if value == min(distance_list):
            return key
        

def select_k_random_centroids(table, k):
    """Returns a list of k random rows from the table to serve as initial
    centroids.

    Args: 
        table: The table to select rows from.
        k: The number of rows to select values from.
    
    Returns: k unique rows. 

    Notes: k must be less than or equal to the number of rows in the table. 

    """
    # Error checking
    if k > table.row_count():
        raise ValueError("k must be less than or equal to the number of rows in the table.")
    
    random_list = []

    while len(random_list) < k:
        random_num = randint(0, table.row_count() - 1)
        if table[random_num] not in random_list:
            random_list.append(table[random_num])

    return random_list


def calulate_centroids(table, columns):
    """Returns a list of centroids for the given clusters.

    Args:
        table: A data table serving as the clusters
        columns: The list of numerical columns for determining distances.
    
    Returns: A list of DataRow objects containing centroids for the given clusters.

    """
    return_list = []

    for col in table.columns():
        if col in columns:
            return_list.append(mean(table, col))
        else:
            return_list.append('NA')

    return DataRow(columns=table.columns(), values=return_list)


def compare_two_centroids(centroid1, centroid2, columns):
    """Compare two list of centroids

    Args:
        centroid1 (list): The list of centroids 1
        centroid2 (list): The list of centroids 2
        columns (list): The list of columns to compare
    """
    centroid1_val = []
    centroid2_val = []

    for centroid in centroid1:
        for col in columns:
            centroid1_val.append(centroid[col])

    for centroid in centroid2:
        for col in columns:
            centroid2_val.append(centroid[col])

    return centroid1_val == centroid2_val


def k_means(table, centroids, columns): 
    """Returns k clusters from the table using the initial centroids for
    the given numerical columns.

    Args:
        table: The data table to build the clusters from.
        centroids: Initial centroids to use, where k is length of centroids.
        columns: The numerical columns for calculating distances.

    Returns: A list of k clusters, where each cluster is represented
        as a data table.

    Notes: Assumes length of given centroids is number of clusters k to find.

    """
    # Error checking
    if any(x not in table.columns() for x in columns):
        raise ValueError("Numerical columns must be valid for the table.")

    # Initialize clusters
    clusters = []
    for i in range(len(centroids)):
        clusters.append(DataTable(table.columns()))

    # Assign each row to a cluster
    for row in range(table.row_count()):
        clusters[closest_centroid(centroids, table[row], columns)].append(table[row].values())

    # Update centroids
    new_centroids = []
    for i in range(len(clusters)):
        new_centroids.append(calulate_centroids(clusters[i], columns))

    # Recursively call k_means until centroids don't change
    if not compare_two_centroids(new_centroids, centroids, columns):
        return k_means(table, new_centroids, columns)

    return clusters


def tss(clusters, columns):
    """Return the total sum of squares (tss) for each cluster using the
    given numerical columns.

    Args:
        clusters: A list of data tables serving as the clusters
        columns: The list of numerical columns for determining distances.
    
    Returns: A list of tss scores for each cluster. 

    """
    tss_list = []

    for item in clusters:
        total_dist = 0
        for row in range(item.row_count()):
            dist = 0
            for col in columns:
                dist += (item[row][col] - mean(item, col)) ** 2
            total_dist += dist
        tss_list.append(total_dist)

    return tss_list


#----------------------------------------------------------------------
# HW-7
#----------------------------------------------------------------------


def same_class(table, label_col):
    """Returns true if all of the instances in the table have the same
    labels and false otherwise.

    Args: 
        table: The table with instances to check. 
        label_col: The column with class labels.

    """
    if label_col not in table.columns():
        raise ValueError(f'Label column must be valid for the table.')
    
    # Iterate through the table and check if all the labels are the same
    for i in range(table.row_count()):
        if table[i][label_col] != table[0][label_col]:
            return False
    
    return True


def build_leaves(table, label_col):
    """Builds a list of leaves out of the current table instances.
    
    Args: 
        table: The table to build the leaves out of.
        label_col: The column to use as class labels

    Returns: A list of LeafNode objects, one per label value in the
        table.

    """
    # Error Checking
    if label_col not in table.columns():
        raise ValueError("Label column must be valid for the table.")
    
    # Get unique labels
    unique_label = get_unique_ordered_list(table.get_column_data(label_col))
    leaf_list = []

    # Iterate through the unique labels and build the leaves
    for label in unique_label:
        leaf_list.append(LeafNode(label, table.get_column_data(label_col).count(label), table.row_count()))

    return leaf_list


def _entropy(col, label_col):
    """Returns the entropy of a column using the given label column.

    Args:
        col: The list of column value to compute entropy from.
        label_col: The list of label values to compute entropy from.

    Returns: The e_new of the column.

    Notes: This function assumes both columns are categorical. Col and label_col must have 
        the same length of corresponding values.

    """
    # Get unique labels
    unique_col_values = get_unique_ordered_list(col)
    unique_label = get_unique_ordered_list(label_col)
    result = 0

    # Iterate through the unique labels and calculate the entropy
    for item in unique_col_values:
        entropy = 0
        for label in unique_label:
            count = 0
            for row in range(len(col)):
                if col[row] == item and label_col[row] == label:
                    count += 1

            prob = count / col.count(item)
            if prob != 0:
                entropy += prob * math.log(prob, 2)

        result += -entropy * (col.count(item) / len(col))

    return result


def get_unique_ordered_list(list):
    """Returns a list of unique values in the original list in the same order.

    Args:
        list: The original list.

    Returns: A list of unique values in the original list in the same order.

    """
    unique_list = []
    for item in list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list


def calc_e_new(table, label_col, columns):
    """Returns entropy values for the given table, label column, and
    feature columns (assumed to be categorical). 

    Args:
        table: The table to compute entropy from
        label_col: The label column.
        columns: The categorical columns to use to compute entropy from.

    Returns: A dictionary, e.g., {e1:['a1', 'a2', ...], ...}, where
        each key is an entropy value and each corresponding key-value
        is a list of attributes having the corresponding entropy value. 

    Notes: This function assumes all columns are categorical.

    """
    # Error Checking
    if any(x not in table.columns() for x in columns):
        raise ValueError("Categorical columns must be valid for the table.")
    if label_col not in table.columns():
        raise ValueError("Label column must be valid for the table.")

    # Calculate entropy for each attribute
    entropy_dict = {}
    for col in columns:
        key = _entropy(table.get_column_data(col), table.get_column_data(label_col))
        if key not in entropy_dict.keys():
            entropy_dict[key] = [col]
        else:
            entropy_dict[key] += [col]

    return entropy_dict


def tdidt(table, label_col, columns): 
    """Returns an initial decision tree for the table using information
    gain.

    Args:
        table: The table to build the decision tree from. 
        label_col: The column containing class labels. 
        columns: The categorical columns. 

    Notes: The resulting tree can contain multiple leaf nodes for a
        particular attribute value, may have attributes without all
        possible attribute values, and in general is not pruned.

    """
    # Base case
    # 1. Check if table is empty (if so, then return None)
    if table.row_count() == 0:
        return None
    # 2. Check if table instances are all of the same class, return single leaf node (calls same_class() function)
    if same_class(table, label_col):
        return build_leaves(table, label_col)
    # 3. If no more attributes to partition on, return leaves from current partition (calls build_leaves function)
    if len(columns) == 0:
        return build_leaves(table, label_col)
    
    # Build an AttributeNode
    # 4. Calculate the entropy of the table (e_new)
    entropy = calc_e_new(table, label_col, columns)

    # 5. Find the lowest entropy value and corresponding column
    lowest_entropy = sorted(list(entropy.keys()))[0]
    lowest_entropy_col = entropy[lowest_entropy][0]

    # 6. Partition the table on the column
    partitioned_table = partition(table, [lowest_entropy_col])
    node = AttributeNode(lowest_entropy_col, {})
    for table in partitioned_table:
        new_columns = [col for col in columns if col != lowest_entropy_col]
        node.values[table[0][lowest_entropy_col]] = tdidt(table, label_col, new_columns)

    return node


def summarize_instances(dt_root):
    """Return a summary by class label of the leaf nodes within the given
    decision tree.

    Args: 
        dt_root: The subtree root whose (descendant) leaf nodes are summarized. 

    Returns: A dictionary {label1: count, label2: count, ...} of class
        labels and their corresponding number of instances under the
        given subtree root.

    """
    result_dict = {}

    if type(dt_root) == LeafNode:
        if dt_root.label not in result_dict.keys():
            result_dict[dt_root.label] = dt_root.count
        else:
            result_dict[dt_root.label] += dt_root.count
    elif type(dt_root) == list and all(type(x) == LeafNode for x in dt_root):
        for item in dt_root:
            if item.label not in result_dict.keys():
                result_dict[item.label] = item.count
            else:
                result_dict[item.label] += item.count
    elif type(dt_root) == AttributeNode:
        for value in dt_root.values:
            if type(dt_root.values[value]) == AttributeNode:
                var = summarize_instances(dt_root.values[value])
                result_dict = {key: result_dict.get(key, 0) + var.get(key, 0) for key in set(result_dict) | set(var)}
            else:
                for item in dt_root.values[value]:
                    if type(item) == LeafNode:
                        if item.label not in result_dict.keys():
                            result_dict[item.label] = item.count
                        else:
                            result_dict[item.label] += item.count

    return result_dict


def unpack_attribute_node_values(dt_root):
    """Returns a list of the values of the attribute node.

    Args:
        dt_root: The root of the decision tree to modify.

    Returns: A list of the values of the attribute node.

    """
    # Base Cases:
    # 1. If dt_root is a leaf node, return a copy of dt_root
    if type(dt_root) == LeafNode:
        return [LeafNode(dt_root.label , dt_root.count , dt_root.total)]
    # 2. If dt_root is a list of leaf nodes, return a copy of the list and leaf nodes:
    if type(dt_root) == list and all(type(x) == LeafNode for x in dt_root):
        return [LeafNode(l.label , l.count , l.total) for l in dt_root]

    # Recursive Step:
    # 3. Create a new decision tree attribute node (same name as dt_root)   
    values_list = []
    # 4. Recursively navigate the tree:
    for val, child in dt_root.values.items(): 
        values_list += unpack_attribute_node_values(child)

    return values_list


def resolve_leaf_nodes(dt_root):
    """Modifies the given decision tree by combining attribute values with
    multiple leaf nodes into attribute values with a single leaf node
    (selecting the label with the highest overall count).

    Args:
        dt_root: The root of the decision tree to modify.

    Notes: If an attribute value contains two or more leaf nodes with
        the same count, the first leaf node is used.

    """
    # Base Cases:
    # 1. If dt_root is a leaf node, return a copy of dt_root
    if type(dt_root) == LeafNode:
        return [LeafNode(dt_root.label , dt_root.count , dt_root.total)]
    # 2. If dt_root is a list of leaf nodes, return a copy of the list and leaf nodes:
    if type(dt_root) == list and all(type(x) == LeafNode for x in dt_root):
        return [LeafNode(l.label , l.count , l.total) for l in dt_root]

    # Recursive Step:
    # 3. Create a new decision tree attribute node (same name as dt_root)   
    new_dt_root = AttributeNode(dt_root.name, {})
    # 4. Recursively navigate the tree:
    for val, child in dt_root.values.items(): 
        new_dt_root.values[val] = resolve_leaf_nodes(child)

    # Backtracking Step:
    for key, val in new_dt_root.values.items():
        # Process if the child is an AttributeNode with all leaf nodes
        if isinstance(val, AttributeNode) and all(isinstance(x, LeafNode) for x in val.values.values()):
            # Recursively process the AttributeNode
            val = resolve_leaf_nodes(val)

        # Process if the child is a list of LeafNodes
        if isinstance(val, list) and all(isinstance(x, LeafNode) for x in val):
            # Summarize the instances of the new decision tree
            summarized_instances = summarize_instances(val)

            # Find the label with the highest count (tie resolved by first occurrence)
            highest_count_label = max(summarized_instances, key=summarized_instances.get)
            highest_count = summarized_instances[highest_count_label]

            # Find the total count for the highest count label
            total = sum(leaf.total for leaf in val if leaf.label == highest_count_label)

            # Replace the leaf nodes with a single leaf node with the highest count
            new_dt_root.values[key] = [LeafNode(highest_count_label, highest_count, total)]

    return new_dt_root


def resolve_attribute_values(dt_root, table):
    """Return a modified decision tree by replacing attribute nodes
    having missing attribute values with the corresponding summarized
    descendent leaf nodes.
    
    Args:
        dt_root: The root of the decision tree to modify.
        table: The data table the tree was built from. 

    Notes: The table is only used to obtain the set of unique values
        for attributes represented in the decision tree.

    """
    # Base Cases
    if isinstance(dt_root, LeafNode):
        return LeafNode(dt_root.label, dt_root.count, dt_root.total)

    if isinstance(dt_root, list) and all(isinstance(x, LeafNode) for x in dt_root):
        summarized = summarize_instances(dt_root)
        # Pick the label with the highest count
        label, count = max(summarized.items(), key=lambda item: item[1])
        total = sum(leaf.total for leaf in dt_root if leaf.label == label)
        return [LeafNode(label, count, total)]

    # Recursive Step for AttributeNode
    new_dt_root = AttributeNode(dt_root.name, {})
    for val, child in dt_root.values.items():
        new_dt_root.values[val] = resolve_attribute_values(child, table)

    # Handle Missing Values
    unique_values = set(table.get_column_data(dt_root.name))
    missing_values = unique_values - new_dt_root.values.keys()
    unique_leaf_nodes = []

    if missing_values:
        # Summarize instances for the missing values
        for key, val in new_dt_root.values.items():            
            # Calculate total
            summarized = summarize_instances(new_dt_root)
            total = sum(item for item in summarized.values())

            if len(summarized.keys()) == 1:
                label, count = max(summarized.items(), key=lambda item: item[1])
                new_dt_root.values[key] = [LeafNode(label, count, total)]
            else:
                if isinstance(new_dt_root.values[key], AttributeNode):
                    new_dt_root.values[key] = resolve_attribute_values(new_dt_root.values[key], table)
                else:
                    new_dt_root.values[key] = [LeafNode(new_dt_root.values[key][0].label, new_dt_root.values[key][0].count, total)]

        # Check if all branches lead to the same leaf node
        for nodes in new_dt_root.values.values():
            if isinstance(nodes, list) and all(isinstance(x, LeafNode) for x in nodes):
                unique_leaf_nodes += nodes

        if check_similar_leaf_nodes(unique_leaf_nodes):
            return [unique_leaf_nodes[0]]
    
    if (
        isinstance(dt_root, AttributeNode)
        and isinstance(list(dt_root.values.values())[0], list)
        and all(isinstance(x, LeafNode) for x in list(dt_root.values.values())[0])
    ):  
        check_result = show_unique_leaf_nodes(unique_leaf_nodes)
        if len(check_result):
            return check_result
    return new_dt_root


def tdidt_predict(dt_root, instance): 
    """Returns the class for the given instance given the decision tree. 

    Args:
        dt_root: The root node of the decision tree. 
        instance: The instance to classify. 

    Returns: A pair consisting of the predicted label and the
       corresponding percent value of the leaf node.

    Note: Assume the node issues are resolved.
    """
    # Base Cases
    if isinstance(dt_root, LeafNode):
        return (dt_root.label, dt_root.count / dt_root.total * 100)
    
    if isinstance(dt_root, list) and all(isinstance(x, LeafNode) for x in dt_root):
        summarized = summarize_instances(dt_root)
        # Pick the label with the highest count
        label, count = max(summarized.items(), key=lambda item: item[1])
        total = sum(leaf.total for leaf in dt_root if leaf.label == label)
        return (label, count / total * 100)
    
    # Recursive Step for AttributeNode
    for val, child in dt_root.values.items():
        if instance[dt_root.name] == val:
            return tdidt_predict(child, instance)


def check_similar_leaf_nodes(leaf_list):
    """Checks if two leaf nodes are similar.

    Args:
        leaf_list (list): A list of leaf nodes.

    Returns:
        bool: True if the two leaf nodes are similar, False otherwise.
    """
    label = None
    count = None
    total = None

    for leaf in leaf_list:
        if label is None:
            label = leaf.label
            count = leaf.count
            total = leaf.total
        else:
            if label != leaf.label or count != leaf.count or total != leaf.total:
                return False
    return True


def show_unique_leaf_nodes(leaf_list):
    """Shows the unique leaf nodes in a list of leaf nodes.

    Args:
        leaf_list (list): A list of leaf nodes.

    Returns:
        list: A list of unique leaf nodes.
    """
    if not len(leaf_list):
        return []
    
    unique_leaf_nodes = []
    key = []
    if isinstance(leaf_list, list) and all(isinstance(x, LeafNode) for x in leaf_list):
        key.append(str(leaf_list[0].label) + '/' + str(leaf_list[0].count) + '/' + str(leaf_list[0].total))
        unique_leaf_nodes.append(leaf_list[0])
        for nodes in leaf_list:
            sample = str(nodes.label) + '/' + str(nodes.count) + '/' + str(nodes.total)
            if sample not in key:
                unique_leaf_nodes.append(nodes)
                key.append(sample)
    
    return unique_leaf_nodes


#----------------------------------------------------------------------
# HW-6
#----------------------------------------------------------------------

def naive_bayes(table, instance, label_col, continuous_cols, categorical_cols=[]):
    """Returns the labels with the highest probabibility for the instance
    given table of instances using the naive bayes algorithm.

    Args:
       table: A data table of instances to use for estimating most probably labels.
       instance: The instance to classify.
       continuous_cols: The continuous columns to use in the estimation.
       categorical_cols: The categorical columns to use in the estimation. 

    Returns: A pair (labels, prob) consisting of a list of the labels
        with the highest probability and the corresponding highest
        probability.

    """
    # Error Checking
    if any(x not in table.columns() for x in continuous_cols):
        raise ValueError("Numerical columns must be valid for the table.")
    
    if not len(categorical_cols) and any(x not in table.columns() for x in categorical_cols):
        raise ValueError("Nominal columns must be valid for the table.")
    
    if len(set(continuous_cols).intersection(set(categorical_cols))) > 0:
        raise ValueError("Numerical and nominal columns must be disjoint.")
    
    if label_col not in table.columns():
        raise ValueError("Label column must be valid for the table.")
    
    # Get unique labels
    unique_label = list(set(table.get_column_data(label_col)))
    
    # Handle continuous columns
    result_dict = {}

    for label in unique_label: # for yes/no labels, should have run twice
        filtered_by_label = filter(table, [label_col], [label])
        prob = 1

        for col in continuous_cols:
            # The mean and stdev should be calculated from the filtered data
            prob *= gaussian_density(instance[col], mean(filtered_by_label, col), std_dev(filtered_by_label, col))
        for col in categorical_cols:
            prob *= categorical_probabilities(filtered_by_label, col, instance[col])

        # Factor in how many times a label occurs
        prob *= categorical_probabilities(table, label_col, label)

        # Assign to result dictionary
        if prob not in result_dict.keys():
            result_dict[prob] = [label]
        else:
            result_dict[prob] += [label]

    # Voting scheme
    largest_prob = max(result_dict, key=float)
    largest_prob_label = result_dict[largest_prob]

    return (largest_prob_label, largest_prob)


def categorical_probabilities(table, col, value):
    """Calculates the probability of a categorical value in a given column of a table.

    Args:
        table (DataTable): a Table object
        col (str): a string representing the column name
        value: a categorical value to compare

    Returns:
        The probability of the categorical value in the column
    """
    count  = 0
    for item in table.get_column_data(col):
        if item == value:
            count += 1

    return count / len(table.get_column_data(col))


def gaussian_density(x, mean, sdev):
    """Return the probability of an x value given the mean and standard
    deviation assuming a normal distribution.

    Args:
        x: The value to estimate the probability of.
        mean: The mean of the distribution.
        sdev: The standard deviation of the distribution.

    """
    first, second = 0, 0 
    
    if sdev > 0:
        first = 1 / (math.sqrt(2 * math.pi) * sdev)
        second = math.e ** (-((x - mean) ** 2) / (2 * (sdev ** 2))) 

    return first * second


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
