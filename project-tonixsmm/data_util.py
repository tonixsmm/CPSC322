"""Data utility functions.

NAME: Tony Nguyen
DATE: Fall 2023
CLASS: CPSC 322

"""

from math import sqrt

from data_table import DataTable, DataRow
import matplotlib.pyplot as plt


def correlation_heatmap(table):
    """Return a heatmap consists of the pairwise Pearson correlation coefficients of columns.

    Args:
        table (DataTable): The table to create a heatmap of.
    """
    corr = []

    for col in table.columns():
        corr.append([correlation_coefficient(table, col, col2) for col2 in table.columns()])

    # Create the heatmap
    plt.figure()
    fig, ax = plt.subplots()
    plt.imshow(corr, cmap='Blues', interpolation='nearest')

    # Formatting
    ax.set_xticks(range(len(table.columns())), table.columns())
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(len(table.columns())), table.columns())

    plt.colorbar()
    plt.show()


def filter(table, filter_col, filter_val):
    """Filters a DataTable based on the values in the specified columns.

    Args:
        table (DataTable): The table to filter.
        filter_col (list): A list of column indices to filter on.
        filter_val (list): A list of values to filter for in the corresponding columns.

    Returns:
        DataTable: A new DataTable containing only the rows that match the filter criteria.
    """
    returned_table = DataTable(table.columns())

    for col in range(len(filter_col)):
        for item in range(table.row_count()):
            if table[item][filter_col[col]] == filter_val[col]:
                returned_table.append(table[item].values())
    
    return returned_table


#----------------------------------------------------------------------
# HW5
#----------------------------------------------------------------------


def normalize(table, column):
    """Normalize the values in the given column of the table. This
    function modifies the table.

    Args:
        table: The table to normalize.
        column: List of column label in the table to normalize.

    """
    for col in column:
        if col not in table.columns():
            raise ValueError(f'Column {col} not in table')

        min_num = min(column_values(table, col))
        max_num = max(column_values(table, col))

        for row in range(table.row_count()):
            table[row][col] = (table[row][col] - min_num) / (max_num - min_num)


def discretize(table, column, cut_points):
    """Discretize column values according to the given list of n-1
    cut_points to form n ordinal values from 1 to n. This function
    modifies the table.

    Args:
        table: The table to discretize.
        column (str): The column in the table to discretize.
        cut_points (list): The list of cut points to use for

    """
    for row in table:
        for i in range(len(cut_points)):
            if row[column] < cut_points[i]:
                row[column] = i + 1
                break
        else:
            row[column] = len(cut_points) + 1


#----------------------------------------------------------------------
# HW4
#----------------------------------------------------------------------


def column_values(table, column):
    """Returns a list of the values (in order) in the given column.

    Args:
        table: The data table that values are drawn from
        column: The column whose values are returned
    
    """
    return DataTable.get_column_data(table, column, include_null=True)


def mean(table, column):
    """Returns the arithmetic mean of the values in the given table
    column.

    Args:
        table: The data table that values are drawn from
        column: The column to compute the mean from

    Notes: 
        Assumes there are no missing values in the column.

    """
    return _mean(DataTable.get_column_data(table, column))


def variance(table, column):
    """Returns the variance of the values in the given table column.

    Args:
        table: The data table that values are drawn from
        column: The column to compute the variance from

    Notes:
        Assumes there are no missing values in the column.

    """
    value = DataTable.get_column_data(table, column)

    return sum([(x - mean(table, column)) ** 2 for x in value]) / len(value)


def std_dev(table, column):
    """Returns the standard deviation of the values in the given table
    column.

    Args:
        table: The data table that values are drawn from
        column: The colume to compute the standard deviation from

    Notes:
        Assumes there are no missing values in the column.

    """
    return sqrt(variance(table, column))



def covariance(table, x_column, y_column):
    """Returns the covariance of the values in the given table columns.
    
    Args:
        table: The data table that values are drawn from
        x_column: The column with the "x-values"
        y_column: The column with the "y-values"

    Notes:
        Assumes there are no missing values in the columns.        

    """
    xval = DataTable.get_column_data(table, x_column)
    yval = DataTable.get_column_data(table, y_column)

    return sum([(x - mean(table, x_column)) * (y - mean(table, y_column)) for x, y
                in zip(xval, yval)]) / len(xval)


def linear_regression(table, x_column, y_column):
    """Returns a pair (slope, intercept) resulting from the ordinary least
    squares linear regression of the values in the given table columns.

    Args:
        table: The data table that values are drawn from
        x_column: The column with the "x values"
        y_column: The column with the "y values"

    """
    mean_x = mean(table, x_column)
    mean_y = mean(table, y_column)

    slope = covariance(table, x_column, y_column) / variance(table, x_column)

    intercept = mean_y - slope * mean_x

    return slope, intercept

def correlation_coefficient(table, x_column, y_column):
    """Return the correlation coefficient of the table's given x and y
    columns.

    Args:
        table: The data table that value are drawn from
        x_column: The column with the "x values"
        y_column: The column with the "y values"

    Notes:
        Assumes there are no missing values in the columns.        

    """
    try:
        result =  covariance(table, x_column, y_column) / (std_dev(table, x_column) * std_dev(table, y_column))
    except ZeroDivisionError:
        result = -1

    return result


def frequency_of_range(table, column, start, end):
    """Return the number of instances of column values such that each
    instance counted has a column value greater or equal to start and
    less than end. 
    
    Args:
        table: The data table used to get column values from
        column: The column to bin
        start: The starting value of the range
        end: The ending value of the range

    Notes:
        start must be less than end

    """
    if start > end:
        raise ValueError("Start must be less than end")

    count = 0

    for row in range(table.row_count()):
        if start <= table[row][column] < end:
            count += 1

    return count


def create_dict_of_ranges(table, column, desired_key, key_condition):
    """Given a table, column, a list of desired keys, and a list of key conditions,
    returns a dictionary where each key is a desired key and the corresponding value
    is the frequency of the range specified by the key condition in the given column of the table.

    Args:
        table: The data table used to get column values from
        column: The column to bin
        desired_key: The desired keys
        key_condition: The key conditions

    Notes:
        desired_key and key_condition must have the same length
    """
    dict = {}

    # Preprocess the key item
    def key_preprocess(item):
        """Preprocesses a key item by parsing it into a tuple of minimum and maximum values.
        
        Args:
            item (str): The key item to preprocess.
        
        Returns:
            tuple: A tuple of minimum and maximum values parsed from the key item.

        Notes:
            Assuming column values are integers.
        """
        if '-' in item:
            temp = item.split('-')
            min = int(temp[0])
            max = int(temp[1])
        elif item[:2] == '<=':
            temp = item.split('<=')
            max = int(item[2:])
            min = None
        elif item[:2] == '>=':
            temp = item.split('>=')
            min = int(item[2:])
            max = None
        else:
            min = int(item)
            max = int(item) + .5
        return min, max

    # Create the dictionary containing the key and frequency
    for key in range(len(desired_key)):
        item = key_condition[key]
        min_val, max_val = key_preprocess(item)
        if min_val is None:
            min_val = min(column_values(table, column))
            if min_val > max_val:
                min_val = max_val
                max_val += 0.5
        if max_val is None:
            max_val = max(column_values(table, column))
            if max_val < min_val:
                max_val = min_val
                min_val -= 0.5
        dict[desired_key[key]] = frequency_of_range(table, column, min_val, max_val)

    return dict


def histogram(table, column, nbins, xlabel, ylabel, title, filename=None):
    """Create an equal-width histogram of the given table column and number of bins.
    
    Args:
        table: The data table to use
        column: The column to obtain the value distribution
        nbins: The number of equal-width bins to use
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # Reset the dataset
    plt.figure()

    # Plot the histogram
    plt.grid(axis='y', color='0.85', zorder=0)
    plt.hist(column_values(table, column), bins=nbins, color='b', alpha = 0.5, rwidth=0.8, zorder=3)

    # Formatting
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Close the plot
    plt.show() if filename is None else plt.savefig(filename, format='svg')
    plt.close()
    

def scatter_plot_with_best_fit(table, xcolumn, ycolumn, xlabel, ylabel, title, filename=None):
    """Create a scatter plot from given values that includes the "best fit" line.
    
    Args:
        table: The data table to use
        xcolumn: The column for x-values
        ycolumn: The column for y-values
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    x_col = column_values(table, xcolumn)
    y_col = column_values(table, ycolumn)

    # Reset the dataset
    plt.figure()

    # Plot the scatter plot
    plt.grid(axis='y', color='0.85', zorder=0)
    plt.scatter(x_col, y_col, alpha=0.3, color='b', zorder=3)

    # Plot the middle point of the distribution
    mean_x = mean(table, xcolumn)
    mean_y = mean(table, ycolumn)
    plt.hlines(mean_y, min(x_col), max(x_col), alpha=0.3, label=r'$\bar{x}$', linestyles='--', color='r', zorder=3)
    plt.vlines(mean_x, min(y_col), max(y_col), alpha=0.3, label=r'$\bar{y}$', linestyles='--', color='r', zorder=3)

    # Plot the best fit line
    slope, intercept = linear_regression(table, xcolumn, ycolumn)
    plt.plot([min(x_col), max(x_col)], [slope * min(x_col) + intercept, slope * max(x_col) + intercept], 
            label = 'Best fit line', color='g', zorder=3)
    
    # Formatting
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.text(min(x_col), max(y_col), r'$r = $' + str(round(correlation_coefficient(table, xcolumn, ycolumn), 3)),
            fontsize=9, color='g')

    # Close the plot
    plt.show() if filename is None else plt.savefig(filename, format='svg')
    plt.close()

    
#----------------------------------------------------------------------
# HW3
#----------------------------------------------------------------------


def distinct_values(table, column):
    """Return the unique values in the given column of the table.
    
    Args:
        table: The data table used to get column values from.
        column: The column of the table to get values from.

    Notes:
        Returns a list of unique values
    """
    return_list = []

    for row in range(table.row_count()):
        if table[row][column] not in return_list:
            return_list.append(table[row][column])
    
    return return_list


def show_missing(table, columns):
    """Returns a new DataTable object containing only the rows from the input table where the specified columns have missing values.
    
    Args:
        table (DataTable): the input DataTable object
        columns (list): a list of column names to check for missing values
    
    Returns:
        new_table (DataTable): a new DataTable object containing only the rows from the input table where the specified columns have missing values
    """
    # Error Checking
    for item in columns:
        if item not in table.columns():
            raise ValueError("Column not in table")
    
    # Create a new DataTable object
    new_table = DataTable(table.columns())

    for row in range(table.row_count()):
        for column in columns:
            if table[row][column] == '':
                new_table.append(table[row].values())
            else:
                break

    return new_table


def remove_missing(table, columns):
    """Return a new data table with rows from the given one without
    missing values.

    Args:
        table: The data table used to create the new table.
        columns: The column names to check for missing values.

    Returns:
        A new data table.

    Notes: 
        Columns must be valid.

    """
    # Error Checking
    for item in columns:
        if item not in table.columns():
            raise ValueError("Column not in table")
        
    new_table = DataTable(table.columns())

    for row in range(table.row_count()):
        remove = False
        for column in columns:
            if table[row][column] == "":
                remove = True
                break        
        if not remove:
            new_table.append(table[row].values())

    return new_table


def duplicate_instances(table):
    """Returns a table containing duplicate instances from original table.
    
    Args:
        table: The original data table to check for duplicate instances.

    """
    new_table = DataTable(table.columns())

    for row in range(table.row_count()):
        for row2 in range(row + 1, table.row_count()):
            if table[row].values() == table[row2].values():
                if new_table.row_count() == 0:
                    new_table.append(table[row].values())
                else:
                    if table[row].values() not in new_table.get_row_data():
                        new_table.append(table[row].values())
                        break

    return new_table

                    
def remove_duplicates(table):
    """Remove duplicate instances from the given table.
    
    Args:
        table: The data table to remove duplicate rows from

    """
    new_table = table.copy()

    duplicated_table = duplicate_instances(new_table)

    for row in range(duplicated_table.row_count()): 
        for row2 in range(new_table.row_count()):
            if duplicated_table[row].values() == new_table[row2].values():
                del new_table[row2]
                break

    return new_table


def partition(table, columns):
    """Partition the given table into a list containing one table per
    corresponding values in the grouping columns.
    
    Args:
        table: the table to partition
        columns: the columns to partition the table on

    Notes: It is a set of data tables, each is a set that has the same values for 
        the given columns. aka pandas groupby
    """
    return_list = []
    key_list = []

    if table.row_count() == 0:
        return return_list

    for item in columns:
        if item not in table.columns():
            raise ValueError("Column not in table")

    for row in range(table.row_count()):
        key = ""
        for column in range(len(columns)):
            key += str(table[row][columns[column]]) + '/'
            key = key[:-1]
        
        if key not in key_list:
            key_list.append(key)
            return_list.append(DataTable(table.columns()))

        for item in range(len(return_list)):
            if key == key_list[item]:
                return_list[item].append(table[row].values())
                break

    return return_list


def _mean(value):
    """Return the mean of the given non-empty list of values.

    Args:
        value: the list of values to compute the mean of
    """
    if len(value) == 0:
        return ''

    return sum(value) / len(value)


def median(value):
    """Return the median of the given non-empty list of values.

    Args:
        value: the list of values to compute the median of
    """
    # value = [x for x in value if x != '']

    value.sort()
    if len(value) % 2 == 0:
        return (value[len(value) // 2] + value[(len(value) // 2) - 1]) / 2
    else:
        return value[len(value) // 2]
    

def mode(value):
    """Return the mode of the given non-empty list of values.

    Args:
        value: the list of values to compute the mode of
    """
    return max(set(value), key=value.count)


def stdev(value):
    """Return the standard deviation of the given non-empty list of values.

    Args:
        value: the list of values to compute the standard deviation of
    """
    value = [x for x in value if x != '']

    return (sum([(x - _mean(value)) ** 2 for x in value]) / len(value)) ** 0.5


def count(value):
    """Return the number of non-empty values in the given list of values.

    Args:
        value: the list of values to compute the count of
    """
    return len([x for x in value if x != ''])


def summary_stat(table, column, function):
    """Return the result of applying the given function to a list of
    non-empty column values in the given table.

    Args:
        table: the table to compute summary stats over
        column: the column in the table to compute the statistic
        function: the function to compute the summary statistic

    Notes: 
        The given function must take a list of values, return a single
        value, and ignore empty values (denoted by the empty string)

    """
    value = table.get_column_data(column)

    if len(value) == 0:
        return None
    return function(value)


def replace_missing(table, column, partition_columns, function): 
    """Replace missing values in a given table's column using the provided
    function over similar instances, where similar instances are
    those with the same values for the given partition columns.

    Args:
        table: the table to replace missing values for
        column: the coumn whose missing values are to be replaced
        partition_columns: for finding similar values
        function: function that selects value to use for missing value

    Notes: 
        Assumes there is at least one instance with a non-empty value
        in each partition

    """
    new_table = table.copy()

    partitioned_list = partition(new_table, partition_columns)

    for row in range(new_table.row_count()):
        if new_table[row][column] == '':
            # Start finding the match partition
            for item in partitioned_list: # item is a DataTable
                if [item[0][i] for i in partition_columns] == [new_table[row][i] for i in partition_columns]:
                    if type(new_table[0][column]) == int:
                        new_table[row][column] = int(function(item.get_column_data(column)))
                    elif type(new_table[0][column]) == float:
                        new_table[row][column] = (function(item.get_column_data(column)))
                    break

    return new_table


def summary_stat_by_column(table, partition_column, stat_column, function):
    """Returns for each partition column value the result of the statistic
    function over the given statistics column.

    Args:
        table: the table for computing the result
        partition_column: the column used to create groups from
        stat_column: the column to compute the statistic over
        function: the statistic function to apply to the stat column

    Notes:
        Returns a list of the groups and a list of the corresponding
        statistic for that group.

    """
    if type(partition_column) != list:
        partition_column = [partition_column]

    partitioned_list = partition(table, partition_column)
    stat_list = []
    returned_part_list = []

    # Compute stats
    for item in partitioned_list:
        stat_list.append(function(item.get_column_data(stat_column)))

    # Get the partition column values
    for item in partitioned_list:
        temp_list = []
        for i in partition_column:
            temp_list.append(item[0][i])
        
        if len(temp_list) == 1:
            returned_part_list.append(temp_list[0])
        else:
            returned_part_list.append(temp_list)

    return returned_part_list, stat_list


def frequencies(table, partition_column):
    """Returns for each partition column value the number of instances
    having that value.

    Args:
        table: the table for computing the result
        partition_column: the column used to create groups

    Notes:

        Returns a list of the groups and a list of the corresponding
        instance count for that group.

    """
    if type(partition_column) != list:
        partition_column = [partition_column]

    partitioned_list = partition(table, partition_column)
    freq_list = []
    returned_part_list = []

    # Compute stats
    for item in partitioned_list:
        freq_list.append(item.row_count())

    # Get the partition column values
    for item in partitioned_list:
        temp_list = []
        for i in partition_column:
            temp_list.append(item[0][i])
        
        if len(temp_list) == 1:
            returned_part_list.append(temp_list[0])
        else:
            returned_part_list.append(temp_list)

    return returned_part_list, freq_list


def dot_chart(xvalues, xlabel, title, filename=None):
    """Create a dot chart from given values.
    
    Args:
        xvalues: The values to display
        xlabel: The label of the x axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # reset figure
    plt.figure()
    # dummy y values
    yvalues = [1] * len(xvalues)
    # create an x-axis grid
    plt.grid(axis='x', color='0.85', zorder=0)
    # create the dot chart (with pcts)
    plt.plot(xvalues, yvalues, 'b.', alpha=0.2, markersize=16, zorder=3)
    # get rid of the y axis
    plt.gca().get_yaxis().set_visible(False)
    # assign the axis labels and title
    plt.xlabel(xlabel)
    plt.title(title)
    # save as file or just show
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()

    
def pie_chart(values, labels, title, filename=None):
    """Create a pie chart from given values.
    
    Args:
        values: The values to display
        labels: The label to use for each value
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    plt.figure()
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)

    plt.grid(axis='x', color='0.85', zorder=0)
    plt.title(title)
    
    # save as file or just show
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    
    # close the plot
    plt.close()


def bar_chart(bar_values, bar_names, xlabel, ylabel, title, filename=None):
    """Create a bar chart from given values.
    
    Args:
        bar_values: The values used for each bar
        bar_names: The label for each bar value
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    plt.figure()
    plt.bar(bar_names, bar_values, alpha=0.5, color='b', zorder=3)
    plt.grid(axis='y', color='0.85', zorder=0)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()

    plt.close()

    
def scatter_plot(xvalues, yvalues, xlabel, ylabel, title, filename=None):
    """Create a scatter plot from given values.
    
    Args:
        xvalues: The x values to plot
        yvalues: The y values to plot
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    plt.figure()
    plt.scatter(xvalues, yvalues, color='b', zorder=3)
    plt.grid(axis='both', color='0.85', zorder=0)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()

    plt.close()


def box_plot(distributions, labels, xlabel, ylabel, title, filename=None):
    """Create a box and whisker plot from given values.
    
    Args:
        distributions: The distribution for each box
        labels: The label of each corresponding box
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    plt.figure()

    plt.grid(axis='y', color='0.85', zorder=0) # create the box plot
    plt.boxplot(distributions, zorder=3)
    
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel)
    plt.title(title) # display plot

    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    
    plt.close()


    
