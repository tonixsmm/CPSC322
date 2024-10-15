"""Data utility functions for CPSC 322 HW-3. 

Basic functions for preprocessing and visualization data sets. 

NAME: Tony Nguyen
DATE: Fall 2023
CLASS: CPSC 322

"""


from data_table import DataTable, DataRow
import matplotlib.pyplot as plt



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
        for column in columns:
            if table[row][column] == '':
                break
            else:
                new_table.append(table[row].values())
                break

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


def mean(value):
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
    # value = [x for x in value if x != '']

    return (sum([(x - mean(value)) ** 2 for x in value]) / len(value)) ** 0.5


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
    plt.bar(bar_names, bar_values, color='b', zorder=3)
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
