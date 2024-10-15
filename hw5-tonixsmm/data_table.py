"""

NAME: Tony Nguyen
DATE: Fall 2023
CLASS: CPSC 322

"""

import csv
import tabulate


class DataRow:
    """A basic representation of a relational table row. The row maintains
    its corresponding column information.

    """
    
    def __init__(self, columns=[], values=[]):
        """Create a row from a list of column names and data values.
           
        Args:
            columns: A list of column names for the row
            values: A list of the corresponding column values.

        Notes: 
            The column names cannot contain duplicates.
            There must be one value for each column.

        """
        # Error checking
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate column names')
        if len(columns) != len(values):
            raise ValueError('mismatched number of columns and values')
        
        # Initialize instance variables
        self.__columns = columns.copy()
        self.__values = values.copy()

        
    def __repr__(self):
        """Returns a string representation of the data row (formatted as a
        table with one row).

        Notes: 
            Uses the tabulate library to pretty-print the row.

        """
        return tabulate.tabulate([self.values()], headers=self.columns())

        
    def __getitem__(self, column):
        """Returns the value of the given column name.
        
        Args:
            column: The name of the column.

        """
        # Error checking
        if column not in self.columns():
            raise IndexError('bad column name')
        
        return self.values()[self.columns().index(column)]


    def __setitem__(self, column, value):
        """Modify the value for a given row column.
        
        Args: 
            column: The column name.
            value: The new value.

        """
        # Error checking
        if column not in self.columns():
            raise IndexError('bad column name')
        
        self.__values[self.columns().index(column)] = value


    def __delitem__(self, column):
        """Removes the given column and corresponding value from the row.

        Args:
            column: The column name.

        """
        # Error checking
        if not column in self.columns():
            raise IndexError('bad column name')
        
        del self.__values[self.columns().index(column)]
        del self.__columns[self.columns().index(column)]

    
    def __eq__(self, other):
        """Returns true if this data row and other data row are equal.

        Args:
            other: The other row to compare this row to.

        Notes:
            Checks that the rows have the same columns and values.

        """
        # For comparing DataRow objects
        if isinstance(other, DataRow):
            if len(self.values()) != len(other.values()):
                raise ValueError("two rows are not equal in length")
            
            if [attr1 for attr1 in self.values()] != [attr2 for attr2 in other.values()]:
                    return False
            return True
        
        # For comparing a DataRow object to a List object since Python does not support overloading
        elif isinstance(other, list):
            if len(self.values()) != len(other):
                raise ValueError("two rows are not equal in length")
            
            if [attr1 for attr1 in self.values()] != [attr2 for attr2 in other]:
                    return False
            return True

    
    def __add__(self, other):
        """Combines the current row with another row into a new row.
        
        Args:
            other: The other row being combined with this one.

        Notes:
            The current and other row cannot share column names.

        """
        # Error checking
        if not isinstance(other, DataRow):
            raise ValueError('expecting DataRow object')
        if len(set(self.columns()).intersection(other.columns())) != 0:
            raise ValueError('overlapping column names')
        
        return DataRow(self.columns() + other.columns(),
                       self.values() + other.values())


    def columns(self):
        """Returns a list of the columns of the row."""
        return self.__columns.copy()


    def values(self, columns=None):
        """Returns a list of the values for the selected columns in the order
        of the column names given.
           
        Args:
            columns: The column values of the row to return. 

        Notes:
            If no columns given, all column values returned.

        """
        # Error checking
        if columns is None:
            return self.__values.copy()
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate column names')
        if set(columns).difference(set(self.columns())):
            raise ValueError('bad column names')
        
        return [self[column] for column in columns]


    def select(self, columns=None):
        """Returns a new data row for the selected columns in the order of the
        column names given.

        Args:
            columns: The column values of the row to include.
        
        Notes:
            If no columns given, all column values included.

        """
        if columns is None:
            return DataRow(self.columns(), self.values())
        return DataRow(columns, self.values(columns))

    
    def copy(self):
        """Returns a copy of the data row."""
        return self.select()

    

class DataTable:
    """A relational table consisting of rows and columns of data.

    Note that data values loaded from a CSV file are automatically
    converted to numeric values.

    """
    
    def __init__(self, columns=[]):
        """Create a new data table with the given column names

        Args:
            columns: A list of column names. 

        Notes:
            Requires unique set of column names. 

        """
        # Error checking
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate column names')
        
        # Initialize instance variables
        self.__columns = columns.copy()
        self.__row_data = []


    def __repr__(self):
        """Return a string representation of the table.
        
        Notes:
            Uses tabulate to pretty print the table.

        """
        return tabulate.tabulate([row.values() for row in self.__row_data], headers=self.columns())

    
    def __getitem__(self, row_index):
        """Returns the row at row_index of the data table.
        
        Notes:
            Makes data tables iterable over their rows.

        """
        return self.__row_data[row_index]
    

    def get_row_data(self):
        """Returns the row data of the data table as a list."""
        data = []

        for i in range(self.row_count()):
            data.append(self[i].values())

        return data
    

    def get_column_data(self, column, include_null=False):
        """Returns the column data of the data table as a list.

        Args:
            column (str): The name of the column to retrieve.

        Returns:
            list: The column data as a list.

        Notes:
            Empty value represented as '' is removed
        """
        data = []

        for i in range(self.row_count()):
            if include_null==True:
                data.append(self[i][column])
            else:
                if self[i][column] != '':
                    data.append(self[i][column])
        return data

    
    def __delitem__(self, row_index):
        """Deletes the row at row_index of the data table.

        """
        # Error checking
        if row_index > len(self.__row_data) - 1:
            raise IndexError('bad index')
        
        del self.__row_data[row_index]

        
    def load(self, filename, delimiter=','):
        """Add rows from given filename with the given column delimiter.

        Args:
            filename: The name of the file to load data from
            delimeter: The column delimiter to use

        Notes:
            Assumes that the header is not part of the given csv file.
            Converts string values to numeric data as appropriate.
            All file rows must have all columns.
        """
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            num_cols = len(self.columns())
            for row in reader:
                row_cols = len(row)                
                if num_cols != row_cols:
                    raise ValueError(f'expecting {num_cols}, found {row_cols}')
                converted_row = []
                for value in row:
                    converted_row.append(DataTable.convert_numeric(value.strip()))
                self.__row_data.append(DataRow(self.columns(), converted_row))

                    
    def save(self, filename, delimiter=','):
        """Saves the current table to the given file.
        
        Args:
            filename: The name of the file to write to.
            delimiter: The column delimiter to use. 

        Notes:
            File is overwritten if already exists. 
            Table header not included in file output.
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter, quotechar='"',
                                quoting=csv.QUOTE_NONNUMERIC)
            for row in self.__row_data:
                writer.writerow(row.values())


    def column_count(self):
        """Returns the number of columns in the data table."""
        return len(self.__columns)


    def row_count(self):
        """Returns the number of rows in the data table."""
        return len(self.__row_data)


    def columns(self):
        """Returns a list of the column names of the data table."""
        return self.__columns.copy()


    def append(self, row_values):
        """Adds a new row to the end of the current table. 

        Args:
            row_values: The row to add as a list of values.
        
        Notes:
            The row must have one value per column. 
        """
        # Error checking
        if not isinstance(row_values, list):
            raise ValueError('expecting List object')
        if len(row_values) != self.column_count():
            raise ValueError('new row_values has an inconsistent attribute count')
        
        self.__row_data.append(DataRow(self.columns(), row_values))

    
    def rows(self, row_indexes):
        """Returns a new data table with the given list of row indexes. 

        Args:
            row_indexes: A list of row indexes to copy into new table.
        
        Notes: 
            New data table has the same column names as current table.

        """
        new_table = DataTable(self.columns())

        try:
            for index in row_indexes:
                new_table.append(self[index].values())            
        except:
            raise IndexError('index is not exist')
        
        return new_table
    

    def drop(self, columns):
        """Removes the given columns from the current table.
        
        Args:
            column: the name of the columns to drop
        """ 
        self.__columns = [item for item in self.columns() if item not in columns]

        for row in self:
            for item in columns:
                del row[item]                        

    
    def copy(self):
        """Returns a copy of the current table."""
        table = DataTable(self.columns())
        for row in self:
            table.append(row.values())
        return table
    

    def update(self, row_index, column, new_value):
        """Changes a column value in a specific row of the current table.

        Args:
            row_index: The index of the row to update.
            column: The name of the column whose value is being updated.
            new_value: The row's new value of the column.

        Notes:
            The row index and column name must be valid. 

        """
        # Error checking
        if row_index >= self.column_count():
            raise IndexError('bad index')
        if column not in self.columns():
            raise IndexError('attribute name not exist')

        self[row_index][column] = new_value

    
    @staticmethod
    def combine(table1, table2, columns=[], non_matches=False):
        """Returns a new data table holding the result of combining table 1 and 2.

        Args:
            table1: First data table to be combined.
            table2: Second data table to be combined.
            columns: List of column names to combine on.
            nonmatches: Include non matches in answer.

        Notes:
            If columns to combine on are empty, performs all combinations.
            Column names to combine are must be in both tables.
            Duplicate column names removed from table2 portion of result.

        """
        # Error checking
        if len(columns):
            if len(set(columns)) != len(columns):
                raise IndexError('duplicate column names')
            for item in columns:
                if item not in table1.columns() or item not in table2.columns():
                    raise IndexError('bad index')
        
        # Create a new table
        new_col = [item for item in table1.columns()]
        col_from_table2 = ([item for item in table2.columns() if item not in new_col])
        new_col.extend(col_from_table2)
        new_table = DataTable(new_col)
        table1_matches = []
        table2_matches = []

        # Combine algorithm
        for row1 in table1:
            row1_key = DataTable.form_keyphrases(row1, columns)
            for row2 in table2:
                row2_key = DataTable.form_keyphrases(row2, columns)
                if row1_key == row2_key:
                    new_row = DataTable.get_item_from_tables(col_from_table2, row1, row2)
                    new_table.append(new_row)

                    table1_matches.append(row1)
                    table2_matches.append(row2)

        # Handle non-matches=True
        if non_matches==True:
            for item in table1:
                if item not in table1_matches:
                    new_row1 = item.values()
                    if len(col_from_table2):
                        for i in range(len(col_from_table2)):
                            new_row1.append('')
                    new_table.append(new_row1)
            
            for item in table2:
                if item not in table2_matches:
                    new_row2 = []
                    for attr in new_col:
                        if attr in item.columns():
                            new_row2.append(item[attr])
                        else:
                            new_row2.append('')

                    new_table.append(new_row2)

        return new_table


    @staticmethod
    def form_keyphrases(row, keys):
        """Returns a list of keyphrases, AKA the combined attribute names

        Args:
            row (DataRow): A DataRow object
            keys (list): List of keys that are used to join

        Returns:
            keyphrases (list): A list of keyphrases
        """
        keyphrases = ''
        for item in keys:
            keyphrases += str(row[item]) + '/'
        keyphrases = keyphrases[:-1]
        return keyphrases
            

    @staticmethod
    def get_item_from_tables(desired_col, original_row, to_join_row):
        """Returns a list of the desired column values

        Args:
            desired_col (list): List of the desired column name of the joined table
            original_row (DataRow): The original row from table 1
            to_join_row (DataRow): The row that is used to join to the original row from table 2

        Returns:
            new_row (list): The list of the joined values
        """
        new_row = original_row.values()
        for item in desired_col:
            if item in to_join_row.columns():
                new_row.append(to_join_row[item])
            else:
                new_row.append('')
        return new_row
    

    @staticmethod
    def convert_numeric(value):
        """Returns a version of value as its corresponding numeric (int or
        float) type as appropriate.

        Args:
            value: The string value to convert

        Notes:
            If value is not a string, the value is returned.
            If value cannot be converted to int or float, it is returned.

         """
        if value==True:
            return 1
        elif value==False:
            return 0
        
        if value.isdigit():
            return int(value)
        
        try:
            return float(value)
        except:
            return value
    
