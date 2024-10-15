from data_table import DataRow, DataTable
from data_util import *

t1 = DataTable(['a', 'b'])
t1.append([1, 2])
t1.append([2, 3])
t1.append([3, 4])
t1.append([3, 4])
t1.append([4, 5])
t1.append(['', 6])
print(column_values(t1, 'a'))

scatter_plot_with_best_fit(t1, 'a', 'b', 'a', 'b', 'a v. b')

