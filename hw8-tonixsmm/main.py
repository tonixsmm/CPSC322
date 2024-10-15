from data_table import *
from data_learn import *
from data_eval import *
from data_util import *

table = DataTable(['a', 'b', 'c'])
table.append([1, 1, 'y'])
table.append([1, 2, 'n'])
table.append([2, 1, 'y'])

test = DataTable(['a', 'b', 'c'])
test.append([2, 2, 'n'])
test.append([1, 1, 'n'])

tree = tdidt_eval(table, test, 'c', ['a', 'b'])
print(tree)