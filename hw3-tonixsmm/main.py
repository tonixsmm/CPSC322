from data_util import *
from data_table import DataTable, DataRow

t1 = DataTable(['x', 'y'])
t1.append([1, 20])
t1.append([3, 40])
t1.append([2, 30])
t2 = DataTable(['y', 'z'])
t2.append([30, 300])
t2.append([20, 100])
t2.append([50, 500])
t2.append([20, 200])
t2.append([60, 600])

t3 = DataTable.combine(t1, t2, ['y'])
print(t3)

t3.drop(['x'])
print(t3)

t1 = DataTable(['a','b','c'])
t1.append([1,1,1])
t1.append([1,'',1])
t1.append([1,3,2])
t1.append([1,'',2])
t1.append([2,4,2])

t2 = replace_missing(t1, 'b', ['a', 'c'], mean)
print(t2)

# g1, s1 = summary_stat_by_column(t2, ['b', 'c'], 'a', max)
# print(g1)
# print(s1)

part = partition(t2, 'a')
print(part, sep='\n')

g2, s2 = frequencies(t2, 'b')
print(g2)
print(s2)