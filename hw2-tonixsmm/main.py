from data_table import *

row = DataRow(['a', 'b', 'c'], [1, 2, 3])
print(row.values())
del row['a']
print(row.values())
# print(row)

r1 = DataRow(['a', 'b', 'c'], [10, 20, 30])
r2 = DataRow(['a', 'b', 'c'], [10, 20, 30])
r3 = DataRow(['a', 'd', 'c'], [20, 40, 30])
print(r1 == r2)

# print(r1.values(['a', 'd']))

t1 = DataTable(['x', 'y'])
for i in range(5):
    t1.append([i, i+1])
print(t1)
t2 = t1.rows([0, 1])
print('****')
print(t2)

value = "6.044"
print(value.isdigit())


table_test = DataTable(['x', 'y', 'z'])
table_test.append([1, 20, 100])
table_test.append([3, 40, 200])
table_test.append([2, 30, 300])


print(list((['x', 'y', 'z'] + ['y', 'z', 'u'])))

table_test2 = DataTable(['y', 'z', 'u'])
new_col = [item for item in table_test.columns()]
new_col.extend([item for item in table_test2.columns() if item not in new_col])
print(new_col)

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

t3 = DataTable.combine(t1, t2, [], True)
print(t3)

print(DataTable.convert_numeric(True))