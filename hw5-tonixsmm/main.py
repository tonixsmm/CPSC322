from data_table import *
from data_learn import *
from data_eval import *
from data_util import *

# Define the training and test data
train = DataTable(['a', 'b', 'label'])
train.append([1, 2, 1])
train.append([2, 4, 1])
train.append([1, 3, 2])
train.append([4, 3, 2])
train.append([3, 1, 3])

test = DataTable(['a', 'b', 'label'])
test.append([2, 2, 1])
test.append([4, 4, 2])
test.append([1, 1, 3])

# for i in range(test.row_count()):
#     knn_result = knn(train, test[i], 3, ['a', 'b'], [])
#     print(knn_result)
# knn_result = knn(train, test[2], 3, ['a', 'b'], [])
# print(knn_result)

result = knn_eval(train, test, weighted_vote, 3, 'label', ['a', 'b'], [])
print(result)

matrix = DataTable(['actual', 1, 2])
matrix.append([1, 10, 2])
matrix.append([2, 4, 12])

print(accuracy(matrix, 1))
# print(accuracy(matrix, 1))