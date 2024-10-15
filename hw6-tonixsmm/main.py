from data_table import *
from data_learn import *
from data_eval import *
from data_util import *

# Define the training and test data
train = DataTable(['a', 'b', 'label'])
# train.append([1, 2, 1])
# train.append([2, 4, 1])
# train.append([1, 3, 2])
# train.append([4, 3, 2])
# train.append([3, 1, 3])
train.append([1, 5, 'yes'])
train.append([2, 6, 'yes'])
train.append([1, 5, 'no'])
train.append([1, 5, 'no'])
train.append([1, 6, 'yes'])
train.append([2, 6, 'no'])
train.append([1, 5, 'yes'])
train.append([1, 6, 'yes'])

test = DataTable(['a', 'b', 'label'])
# test.append([2, 2, 1])
# test.append([4, 4, 2])
# test.append([1, 1, 3])
test.append([1, 5, 'yes'])
test.append([2, 5, 'yes'])
test.append([2, 5, 'no'])
test.append([1, 6, 'no'])

table = DataTable(['a', 'b', 'c'])
for i in range(17):
    table.append([0, (i % 3) + 1, 0])
# print(train)
# print(test)

# result = naive_bayes_eval(train, test, 'label', [], ['a', 'b'])
result = naive_bayes_stratified(train, 2, 'label', [], ['a', 'b'])
print(result)