from data_eval import *
from data_learn import *
from data_table import *
from data_util import *

import random

table = DataTable(['a', 'b', 'c', 'd'])

for i in range(4):
    table.append([random.randint(0, 4) for x in range(4)])

print(table)

x = -((2/8)*(-2)+(6/8)*(-0.42))
print(x*(8/12))
correlation_heatmap(table)
