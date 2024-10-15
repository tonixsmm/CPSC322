# Tony Nguyen
# CPSC 322
# 9/5/2023
# I don't have any problems setting up the conda environment.

import random

print("Random Number Program")
print("---------------------")

user_input = input("Enter a number: ")

generated_num = random.randint(1, int(user_input))
user_input = int(user_input)

for i in range(1, generated_num + 1):
    user_input += i

print("Randomly-generated number: " + str(user_input))

if user_input % 2 == 0:
    print("The number is even")
else:
    print("The number is odd")