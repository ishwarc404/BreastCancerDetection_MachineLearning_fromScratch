import itertools

a = [1,2,3,4,5]
b = [7,8,9,10,11]

for i in itertools.product(a,b):
    print(i)