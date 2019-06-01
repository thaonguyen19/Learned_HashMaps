# hashes using Polynomial hashing and float-modular

from hashtable import PolyHashTable
from hashtable import HashTable
import numpy as np
import random
import math

hash_size = 17
input_dim = 10

# name = 'mixed1'
# input_dim = 4

name = 'multivariate_normal1'
input_dim = 2

f = open(name + '.txt', 'r')

data = [[float(num) for num in line.split()] for line in f]
f.close()

hashtable = PolyHashTable(hash_size, input_dim)


data.sort()
# f = open('data_sorted_' + name + '.txt', "w")
# for x in data:
# 	for y in x:
# 		f.write("%10f " % y)
# 	f.write("\n")
# f.close()

for arr in data:
	hashtable.__setitem__(arr, 'added')

hashtable.makeDict(name)



# lsh = LSH(num_tables, hash_size, input_dim)