# Hashes using random projections

from hashtable import HashTable
from hashtable import LSH
import numpy as np

hash_size = 17
input_dim = 10
N = 100000

# f = open('multivariate_normal1.txt', 'r')
# input_dim = 2
f = open('mixed1.txt', 'r')
input_dim = 4
data = [[float(num) for num in line.split()] for line in f]
f.close()


hashtable = HashTable(hash_size, input_dim)

# np.savetxt("randProjections.csv", hashtable.getProjections(), delimiter=",")

data.sort()

for arr in data:
	hashtable.__setitem__(arr, 'added')

hashtable.makeDict()
