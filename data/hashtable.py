# Code adapted from Santho Shhari
# https://santhoshhari.github.io/Locality-Sensitive-Hashing/

import numpy as np
import random
import math

class PolyHashTable:
    def __init__(self, hash_size, inp_dimensions):
        random.seed(137)
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.kLargePrime = (2 ** 17) - 1
        self.coef = random.randint(0, self.kLargePrime)

    def generate_hash(self, inp_vector):
        num = 0.0
        for i in range(self.inp_dimensions):
            num += ((self.coef ** (self.inp_dimensions - i - 1))*inp_vector[i])
        return str(int(math.fmod(abs(num), self.kLargePrime)))

    def __setitem__(self, inp_vec, label='added'):
        hash_value = self.generate_hash(inp_vec)
        if hash_value in self.hash_table:
            self.hash_table[hash_value] = self.hash_table[hash_value] + 1
        else:
            self.hash_table[hash_value] = 1

    def makeDict(self, name):
        f = open('dict_poly_' + name + '.txt', "w")
        maxKey = 0
        maxKeyNum = 0
        for key in self.hash_table.keys():
            f.write(str(key) + ' : ' + str(self.hash_table[key]) + '\n')
            if maxKeyNum < self.hash_table[key]:
                maxKey = key
                maxKeyNum = self.hash_table[key]
        f.close()
        print(str(maxKey) + ' ' + str(maxKeyNum))
        
    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        if hash_value in self.hash_table:
            return self.hash_table[hash_value]
        else:
            return 0

# class PolyHashTableMult:
#     def __init__(self, hash_size, inp_dimensions, poly_dim):
#         self.hash_size = hash_size
#         self.inp_dimensions = inp_dimensions
#         self.hash_table = dict()
#         self.poly_dim = poly_dim
#         self.kLargePrime = (2 ** int(17 / inp_dimensions + 3)) - 1

#     def polyHash(self, key):
#         coef = [0 for i in range(self.poly_dim)]
#         for i in range(self.poly_dim):
#             coef[i] = self.randFieldElem(i)
#         num = 0.0
#         for i in range(self.poly_dim):
#             num += ((key **(self.poly_dim - i - 1))*coef[i])
#         return int(math.fmod(abs(num), self.kLargePrime))

#     def generate_hash(self, inp_vector):
#         nums = [self.polyHash(inp_vector[i]) for i in range(self.inp_dimensions)]
#         hashes = (np.asarray(nums)).astype('int')
#         return ','.join(hashes.astype('str'))

#     def randFieldElem(self, f):
#         random.seed(f + 137)
#         return random.randint(0, self.kLargePrime)

#     def __setitem__(self, inp_vec, label='added'):
#         hash_value = self.generate_hash(inp_vec)
#         if hash_value in self.hash_table:
#             self.hash_table[hash_value] = self.hash_table[hash_value] + 1
#         else:
#             self.hash_table[hash_value] = 1
#         # self.hash_table[hash_value] = self.hash_table\
#             # .get(hash_value, list()) + [label]

#     def makeDict(self, name):
#         f = open('dict_' + name + '.txt', "w")
#         maxKey = 0
#         maxKeyNum = 0
#         for key in self.hash_table.keys():
#             f.write(str(key) + ' : ' + str(self.hash_table[key]) + '\n')
#             if maxKeyNum < self.hash_table[key]:
#                 maxKey = key
#                 maxKeyNum = self.hash_table[key]
#         f.close()
#         print(str(maxKey) + ' ' + str(maxKeyNum))
        
#     def __getitem__(self, inp_vec):
#         hash_value = self.generate_hash(inp_vec)
#         if hash_value in self.hash_table:
#             return self.hash_table[hash_value]
#         else:
#             return 0
    
class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = np.random.randn(self.hash_size, inp_dimensions)
        
    def generate_hash(self, inp_vector):
        nums = np.dot(inp_vector, self.projections.T) 
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return ''.join(bools.astype('str'))

    def __setitem__(self, inp_vec, label='added'):
        hash_value = self.generate_hash(inp_vec)
        if hash_value in self.hash_table:
            self.hash_table[hash_value] = self.hash_table[hash_value] + 1
        else:
            self.hash_table[hash_value] = 1
        # self.hash_table[hash_value] = self.hash_table\
            # .get(hash_value, list()) + [label]

    def makeDict(self, name):
        f = open('dict_' + name + '.txt', "w")
        for key in self.hash_table.keys():
            f.write(str(key) + ' : ' + str(self.hash_table[key]) + '\n')
        f.close()
        
    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        if hash_value in self.hash_table:
            return self.hash_table[hash_value]
        else:
            return 0
        # return self.hash_table.get(hash_value, [])

    def getProjections(self):
        return self.projections

class LSH:
    def __init__(self, num_tables, hash_size, inp_dimensions):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = list()
        for i in range(self.num_tables):
            self.hash_tables.append(HashTable(self.hash_size, self.inp_dimensions))
    
    def __setitem__(self, inp_vec, label):
        for table in self.hash_tables:
            table[inp_vec] = label
    
    def __getitem__(self, inp_vec):
        results = list()
        for table in self.hash_tables:
            results.extend(table[inp_vec])
        return list(set(results))