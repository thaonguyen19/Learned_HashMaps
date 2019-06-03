from hashtable import Murmur3HashTable
from hashtable import BuiltInHashTable
from hashtable import PolyHashTable
from hashtable import HashTable

files = ['linear_a=2_b=1_noise', 'linear_a=2_b=1', 'lognormal_mean=0_std=0.25', 
			'mixed1', 'multivariate_normal1', 'normal_mean=1_std=1', 'normal_mean=2_std=0.001', 
			'normal_mean=3_std=0.00001', 'shuttle_trn', 'shuttle_tst']

buckets = [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 
			43500, 14500]

floatsSettings = [True, True, True, True, True, True, True, True, False, False]

dimensions = [1, 1, 1, 4, 2, 1, 1, 1, 9, 9]

extraFiles = ['shuttle.trn', 'shuttle.tst']

# choose an index from 0 to 9 to run on a file
# choose -1 to run all of them
inp_file = -1 

def logN(n):
	cur = 1
	idx = 1
	while cur < n:
		cur *= 2
		idx += 1
	return idx

def runMurmur(name, fileName, buckets, floats, i):
	hashtable = Murmur3HashTable(buckets, floats)
	f = open(fileName, 'r')
	for line in f:
		tokens = line.split()
		arr = []
		if i >= 0 and i <= 7:
			arr = [float(tokens[i]) for i in range(len(tokens))]
		elif i >= 8 and i <= 9:
			arr = [int(tokens[i]) for i in range(len(tokens) - 1)]
		hashtable.__setitem__(arr)
	f.close()
	# hashtable.makeDict(name)
	print(str(name) + ' with Murmur: ' + str(hashtable.conflicts()[0]) + ' and ' + str(hashtable.conflicts()[1]))

def runBuiltIn(name, fileName, buckets, floats, i):
	hashtable = BuiltInHashTable(buckets, floats)
	f = open(fileName, 'r')
	for line in f:
		tokens = line.split()
		arr = []
		if i >= 0 and i <= 7:
			arr = [float(tokens[i]) for i in range(len(tokens))]
		elif i >= 8 and i <= 9:
			arr = [int(tokens[i]) for i in range(len(tokens) - 1)]
		hashtable.__setitem__(arr)
	f.close()
	# hashtable.makeDict(name)
	print(str(name) + ' with BuiltIn: ' + str(hashtable.conflicts()[0]) + ' and ' + str(hashtable.conflicts()[1]))

def runPoly(name, fileName, buckets, floats, i):
	hashtable = PolyHashTable(buckets, dimensions[i])
	f = open(fileName, 'r')
	for line in f:
		tokens = line.split()
		arr = []
		if i >= 0 and i <= 7:
			arr = [float(tokens[i]) for i in range(len(tokens))]
		elif i >= 8 and i <= 9:
			arr = [int(tokens[i]) for i in range(len(tokens) - 1)]
		hashtable.__setitem__(arr)
	f.close()
	# hashtable.makeDict(name)
	print(str(name) + ' with Poly: ' + str(hashtable.conflicts()[0]) + ' and ' + str(hashtable.conflicts()[1]))

def runHash(name, fileName, buckets, floats, i):
	hashtable = HashTable(logN(buckets), dimensions[i])
	f = open(fileName, 'r')
	for line in f:
		tokens = line.split()
		arr = []
		if i >= 0 and i <= 7:
			arr = [float(tokens[i]) for i in range(len(tokens))]
		elif i >= 8 and i <= 9:
			arr = [int(tokens[i]) for i in range(len(tokens) - 1)]
		hashtable.__setitem__(arr)
	f.close()
	# hashtable.makeDict(name)
	print(str(name) + ' with Normal: ' + str(hashtable.conflicts()[0]) + ' and ' + str(hashtable.conflicts()[1]))


def runHashes(name, fileName, buckets, floats, i):
	runMurmur(name, fileName, buckets, floats, i)
	runBuiltIn(name, fileName, buckets, floats, i)
	runPoly(name, fileName, buckets, floats, i)
	runHash(name, fileName, buckets, floats, i)

if inp_file == -1:
	for i in range(8):
		runHashes(files[i], files[i] + '.txt', buckets[i], floatsSettings[i], i)
	for i in range(8, 10):
		runHashes(files[i], extraFiles[i - 8], buckets[i], floatsSettings[i], i)
else:
	if inp_file >= 0 and inp_file <= 7:
		runHashes(files[inp_file], files[inp_file] + '.txt', buckets[inp_file], floatsSettings[inp_file], inp_file)
	elif inp_file >= 8 and inp_file <= 9:
		runHashes(files[inp_file], extraFiles[inp_file - 8], buckets[inp_file], floatsSettings[inp_file], inp_file)

