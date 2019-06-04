from hashtable import Murmur3HashTable
from hashtable import BuiltInHashTable
from hashtable import PolyHashTable
from hashtable import HashTable

files = ['linear', 'lognormal', 'normal', 'multivariate', 'shuttle']

buckets = [20000, 20000, 20000, 20000, 14500]

floatsSettings = [True, True, True, True, False]

dimensions = [1, 1, 1, 2, 9]

fileExtensions = ['.test', '.test', '.test', '.test', '.tst']

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
		tokens = line.split(',')
		if tokens[0] != 'feature1':
			arr = [float(tokens[i]) for i in range(dimensions[i])]
			hashtable.__setitem__(arr)
	f.close()
	# hashtable.makeDict(name)
	print(str(fileName) + ' with Murmur: ' + str(hashtable.conflicts()[0]) + ' and ' + str(hashtable.conflicts()[1]) + ' w/ avg. bucket height of ' + str(hashtable.conflicts()[2]))
	# hashtable.generateScatter(fileName)
	# hashtable.generateBar(fileName)

def runBuiltIn(name, fileName, buckets, floats, i):
	hashtable = BuiltInHashTable(buckets, floats)
	f = open(fileName, 'r')
	for line in f:
		tokens = line.split(',')
		if tokens[0] != 'feature1':
			arr = [float(tokens[i]) for i in range(dimensions[i])]
			hashtable.__setitem__(arr)
	f.close()
	# hashtable.makeDict(name)
	print(str(fileName) + ' with BuiltIn: ' + str(hashtable.conflicts()[0]) + ' and ' + str(hashtable.conflicts()[1]) + ' w/ avg. bucket height of ' + str(hashtable.conflicts()[2]))
	# hashtable.generateScatter(fileName)
	# hashtable.generateBar(fileName)

def runPoly(name, fileName, buckets, floats, i):
	hashtable = PolyHashTable(buckets, dimensions[i])
	f = open(fileName, 'r')
	for line in f:
		tokens = line.split(',')
		if tokens[0] != 'feature1':
			arr = [float(tokens[i]) for i in range(dimensions[i])]
			hashtable.__setitem__(arr)
	f.close()
	# hashtable.makeDict(name)
	print(str(fileName) + ' with Poly: ' + str(hashtable.conflicts()[0]) + ' and ' + str(hashtable.conflicts()[1]) + ' w/ avg. bucket height of ' + str(hashtable.conflicts()[2]))
	# hashtable.generateScatter(fileName)
	# hashtable.generateBar(fileName)

def runHash(name, fileName, buckets, floats, i):
	hashtable = HashTable(logN(buckets), dimensions[i])
	f = open(fileName, 'r')
	for line in f:
		tokens = line.split(',')
		if tokens[0] != 'feature1':
			arr = [float(tokens[i]) for i in range(dimensions[i])]
			hashtable.__setitem__(arr)
	f.close()
	# hashtable.makeDict(name)
	print(str(fileName) + ' with Normal: ' + str(hashtable.conflicts()[0]) + ' and ' + str(hashtable.conflicts()[1]) + ' w/ avg. bucket height of ' + str(hashtable.conflicts()[2]))
	# hashtable.generateScatter(fileName)

def runHashes(name, fileName, buckets, floats, i):
	runMurmur(name, fileName, buckets, floats, i)
	runBuiltIn(name, fileName, buckets, floats, i)
	runPoly(name, fileName, buckets, floats, i)
	# runHash(name, fileName, buckets, floats, i)

if inp_file == -1:
	for i in range(4):
		runHashes(files[i], files[i] + fileExtensions[i], buckets[i], floatsSettings[i], i)
else:
	runHashes(files[inp_file], files[inp_file] + fileExtensions[inp_file], buckets[inp_file], floatsSettings[inp_file], inp_file)

