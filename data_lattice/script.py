import numpy as np
import sys

np.random.seed(166)
name = sys.argv[1]
N = int(sys.argv[2])

def write_data_to_file(data, value, filename):
    f = open(filename, "w")
    f.write("feature1,value\n")
    for i in range(data.shape[0]):
        f.write("%.10f,%.10f\n" % (data[i], value[i]))

if name == "linear":
    data = np.random.uniform(-1000, 1000, size=N)
elif name == "normal":
    data = np.random.normal(10, 0.0001, size=N)
elif name == "lognormal":
    data = np.random.lognormal(0, 2, size=N)

data.sort()
indices = np.arange(N)
np.random.shuffle(indices)

train_cutoff, val_cutoff = int(0.6*N), int(0.8*N)
train_indices, val_indices, test_indices = indices[:train_cutoff], indices[train_cutoff:val_cutoff], indices[val_cutoff:]

write_data_to_file(data[train_indices], train_indices/N, name + ".train")
write_data_to_file(data[val_indices], val_indices/N, name + ".val")
write_data_to_file(data[test_indices], test_indices/N, name + ".test")

