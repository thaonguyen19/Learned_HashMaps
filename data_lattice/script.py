import numpy as np

N = 100000

def write_data_to_file(data, value, filename):
    f = open(filename, "w")
    f.write("feature1,value\n")
    for i in range(data.shape[0]):
        f.write("%.10f,%.10f\n" % (data[i], value[i]))

#data = np.random.uniform(-1000, 1000, size=N)
#data = np.random.lognormal(0, 2, N)
data = np.random.normal(10, 0.0001, N)
data.sort()
indices = np.arange(N)
np.random.shuffle(indices)
train_indices = indices[:60000]
val_indices = indices[60000:80000]
test_indices = indices[80000:]
write_data_to_file(data[train_indices], train_indices/N, "normal2.train")
write_data_to_file(data[val_indices], val_indices/N, "normal2.val")
write_data_to_file(data[test_indices], test_indices/N, "normal2.test")

