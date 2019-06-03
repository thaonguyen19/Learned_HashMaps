import numpy as np

N = 100000

def write_data_to_file(data, filename):
    data = np.sort(data)
    f = open(filename, "w")
    f.write("feature1,value\n")
    for i in range(data.shape[0]):
        f.write("%.10f,%.10f\n" % (data[i], i/N))

data = np.random.uniform(-1000, 1000, size=N)
write_data_to_file(data, "linear.train")

