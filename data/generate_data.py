import numpy as np

N = 100000

def write_data_to_file(data, filename):
    data = np.sort(data, axis=0)
    f = open(filename, "w")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write("%.10f " % data[i][j])
        f.write("\n")

def generate_linear_data(a, b):
    data = [a*np.random.uniform() + b for _ in range(N)]
    return np.expand_dims(np.array(data), axis=1)

def generate_univariate_normal_data(mean, std):
    data = np.random.normal(mean,std, N)
    return np.expand_dims(data, axis=1)

def generate_multivariate_normal_data(mean, cov):
    data = np.random.multivariate_normal(mean, cov, N)
    return data

def main():
    np.random.seed(166)
    write_data_to_file(generate_linear_data(2, 1), "linear_a=2_b=1.txt")
    write_data_to_file(generate_univariate_normal_data(1, 1), "normal_mean=1_std=1.txt")
    write_data_to_file(generate_univariate_normal_data(2, 0.001), "normal_mean=2_std=0.001.txt")
    write_data_to_file(generate_multivariate_normal_data([-1, 1], [[1, 0.5], [0.5, 1]]), "multivariate_normal1.txt")
    write_data_to_file(np.hstack((generate_multivariate_normal_data([1, 0], [[1, 0.2], [0.2, 1]]), \
                                  generate_linear_data(-2, 1), \
                                  generate_univariate_normal_data(2, 0.01))), "mixed1.txt")

if __name__ == "__main__":
    main()
