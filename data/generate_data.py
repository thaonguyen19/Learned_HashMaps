import numpy as np

N = 100000

def generate_linear_data(a, b, filename):
    data = [a*np.random.uniform() + b for _ in range(N)]
    data.sort()
    f = open(filename, "w")
    for x in data:
        f.write("%.10f\n" % x)

def generate_linear_data(mean, std, filename):
    data = [np.random.normal(mean,std) for _ in range(N)]
    data.sort()
    f = open(filename, "w")
    for x in data:
        f.write("%.10f\n" % x)

def main():
    generate_linear_data(2, 1, "linear_a=2_b=1.txt")
    generate_linear_data(1, 1, "normal_mean=1_std=1.txt")
    generate_linear_data(2, 0.001, "normal_mean=2_std=0.001.txt")

if __name__ == "__main__":
    main()
