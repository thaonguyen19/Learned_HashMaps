import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_empirical_cdf(keys, positions, preds, N, name):
    #assume data is sorted
    if keys:
        plt.plot(keys, positions, label='Actual CDF')
    if preds:
        if len(preds) > 100: #plot all test predictions
            plt.plot(keys, preds, label='Learned CDF')
        else: #divided by expert
            def unique_color():
                return plt.cm.gist_ncar(np.random.random())
            def get_cmap(n, name='hsv'):
                return plt.cm.get_cmap(name, n)

            n_experts = len(preds)
            expert_factor = 1/n_experts
            cmap = get_cmap(n_experts)
            for i, expert_preds in enumerate(preds):
                n_items = len(expert_preds)
                plt.plot(keys[:n_items], expert_preds, color=cmap(i), label='Expert '+str(i+1))
                keys = keys[n_items:]
    plt.legend()
    plt.xlabel('Key value')
    plt.ylabel('Position in CDF')
    plt.title('Expert assignment for %s test data' % name)
    plt.show()

def plot_empirical_pdf(data):
    kde = gaussian_kde(data)
    dist_space = np.linspace(min(data), max(data), 100)
    plt.plot(dist_space, kde(dist_space))
    plt.show()

if __name__ == "__main__":
    N = 20000
    name = 'lognormal'
    preds_file = '../%s_pred.txt' % name
    expert_preds = []
    test_preds = []
    with open(preds_file, 'r') as f:
        for line in f:
            preds = line.strip().split()
            preds = [float(p) for p in preds]
            expert_preds.append(preds)
            test_preds.extend(preds)
    assert(len(test_preds) == N)

    data = []
    labels_file = '../data_lattice/%s.test' % name
    with open(labels_file, 'r') as f:
        f.readline()
        for line in f:
            tup = line.strip().split(',')
            tup = [float(d) for d in tup]
            data.append(tup)
    sorted_data = sorted(data, key=lambda x: x[0]) #sort by key
    keys = [d[0] for d in sorted_data]
    positions = [d[1] for d in sorted_data]
    print(positions[:10])

    #plot empirical & true cdf, may want to sample only a few elements to see at a smaller scale
    plot_empirical_cdf(keys, positions, test_preds, N, name)
    #only plot expert assignment:
    plot_empirical_cdf(None, positions, expert_preds, N)
                
