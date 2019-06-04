from sklearn.model_selection import train_test_split
import argparse
import lightgbm as lgb
import numpy as np

def load_data(data_path, normalized_label):
    keys = []
    with open(data_path, 'r') as file:
        for line in file:
            keys.append([float(x) for x in line.split()])
    num_keys = len(keys)
    keys = np.array(keys)
    positions = np.arange(num_keys).astype(float)
    if normalized_label: #normalize so that values lie in range [0, 1]
        print("Normalizing labels to be between [0,1]")
        positions /= num_keys
    print(keys.shape)
    return keys, positions

def train(args):
    keys, positions = load_data(args.data_dir, args.norm_label)
    x_train, x_test, y_train, y_test = train_test_split(keys, positions, test_size=0.2, random_state=0)

    # params = {}
    # params['learning_rate'] = args.lr
    # params['boosting_type'] = 'gbdt'
    # params['objective'] = 'regression'
    # params['metric'] = 'l2'
    # params['num_leaves'] = 10
    # params['min_child_samples'] = 10
    assert(model == 'gbm' or model == 'xgb')
    d_train = lgb.Dataset(x_train, label=y_train, params={'monotone_constraints':1})

    if model == 'gbm':
        clf = lgb.LGBMRegressor(learning_rate=args.lr, boosting_type='gbdt', \
                           objective='regression', metric='l2', num_leaves=1000, min_child_samples=1)
    else:
        pass

    clf.fit(x_train, y_train)
    y_pred=clf.predict(x_test)
    print(y_test)
    print(y_pred)
    print(np.mean(np.abs(y_pred - y_test)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-data_dir', default='../data/normal_mean=1_std=1.txt')
    parser.add_argument('-lr', type=float)
    parser.add_argument('-norm_label', action='store_true', help='Whether to normalize labels to be within [0,1]')
    #parser.add_argument('-fix_inputs', action='store_true', help='Whether to keep input distribution the same and avoid standardization')
    parser.add_argument('-model')
    args = parser.parse_args()
    train(args)

