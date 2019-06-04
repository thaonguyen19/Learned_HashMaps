import pandas as pd
import tensorflow as tf
import tensorflow_lattice as tfl
import logging
logging.getLogger().setLevel(logging.INFO)
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import shutil

data_dir = sys.argv[1]
USING_LATTICE = False
NUM_FEATURES = 1
output_dir = "results/"
quantiles_dir = "quantiles/"

CSV_COLUMNS = ["feature" + str(i) for i in range(1, NUM_FEATURES + 1)] + ["value"]

def get_test_input_fn():
  return get_input_fn(data_dir + ".test", batch_size=10000, num_epochs=1, shuffle=False)

def get_val_input_fn():
  return get_input_fn(data_dir + ".val", batch_size=10000, num_epochs=1, shuffle=False)
    
def get_train_input_fn(batch_size=10000, num_epochs=1, shuffle=False):
  return get_input_fn(data_dir + ".train", batch_size, num_epochs, shuffle)

def get_test_expert_input_fn(idx):
  return get_input_fn(data_dir + "_expert" + str(idx) + ".test", batch_size=10000, num_epochs=1, shuffle=False)

def get_train_expert_input_fn(idx, batch_size=10000, num_epochs=1, shuffle=False):
  return get_input_fn(data_dir + "_expert" + str(idx) + ".train", batch_size, num_epochs, shuffle)

def get_input_fn(file_path, batch_size, num_epochs, shuffle):
  df_data = pd.read_csv(
      tf.gfile.Open(file_path),
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=1)
  labels = df_data["value"]
  return tf.estimator.inputs.pandas_input_fn(
      x=df_data,
      y=labels,
      batch_size=batch_size,
      shuffle=shuffle,
      num_epochs=num_epochs,
      num_threads=1)

def create_feature_columns():
  # Categorical features.
  return [tf.feature_column.numeric_column("feature" + str(i)) for i in range(1, NUM_FEATURES + 1)]

def create_quantiles(quantiles_dir):
    """Creates quantiles directory if it doesn't yet exist."""
    input_fn = get_test_input_fn()
    tfl.save_quantiles_for_keypoints(
        input_fn=input_fn,
        save_dir=quantiles_dir,
        feature_columns=create_feature_columns(),
        num_steps=None)

def create_calibrated_linear(feature_columns, config, quantiles_dir):
    feature_names = [fc.name for fc in feature_columns]
    hparams = tfl.CalibratedLinearHParams(
                feature_names=feature_names,
                num_keypoints=1000,
                learning_rate=0.01)
    hparams.set_feature_param("feature1", "monotonicity", 1)
    return tfl.calibrated_linear_regressor(
            feature_columns=feature_columns,
            model_dir=config.model_dir,
            config=config,
            hparams=hparams,
            quantiles_dir=quantiles_dir)

def create_calibrated_lattice(feature_columns, config, quantiles_dir):
    feature_names = [fc.name for fc in feature_columns]
    hparams = tfl.CalibratedLatticeHParams(
                feature_names=feature_names,
                num_keypoints=20000,
                lattice_l2_laplacian_reg=1e-2,
                lattice_l2_torsion_reg=1e-2,
                learning_rate=0.00001,
                lattice_size=2)
    hparams.set_feature_param("feature1", "monotonicity", 1)
    return tfl.calibrated_lattice_classifier(
            feature_columns=feature_columns,
            model_dir=config.model_dir,
            config=config,
            hparams=hparams,
            quantiles_dir=quantiles_dir)

def create_estimator(config, quantiles_dir):
    """Creates estimator for given configuration based on --model_type."""
    feature_columns = create_feature_columns()
    if USING_LATTICE:
        return create_calibrated_lattice(feature_columns, config, quantiles_dir)
    else:
        return create_calibrated_linear(feature_columns, config, quantiles_dir)

def calculate_collision_rate(estimator, input_fn, extension, N):
    data = [] 
    header = None
    is_header = True
    with open(data_dir + extension, "r") as f:
        for line in f:
            if is_header:
                is_header = False
                header = line.strip()
            else:
                data.append([float(x) for x in line.strip().split(",")])

    results = estimator.predict(input_fn=input_fn)
    buckets = defaultdict(int)
    header += ",stage1"
    idx = 0
    for result in results:
        data[idx].append(result['predictions'][0])
        idx += 1
        if USING_LATTICE:
            bucket = min(N - 1, max(0, int(result['logistic'][0]*N)))
        else:
            bucket = min(N - 1, max(0, int(result['predictions'][0]*N)))
        buckets[bucket] += 1

    expert_data = [[] for _ in range(10)]
    for row in data:
        expert = int(row[1]*10)
        expert_data[expert].append(row)
    for expert in range(10):
        output = open(data_dir + "_expert" + str(expert) + extension, "w")
        output.write(header + "\n")
        for row in expert_data[expert]:
            output.write(",".join(["%.10f" % entry for entry in row]) + "\n")
        output.close()

    num_collisions, buckets_used, avg = 0, 0, 0
    cnt = Counter()
    for i in range(N):
        if buckets[i] > 1:
            num_collisions += 1
            avg += buckets[i]
        if buckets[i] > 0:
            buckets_used += 1
            cnt[buckets[i]] += 1
            #print("bucket %d, number of entries: %d" % (i, buckets[i]))

    print("collision rate:", num_collisions/buckets_used)
    print("average number of entries in a bad bucket:", avg/num_collisions)
    print(cnt)

def reset():
    if os.path.exists(quantiles_dir):
        shutil.rmtree(quantiles_dir)
    os.mkdir(quantiles_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

def main():
    reset()
    create_quantiles(quantiles_dir)

    # Create config and then model.
    config = tf.estimator.RunConfig().replace(model_dir=output_dir)
    estimator = create_estimator(config, quantiles_dir)
   
    estimator.train(input_fn=get_train_input_fn(batch_size=64, num_epochs=1, shuffle=True))
    print("training stats:")
    calculate_collision_rate(estimator, get_train_input_fn(), ".train", 60000)
    #print("validation stats:")
    #calculate_collision_rate(estimator, get_val_input_fn(), 20000)
    print("test stats:")
    calculate_collision_rate(estimator, get_test_input_fn(), ".test", 20000)
    
    train_buckets, test_buckets = defaultdict(int), defaultdict(int)
    final_output = open("final_predictions.txt", "w")
    for expert in range(10):
        cur_estimator = create_estimator(config, quantiles_dir)
        cur_estimator.train(input_fn=get_train_expert_input_fn(expert, batch_size=32, num_epochs=1, shuffle=True))

        results = cur_estimator.predict(input_fn=get_train_expert_input_fn(expert))
        N = 60000
        for result in results:
            bucket = min(N - 1, max(0, int(result['predictions'][0]*N)))
            train_buckets[bucket] += 1
            
        results = cur_estimator.predict(input_fn=get_test_expert_input_fn(expert))
        final_pred = []
        N = 20000
        for result in results:
            bucket = min(N - 1, max(0, int(result['predictions'][0]*N)))
            final_pred.append(result['predictions'][0])
            test_buckets[bucket] += 1
        final_pred.sort()
        final_output.write(" ".join(["%.10f" % x for x in final_pred]) + "\n")

    final_output.close()

    num_collisions, buckets_used, avg = 0, 0, 0
    cnt = Counter()
    for i in range(60000):
        if train_buckets[i] > 1:
            num_collisions += 1
            avg += train_buckets[i]
        if train_buckets[i] > 0:
            buckets_used += 1
            cnt[train_buckets[i]] += 1
            #print("bucket %d, number of entries: %d" % (i, buckets[i]))

    print("training collision rate:", num_collisions/buckets_used)
    print("average number of entries in a bad bucket:", avg/num_collisions)
    print(cnt)

    num_collisions, buckets_used, avg = 0, 0, 0
    cnt = Counter()
    for i in range(20000):
        if test_buckets[i] > 1:
            num_collisions += 1
            avg += test_buckets[i]
        if test_buckets[i] > 0:
            buckets_used += 1
            cnt[test_buckets[i]] += 1
            #print("bucket %d, number of entries: %d" % (i, buckets[i]))

    print("test collision rate:", num_collisions/buckets_used)
    print("fraction of buckets used:", buckets_used/20000)
    print("average number of entries in a bad bucket:", avg/num_collisions)
    print(cnt)
    x, y = [], []
    for e in cnt:
        x.append(e)
        y.append(cnt[e])
    plt.bar(x, y)
    plt.xlabel("Number of Elements in a Bucket")
    plt.ylabel("Number of Buckets")
    plt.title("Element Distribution in Buckets for Learned Index: lognormal.test")
    plt.savefig("lognormal.png")

main()
