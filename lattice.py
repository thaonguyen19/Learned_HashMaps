import pandas as pd
import tensorflow as tf
import tensorflow_lattice as tfl
import logging
logging.getLogger().setLevel(logging.INFO)
from collections import defaultdict
import matplotlib.pyplot as plt

train_dir = "data_lattice/linear.train"
test_dir = "data_lattice/linear.test"
output_dir = "results/"
quantiles_dir = "quantiles/"

CSV_COLUMNS = [
    "feature1", "value"
]

def get_test_input_fn(batch_size, num_epochs, shuffle):
  return get_input_fn(test_dir, batch_size, num_epochs, shuffle)

def get_train_input_fn(batch_size, num_epochs, shuffle):
  return get_input_fn(train_dir, batch_size, num_epochs, shuffle)

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
  return [tf.feature_column.numeric_column("feature1")]

def create_quantiles(quantiles_dir):
    """Creates quantiles directory if it doesn't yet exist."""
    batch_size = 10000
    input_fn = get_test_input_fn(
        batch_size=batch_size, num_epochs=1, shuffle=False)
    # Reads until input is exhausted, 10000 at a time.
    tfl.save_quantiles_for_keypoints(
        input_fn=input_fn,
        save_dir=quantiles_dir,
        feature_columns=create_feature_columns(),
        num_steps=None)

def create_calibrated_linear(feature_columns, config, quantiles_dir):
    feature_names = [fc.name for fc in feature_columns]
    hparams = tfl.CalibratedLinearHParams(feature_names=feature_names)
    return tfl.calibrated_linear_regressor(
        feature_columns=feature_columns,
        model_dir=config.model_dir,
        config=config,
        hparams=hparams,
        quantiles_dir=quantiles_dir)

def create_estimator(config, quantiles_dir):
    """Creates estimator for given configuration based on --model_type."""
    feature_columns = create_feature_columns()
    return create_calibrated_linear(feature_columns, config, quantiles_dir)

def calculate_collision_rate(estimator, input_fn, N):
    results = estimator.predict(input_fn=input_fn)
    buckets = defaultdict(int)
    for result in results:
        bucket = int(result['predictions'][0]*N)
        if bucket < 0:
            bucket = 0
        if bucket > N - 1:
            bucket = N - 1
        buckets[bucket] += 1
    num_collisions = 0
    x, y = [], []
    avg = 0
    for i in range(N):
        if buckets[i] > 1:
            num_collisions += 1
            avg += buckets[i]
        if buckets[i] > 0:
            print("bucket %d, number of entries: %d" % (i, buckets[i]))
        x.append(i)
        y.append(buckets[i])
    print("collision rate:", num_collisions/N)
    print("average number of entries in a bad bucket:", avg/num_collisions)
    #plt.bar(x, y)
    #plt.xlabel("bucket number")
    #plt.ylabel("number of elements")
    #plt.savefig("normal2.png")

def main():
    create_quantiles(quantiles_dir)

    # Create config and then model.
    config = tf.estimator.RunConfig().replace(model_dir=output_dir)
    estimator = create_estimator(config, quantiles_dir)
    
    estimator.train(input_fn=get_train_input_fn(batch_size=64, num_epochs=10, shuffle=True))
    calculate_collision_rate(estimator, get_train_input_fn(batch_size=10000, num_epochs=1, shuffle=False), 60000)
    calculate_collision_rate(estimator, get_test_input_fn(batch_size=10000, num_epochs=1, shuffle=False), 20000)
    
main()
