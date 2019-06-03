import tensorflow_lattice as tfl

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
    if FLAGS.model_type == "calibrated_linear":
        return create_calibrated_linear(feature_columns, config, quantiles_dir)
    elif FLAGS.model_type == "calibrated_lattice":
        return create_calibrated_lattice(feature_columns, config, quantiles_dir)

def main(args):
    # Create config and then model.
    config = tf.estimator.RunConfig().replace(model_dir=output_dir)
    estimator = create_estimator(config, quantiles_dir)
    
    if FLAGS.run == "train":
        train(estimator)
    elif FLAGS.run == "evaluate":
        evaluate(estimator)
