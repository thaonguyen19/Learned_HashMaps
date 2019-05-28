from rmi import *
from dataloader import *

VAL_RATIO = 0.2
TEST_RATIO = 0.2


def train(data_path):
    data_set = load_synthetic_data(data_path)
    data_sets = create_train_validate_test_data_sets(data_set, VAL_RATIO, TEST_RATIO)
    model = RMI_simple(train_set, hidden_layer_widths=[16,16], num_experts=10)
    model.run_training(batch_sizes=batch_sizes, max_steps=max_steps,
                        learning_rates=learning_rates, model_save_dir=model_save_dir)
    model.get_weights_from_trained_model()
    model.calc_min_max_errors()
    return model


#TODO: use val set for hyperparameter tuning & run on test set
def inference():
    pass

if __name__ == '__main__':
    pass




