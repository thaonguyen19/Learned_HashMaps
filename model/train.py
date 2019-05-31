from rmi import *
from dataloader import *
import argparse

VAL_RATIO = 0.2
TEST_RATIO = 0.2


def train(args):
    data_set = load_synthetic_data(args.data_dir)
    data_sets = create_train_validate_test_data_sets(data_set, VAL_RATIO, TEST_RATIO)
    model = RMI_simple(data_sets.train, hidden_layer_widths=[args.hidden_width], num_experts=args.num_experts)
    max_steps = [data_sets.train.num_keys//b for b in args.batch_size]
    model.run_training(batch_sizes=args.batch_size, max_steps=max_steps,
                        learning_rates=args.lr, model_save_dir=args.model_save_dir, epoch=args.epoch)
    model.get_weights_from_trained_model()
    x = data_sets.train.next_batch(10)[0]
    print(x)
    model.inspect_inference_steps(x)
    model.calc_min_max_errors()
    return model


#TODO: use val set for hyperparameter tuning & run on test set
def inference():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-model_save_dir', default='results/')
    parser.add_argument('-data_dir', default='../data/linear_a=2_b=1.txt')
    parser.add_argument('-lr', nargs='+', type=float, default=[1e-3, 1e-3])
    parser.add_argument('-batch_size', nargs='+', type=int, default=[64, 64])
    parser.add_argument('-num_experts', type=int, default=10)
    parser.add_argument('-hidden_width', type=int, default=16)
    parser.add_argument('-epoch', type=int, default=5)
    args = parser.parse_args()
    train(args)
