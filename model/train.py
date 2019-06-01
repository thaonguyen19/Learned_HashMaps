from rmi import *
from dataloader import *
import argparse
import copy

VAL_RATIO = 0.2
TEST_RATIO = 0.2
LOAD_FACTOR = 0.5

def get_stats(model, dataset, name):
    N = len(dataset.keys)
    pred, _ = model._run_inference_tensorflow(dataset.keys)
    mse = np.mean((pred - dataset.positions)**2)
    print(name + " MSE: %.10f" % mse)
    num_buckets = int(N/LOAD_FACTOR)
    unique_index = set([int(num_buckets*pred[i]) for i in range(N)])
    num_collisions = N - len(unique_index)
    print(name + " number of collisions: %d" % num_collisions)
    return mse

def train(args):
    print("training on:", args)
    data_set = load_synthetic_data(args.data_dir)
    data_sets = create_train_validate_test_data_sets(data_set, VAL_RATIO, TEST_RATIO)
    model = RMI_simple(data_sets.train, hidden_layer_widths=args.hidden_width, num_experts=args.num_experts)
    max_steps = [data_sets.train.num_keys//b for b in args.batch_size]
    model.run_training(batch_sizes=args.batch_size, max_steps=max_steps,
                       learning_rates=args.lr, model_save_dir=args.model_save_dir, epoch=args.epoch)
    model.get_weights_from_trained_model()

    print(data_sets.train.positions[:10])
    model.inspect_inference_steps(data_sets.train.keys[:10])
    model.calc_min_max_errors()

    train_mse = get_stats(model, data_sets.train, 'Training')
    val_mse = get_stats(model, data_sets.validate, 'Validation')
    return model, val_mse

def inference(args):
    best_params, best_model, best_mse = {}, None, 100
    for stage1_lr in [1e-4, 1e-3, 1e-2, 1e-1]:
        for stage2_lr in [1e-4, 1e-3, 1e-2, 1e-1]:
            for num_experts in [2, 4, 8, 16]:
                for h in [16, 32, 64, 128]:
                    for num_layers in [1, 2, 3, 4]:
                        args.lr = [stage1_lr, stage2_lr]
                        args.num_experts = num_experts
                        args.hidden_width = [h] * num_layers
                        cur_model, cur_mse = train(args) 
                        if cur_mse < best_mse:
                            best_params = copy.deepcopy(args)
                            best_model = copy.deepcopy(cur_model)
                            best_mse = cur_mse
    print("Best MSE: %.10f", best_mse)
    print("Best params:", best_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-model_save_dir', default='results/')
    parser.add_argument('-data_dir', default='../data/normal_mean=1_std=1.txt')
    parser.add_argument('-lr', nargs='+', type=float, default=[1e-3, 1e-3])
    parser.add_argument('-batch_size', nargs='+', type=int, default=[64, 64])
    parser.add_argument('-num_experts', type=int, default=10)
    parser.add_argument('-hidden_width', nargs='+', type=int, default=[16])
    parser.add_argument('-epoch', type=int, default=5)
    args = parser.parse_args()
    #inference(args)
    train(args)
