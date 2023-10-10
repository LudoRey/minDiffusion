import argparse

def parse_options():
    parser = argparse.ArgumentParser()
    # Options related to dataset
    parser.add_argument("--dataset_name", type=str, default="DendritesDataset", help="Name of Dataset class. Must be imported.")
    parser.add_argument("--crop", type=bool, default=False, help="Passed to Dataset argument.")
    # Options related to training
    parser.add_argument("--n_epoch", type=int, default=100, help="The training run stops after the epoch count reaches n_epoch")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    # Options related to loading and saving checkpoints (as well as saving other stuff)
    parser.add_argument("--load_checkpoint", type=str, default=None, help="The checkpoint folder to be loaded. Will continue to save data/log info in the same folder.")
    parser.add_argument("--load_epoch", type=int, default=-1, help="The model ./checkpoint/<load_checkpoint>/net_<load_epoch>.pth will be loaded.")
    parser.add_argument("--save_every", type=int, default=10, help="Frequency of checkpoint/samples saves.")

    opt = parser.parse_args()
    return opt