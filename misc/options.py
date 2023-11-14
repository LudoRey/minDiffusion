import argparse

def parse_options():
    parser = argparse.ArgumentParser()
    # Options related to dataset
    parser.add_argument("--dataset_name", type=str, default="DendritesDataset", help="Name of Dataset class. Must be imported.")
    # Options related to training
    parser.add_argument("--n_epoch", type=int, default=100, help="The training run stops after the epoch count reaches n_epoch")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=None, help="Patience parameter for ReduceLROnPlateau. None means no scheduler.")
    # Options related to TA-DM specifically
    parser.add_argument("--denoising_net_name", type=str, default="simple_baseline", help="Either 'simple_baseline' or 'sr3'.")
    parser.add_argument("--denoising_target", type=str, default="y", help="Either 'y' or 'eps'.")
    parser.add_argument("--lambda_TA", type=float, default=1, help="Weight used for the task loss")
    # Options related to loading and saving checkpoints (as well as saving other stuff)
    parser.add_argument("--load_checkpoint", type=str, default=None, help="The checkpoint folder to be loaded.")
    parser.add_argument("--load_epoch", type=int, default=0, help="The model ./checkpoint/<load_checkpoint>/net_<load_epoch>.pth will be loaded.")
    parser.add_argument("--change_dir", action="store_true", default=False, help="By default, will continue to save data/log info in the same checkpoint folder.")
    parser.add_argument("--save_every", type=int, default=10, help="Frequency of checkpoint/samples saves.")

    opt = parser.parse_args()
    return opt