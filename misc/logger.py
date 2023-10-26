import os
import csv
import time
import pandas as pd
import torch

class Logger():
    def __init__(self, opt):
        self.opt = opt

        if opt.load_checkpoint is not None and opt.keep_dir:
            self.checkpoint_dir = os.path.join("./checkpoints", opt.load_checkpoint)
        else:
            self.checkpoint_dir = self.make_checkpoint_dir()

        self.log_buffer = pd.DataFrame()
        self.save_training_info()

    
    # Initializing stuff
    def make_checkpoint_dir(self):
        opt = self.opt
        date_time_string = time.strftime("%Y%m%d_%H%M", time.localtime())
        checkpoint_dir = f"./checkpoints/{opt.dataset_name}_{date_time_string}"
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        return checkpoint_dir

    # Updating stuff
    def log(self, data_dict):
        new_row = pd.DataFrame.from_records([data_dict])
        self.log_buffer = pd.concat([self.log_buffer, new_row], ignore_index=True)

    # Saving stuff
    def save_training_info(self):
        file_path = os.path.join(self.checkpoint_dir, "training_info.txt")
        opt = self.opt
        # Write info about training run in checkpoint directory
        with open(file_path, "a") as file:
            file.write("Training options: \n")
            for key, value in vars(opt).items():
                file.write(f"{key}: {value} \n")
            file.write("\n")

    def save_csv_log(self):
        file_path = os.path.join(self.checkpoint_dir, "log.csv")
        if not os.path.exists(file_path):
            self.log_buffer.to_csv(file_path, header=True, mode='w', index=False)
        else:
            self.log_buffer.to_csv(file_path, header=False, mode='a', index=False)
        # Reset buffer
        self.log_buffer = pd.DataFrame()