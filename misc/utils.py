import os
import csv
import time

def get_checkpoint_dir(opt):
    # Create checkpoint directory if no checkpoint is loaded
    if opt.load_checkpoint is None:
        date_time_string = time.strftime("%Y%m%d_%H%M", time.localtime())
        checkpoint_dir = "./checkpoints/"+opt.dataset_name+"_"+ date_time_string
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
    # Keep the loaded checkpoint directory
    else:
        checkpoint_dir = os.path.join("./checkpoints", opt.load_checkpoint)
    return checkpoint_dir
   
def write_training_info(opt, file_path):
    # Write info about training run in checkpoint directory
    with open(file_path, "a") as file:
        file.write("Training options:" + "\n")
        for key, value in vars(opt).items():
            file.write(f"{key}: {value} \n")
        file.write("\n")
    
def save_loss(logged_loss, file_path):
    # Create csv file to store loss if does not exist
    if not os.path.exists(file_path):
        with open(file_path, mode="a", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Epoch', 'Loss'])
            writer.writeheader()
    # Append logged loss to file, reset logged_loss
    with open(file_path, mode="a", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Epoch', 'Loss'])
        writer.writerows(logged_loss)
    
       