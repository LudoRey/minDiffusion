import os
import csv
import time
import torch
#import cv2
from matplotlib import cm

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
    
def save_logger(logger, file_path):
    fieldnames = list(logger[0].keys())
    # Create csv file to store loss if does not exist
    if not os.path.exists(file_path):
        with open(file_path, mode="a", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
    # Append logged loss to file, reset logged_loss
    with open(file_path, mode="a", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerows(logger)

def apply_colormap(x, vmin=0, vmax=1, cmap=cm.get_cmap('hot')):
    '''Input: a Bx1xHxW tensor. Output: a Bx3xHxW tensor.'''
    x = torch.clamp(x, vmin, vmax)
    x = (x-vmin)/(vmax-vmin)
    y = cmap(x.squeeze(1).cpu().numpy())
    y = torch.Tensor(y).permute(0,3,1,2) # BxHxWxC -> BxCxHxW
    return y
    
def write_video(output_path, frames, fps):
    # Define the codec and its settings
    fourcc = cv2.VideoWriter_fourcc(*'FFV1') 
    frame_width, frame_height = frames.shape[2], frames.shape[1]  # Frame dimensions

    # Set a higher bitrate for better quality (adjust as needed)
    bitrate = 80000  # 80000 kbps (80 Mbps)

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=True)
    out.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'FFV1'))
    out.set(cv2.CAP_PROP_BITRATE, bitrate * 1000)  # Convert kbps to bps

    # Write frames to the video file
    for i in range(frames.shape[0]):
        frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)  # Convert to BGR format
        out.write(frame)

    # Release the video writer
    out.release()