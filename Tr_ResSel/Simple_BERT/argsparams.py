import argparse
import os
from input_params import *
from config_params import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#print(os.getcwd())
## Required parameters
parser = argparse.ArgumentParser()

parser.add_argument("--numClusters",
                    default=num_clusters)

parser.add_argument("--dataPath",
                    default=data_file_path,
                    type=str,
                    help="The dataset used for training and test.")

parser.add_argument("--savePath",
                    default=save_model_path,   #"../../../../data/BERT/FT_checkpoint/",
                    type=str,
                    help="The path to save model.")
# parser.add_argument("--score_file_path",
#                     #default="./Fine-Tuning/scorefile.txt",
#                     default="../../../../data/BERT/scorefile.txt",
#                     type=str,
#                     help="The path to save model.")

parser.add_argument("--task",
                    default=dataset,
                    type=str,
                    help="The dataset used for training and test.")

parser.add_argument("--/is_training",
                    action='store_true',
                    help="Training model or testing model?")

parser.add_argument("--batch_size",
                    default=batch_size,
                    type=int,
                    help="The batch size.")
parser.add_argument("--learning_rate",
                    default=learning_rate,
                    type=float,
                    help="The initial learning rate for Adamw.")
parser.add_argument("--epochs",
                    default=num_epochs,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--do_lower_case", action='store_true', default=True,
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--network_num",
                    default=0,
                    type=int)

args = parser.parse_args()
#args.savePath += args.task + '.' + "0.pt"
#args.score_file_path = args.score_file_path
# load bert

print(args)
print("Task: ", args.task)
