import argparse
import os
from module.utils import dic_functions
from learner import trainer

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()

parser.add_argument("--run_type", default="LDD_simsiam", help="Run Type")
parser.add_argument("--dataset_in",default="CIFAR", help="Name of the Dataset")
parser.add_argument("--model_in", default="resnet18", help="Name of the model")
parser.add_argument("--train_samples", default=1000, type=int,help="Number of training samples")
parser.add_argument("--bias_ratio", default=0.05, type = float,help="Bias ratio")
parser.add_argument("--runs", default=1, type = int,help="Number of runs")
parser.add_argument("--reduce", default = 1, type = int, help = "Reduce the number of samples")
args = parser.parse_args()

set_seed = dic_functions['set_seed']

write_to_file = dic_functions['write_to_file']

run = trainer(args)

for run_num in range(args.runs):
    run.get_results(run_num)