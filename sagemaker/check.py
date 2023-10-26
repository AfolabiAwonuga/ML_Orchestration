import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train-file', type=str, default="train.csv")
args = parser.parse_args()

print(args)