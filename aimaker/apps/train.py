import aimaker.trainer as trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('setting_files', nargs='?', default='settings')
parser.add_argument('n_epoch', nargs='?', default=1000000, type=int)

args = parser.parse_args()

t = trainer.trainer(args.setting_files)
t.train(args.n_epoch)
