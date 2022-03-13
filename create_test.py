from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from random import random
import pdb
from datetime import datetime

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file', help='which file to use for sampling', default='data/mergedfiles2.txt')
    parser.add_argument('--output-file', help='which file to use for sampling', default=f'data/test_file_{datetime.now().strftime("%s")}.txt')
    args = parser.parse_args()

    with open(args.file, "r") as data_file, open(args.output_file, "w") as test_file:
        for line in data_file:
            line = line.strip()
            if not line:
                continue
            truncate = int(random() * len(line)//2 + len(line)//2)
            print(f"{line[:truncate]}\t{line[truncate]}", file=test_file)
