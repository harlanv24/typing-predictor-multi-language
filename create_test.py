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
            if not line.strip():
                continue
            lines = line.strip().split(".")
            for short_line in lines: 
                if len(short_line) <= 20:
                    continue
                
                truncate = int(random() * 10 + 10)
                print(short_line, truncate)
                print(f"{short_line[:truncate]}\t{short_line[truncate]}", file=test_file)
