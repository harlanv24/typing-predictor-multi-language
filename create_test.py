from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from random import random
import pdb
from datetime import datetime

if __name__ == '__main__':
    language_set = ['zh', 'es','en', 'ar', 'pt', 'ru', 'ja', 'fr']
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file', help='which file to use for sampling', default='data/en/mergedfiles.txt')
    parser.add_argument('--output-file', help='which file to use for sampling', default=f'data/test_file_{datetime.now().strftime("%s")}.txt')
    args = parser.parse_args()

    if args.file == 'all':
        files_handles = [open(f'data/{lang}/mergedfiles.txt', 'r') for lang in language_set]
    else:
        files_handles = [open(args.file, 'r')]
     
    TESTFILE_LINES = 90
    with open(args.output_file, "w") as test_file:
        for fileh in files_handles:
            for _ in range(TESTFILE_LINES//len(language_set)):
                line = fileh.readline()
                if not line.strip():
                    continue
                lines = line.strip().split(".")
                for short_line in lines: 
                    if len(short_line) <= 20:
                        continue
                    
                    truncate = int(random() * 10 + 10)
                    print(short_line, truncate)
                    print(f"{short_line[:truncate]}\t{short_line[truncate]}", file=test_file)
    [f.close() for f in files_handles]
