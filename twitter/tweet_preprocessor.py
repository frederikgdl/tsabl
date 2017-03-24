import logging

from os import path

from utils import file_ops
from utils import text_processing


def main():
    texts = file_ops.read_tweets_tsv_file(args.in_file)
    # texts = []
    # with open(args.in_file, 'r') as f:
    #     for line in f:
    #         line = line.split('\t')[1]
    #         texts.append(line)

    # texts = list(map(lambda tweet: ' '.join(text_processing.clean_and_twokenize(tweet)), texts))

    preprocessed_texts = []
    for i, text in enumerate(texts):
        preprocessed_texts.append(text_processing.clean_and_twokenize(text))
        print('Processed tweet nr. {}'.format(i), end='\r')
    texts = preprocessed_texts

    file_ops.save_text_tweets(texts, args.out_file)


def print_intro():
    print()
    print('Preprocessing tweets')
    print()
    print('Input file:\t\t\t' + args.in_file)
    print()
    print('Output file:\t\t\t' + args.out_file)
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess tweets')

    # Input file
    parser.add_argument('in_file', help='a tsv file containing tweets')

    # Output file
    parser.add_argument('out_file', help='text file to save preprocessed tweets to')


    # Directory
    parser.add_argument('--dir', nargs='?', default='.',
                        help='optional base directory for in_file and out_file')

    # Data parameters
    # parser.add_argument('-l', '--labeled', action='store_true',
    #                     help='set if raw tweets are labeled (default: False)')

    # Logger verbosity parameters
    parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level, repeat to increase')
    parser.add_argument('-q', '--quiet', action='store_true', help='no print to console')

    args = parser.parse_args()
    args.in_file = path.join(args.dir, args.in_file)
    args.out_file = path.join(args.dir, args.out_file)

    # Set logger verbosity level. Default is logging.INFO.
    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(len(levels) - 1, args.verbose + 2)]  # capped to number of levels
    logging.basicConfig(level=level, format="%(asctime)s\t%(levelname)s\t%(message)s")

    # Check if in_file exists
    if not path.isfile(args.in_file):
        logging.critical('\nCould not find file called ', args.in_file)
        exit(1)

    if args.quiet:
        logging.disable(levels[0])
    else:
        print_intro()

    main()
