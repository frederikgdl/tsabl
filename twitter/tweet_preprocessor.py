import logging

from time import time
from os import path

from utils import file_ops
from utils import text_processing


def main():
    logging.info('Loading tweet data')
    t = time()
    if args.tsv:
        texts = file_ops.read_tweets_tsv_file(args.in_file)
    else:
        texts = file_ops.read_lines(args.in_file)
    logging.debug('Done. {}s'.format(str(time() - t)))

    logging.info('Cleaning and tokenizing tweets')
    t = time()
    preprocessed_texts = []
    for i, text in enumerate(texts):

        # Reduces text to have maximum 10 repeating character
        # Prevents URL matching from taking very long time due of catastrophic backtracking
        text = text_processing.reduce_excessive_lengthening(text)

        preprocessed_texts.append(' '.join(text_processing.clean_and_twokenize(text)))
        print('Processed tweet nr. {}'.format(i + 1), end='\r')
    texts = preprocessed_texts
    logging.debug('Done. {}s'.format(str(time() - t)))

    if args.lowercase:
        logging.info('Lowercasing tweets')
        t = time()
        texts = [text.lower() for text in texts]
        logging.debug('Done. {}s'.format(str(time() - t)))

    logging.info('Writing preprocessed tweets to file')
    t = time()
    file_ops.write_tweets(texts, args.out_file)
    logging.debug('Done. {}s'.format(str(time() - t)))


def print_intro():
    print()
    print('Preprocessing tweets')
    print()
    print('Input file:\t\t\t{}'.format(args.in_file))
    print()
    print('Output file:\t\t\t{}'.format(args.out_file))
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tweet preprocessor tweets')

    # Input file
    parser.add_argument('in_file', help='a tsv file containing tweets')

    # Output file
    parser.add_argument('out_file', help='text file to save preprocessed tweets to')

    # Input parameters
    parser.add_argument('-tsv', action='store_true', help='input file has tsv format')

    parser.add_argument('-l', '--lowercase', action='store_true', help='lowercase the tweets')

    # Directory
    parser.add_argument('--dir', nargs='?', default='.',
                        help='optional base directory for in_file and out_file')

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
        logging.critical('Could not find file called: {}'.format(args.in_file))
        exit(1)

    if args.quiet:
        logging.disable(levels[0])
    else:
        print_intro()

    main()
