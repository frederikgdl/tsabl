import sys
import os
import logging

from os import path
from time import time

from utils import file_ops
from api import TwitterApi

twitter_api = TwitterApi()

# Fetch tweet id from id, sentiment pair from previous step
def get_tweet_ids_from_raw(lines):
    return [line.split()[0] for line in lines]

# Turn a list of tweet ids into a list of tweet objects
def get_tweets(ids):
    return twitter_api.bulk_get_statuses(ids)

def main():

    t = time()
    if args.extension:
        logging.info('Loading raw tweet data from files with extension {}'.format(args.extension))
        lines = file_ops.read_and_concatenate_files(args.dir, args.extension)
    else:
        logging.info('Loading raw tweet data from file {}'.format(args.in_file))
        lines = file_ops.read_lines(args.in_file)
    logging.debug('Done. {}s'.format(str(time() - t)))

    ids = get_tweet_ids_from_raw(lines)

    logging.info('Fetching {} tweets...'.format(len(ids)))
    tweets = get_tweets(ids)
    logging.info('Fetched {} tweets. {} tweets have been deleted.'.format(len(tweets), len(ids) - len(tweets)))

    annotations = None
    if args.labeled:
        # Map training annotations to tweets
        # annotations = { "12389425": "positive", ... }
        existing_tweet_ids = list(map(lambda t: t['id_str'], tweets))
        annotations = {}
        for annotation_tuple in map(lambda line: line.split(), lines):
            if annotation_tuple[0] in existing_tweet_ids:
                annotations[annotation_tuple[0]] = annotation_tuple[1]

    logging.info('Saving processed tweets')
    t = time()
    file_ops.save_json_tweets(tweets, args.out_file_json, annotations)

    if args.out_file_text is not None:
        file_ops.save_text_tweets(tweets, args.out_file_text, annotations)
    logging.debug('Done. {}s'.format(str(time() - t)))

def print_intro():
    print()
    print('Processing RAW_PATH tweets')
    print()

    if args.extension:
        print('Raw tweets files extension:\t{}'.format(args.extension))
    else:
        print('Raw tweets data file:\t\t{}'.format(args.in_file))

    print()
    print('JSON output file:\t\t{}'.format(args.out_file_json))

    if args.out_file_text is not None:
        print('Text output file:\t\t{}'.format(args.out_file_text))

    print()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fetching tweets from Twitter by ids.')

    # Input file
    parser.add_argument('in_file', nargs='?', default='data/raw/tweets.id',
                        help='a text file containing tweets (default: data/raw/tweets.id)')

    # Output file
    parser.add_argument('out_file_json', nargs='?', default='data/raw/tweets.json',
                        help='file to save tweets to in json format (default: data/raw/tweets.json)')

    parser.add_argument('out_file_text', nargs='?', default=None,
                        help='text file to save tweets to (default: None)')

    # Directory
    parser.add_argument('--dir', nargs='?', default='.',
                        help='optional base directory for in_file and out files')

    # Data parameters
    parser.add_argument('-l', '--labeled', action='store_true',
                        help='set if raw tweets are labeled (default: False)')

    parser.add_argument('-e', '--extension', nargs='?', default=None,
                        help='set file extension to read multiple files (default: None)')

    # Logger verbosity parameters
    parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level, repeat to increase')
    parser.add_argument('-q', '--quiet', action='store_true', help='no print to console')

    args = parser.parse_args()
    args.in_file = path.join(args.dir, args.in_file)
    args.out_file_json = path.join(args.dir, args.out_file_json)

    if args.out_file_text is not None:
        args.out_file_text = path.join(args.dir, args.out_file_text)

    # Set logger verbosity level. Default is logging.INFO.
    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(len(levels) - 1, args.verbose + 2)]  # capped to number of levels
    logging.basicConfig(level=level, format='%(asctime)s\t%(levelname)s\t%(message)s')

    # Check if in_file exists
    if not args.extension and not path.isfile(args.in_file):
        logging.critical('\nCould not find file called {}'.format(args.in_file))
        exit(1)

    if args.extension:
        for fname in os.listdir(args.dir):
            if fname.endswith(args.extension):
                break
        else:
            logging.critical('\nCould not find files with extension {}'.format(args.extension))
            exit(1)

    if args.quiet:
        logging.disable(levels[0])
    else:
        print_intro()

    main()
