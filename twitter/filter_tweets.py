import json
import logging

import langid

from os import path
from time import time

from utils import file_ops as utils


# Filters data/raw/*.json, data/raw/*.txt and saves corresponding files in data/filtered
# Reads data/raw/tweets.json and return filtered list of tweet dictionary objects
def filter_tweets(lines):
    # Returns true if a tweet is to be kept
    def filter_tweet(tweet: dict) -> bool:
        return 'created_at' in tweet \
               and (tweet['lang'] == 'en' or (tweet['lang'] == 'und' and langid.classify(tweet['text'])[0] == 'en')) \
               and 'retweeted_status' not in tweet

    return list(filter(filter_tweet, map(json.loads, lines)))


def main():
    logging.info('Loading raw tweet data from file {}'.format(args.in_file))
    t = time()
    if args.labeled:
        raw_tweets, labels = utils.read_labeled_file(args.in_file)
        ids = [json.loads(tweet)['id_str'] for tweet in raw_tweets]
        annotations = dict(zip(ids, labels))
    else:
        raw_tweets = utils.read_lines(args.in_file)
    logging.debug('Done. {}s'.format(str(time() - t)))

    logging.info('Filtering tweets')
    t = time()
    try:
        filtered_tweets = filter_tweets(raw_tweets)
    except:
        logging.critical('\nCould not process JSON objects. If data is labeled use the "-l" option.')
        exit(1)

    new_annotations = None
    if args.labeled:
        filtered_ids = [tweet['id_str'] for tweet in filtered_tweets]
        new_annotations = dict((tweet_id, label) for tweet_id, label
                               in annotations.items() if tweet_id in filtered_ids)
    logging.debug('Done. {}s'.format(str(time() - t)))

    logging.info('Saving processed tweets')
    t = time()
    utils.save_text_tweets(filtered_tweets, args.out_file_text, new_annotations)

    if args.out_file_json is not None:
        utils.save_text_tweets(filtered_tweets, args.out_file_json, annotations)
    logging.debug('Done. {}s'.format(str(time() - t)))

    logging.info('Turned {} into {} tweets after filtering.'.format(str(len(raw_tweets)), str(len(filtered_tweets))))


def print_intro():
    print()
    print('Filtering unwanted tweets')
    print()
    print('Tweets data file:\t\t{}'.format(args.in_file))
    print()
    print('Output text file:\t\t{}'.format(args.out_file_text))

    if args.out_file_json is not None:
        print('Output JSON file:\t\t{}'.format(args.out_file_json))

    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Remove unwanted tweets')

    # Input file
    parser.add_argument('in_file', nargs='?', default='data/raw/tweets.json',
                        help='a JSON file containing tweets (default: data/raw/tweets.json)')

    # Output file
    parser.add_argument('out_file_text', nargs='?', default='data/filtered/tweets.txt',
                        help='text file to save filtered tweets to (default: data/filtered/tweets.txt)')

    parser.add_argument('out_file_json', nargs='?', default=None,
                        help='JSON file to save filtered tweets to (default: None)')

    # Directory
    parser.add_argument('--dir', nargs='?', default='.',
                        help='optional base directory for in_file and out_file')

    # Data parameters
    parser.add_argument('-l', '--labeled', action='store_true',
                        help='set if raw tweets are labeled (default: False)')

    # Logger verbosity parameters
    parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level, repeat to increase')
    parser.add_argument('-q', '--quiet', action='store_true', help='no print to console')

    args = parser.parse_args()
    args.in_file = path.join(args.dir, args.in_file)
    args.out_file_text = path.join(args.dir, args.out_file_text)

    if args.out_file_json is not None:
        args.out_file_json = path.join(args.dir, args.out_file_json)

    # Set logger verbosity level. Default is logging.INFO.
    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(len(levels) - 1, args.verbose + 2)]  # capped to number of levels
    logging.basicConfig(level=level, format='%(asctime)s\t%(levelname)s\t%(message)s')

    # Check if in_file exists
    if not path.isfile(args.in_file):
        logging.critical('\nCould not find file called {}'.format(args.in_file))
        exit(1)

    if args.quiet:
        logging.disable(levels[0])
    else:
        print_intro()

    main()
