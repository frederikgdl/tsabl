import logging
import classifiers.train_and_test as train_and_test

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='tsabl - train classifiers')

    # Logger verbosity parameters
    parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level, repeat to increase')
    parser.add_argument('-q', '--quiet', action='store_true', help='no print to console')

    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(len(levels) - 1, args.verbose + 2)]  # capped to number of levels
    logging.basicConfig(level=level, format="%(asctime)s\t%(levelname)s\t%(message)s")

    args.skip_testing = True
    args.skip_training = False

    if args.quiet:
        logging.disable(levels[0])
    else:
        train_and_test.print_intro(args)

    train_and_test.main(args)
