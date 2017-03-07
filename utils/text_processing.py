import preprocessor
from lib.twokenize import twokenize


def clean_twitter_tokens(text):
    """
    Removes URLs, Hashtags and @-mentions from argument text
    :param text: Text to clean
    :return: Cleaned text
    """
    preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.HASHTAG, preprocessor.OPT.MENTION)
    return preprocessor.clean(text)


def clean_and_twokenize(text):
    """
    Clean Urls, Hashtags and @-mentions, then tokenize using Twokenize
    :param text:
    :return:
    """
    cleaned_text = clean_twitter_tokens(text)
    twokenized_text = twokenize.tokenize(cleaned_text)

    return twokenized_text
