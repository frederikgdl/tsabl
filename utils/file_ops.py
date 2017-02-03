# Read and strip lines in file
def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def read_twitter_id_file(file_path, labeled=False):
    """
    Return a list of id strings from text file
    :param file_path: The path of the input file
    :param labeled: Set to true if ids have an annotation associated on each line
    :return: A list of id strings
    """
    ids = []
    with open(file_path) as f:
        for line in f:
            ids.append(line.strip().split()[0] if labeled else line.strip())
    return ids
