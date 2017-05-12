def sign(x):
    """
    Get the sign of number x
    
    :param x: A number
    :return: The sign of x. 1 for positive numbers, -1 for negative numbers and 0 for 0. 
    """
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def sorted_by_suffix(list_of_strings):
    try:
        return sorted(list_of_strings, key=lambda x: float(x.split("-")[-1]))
    except ValueError:
        return sorted(list_of_strings)
