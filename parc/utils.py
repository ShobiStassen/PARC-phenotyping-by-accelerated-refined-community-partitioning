def get_mode(a_list: list[any]) -> any:
    """Get the value which appears most often in the list (the mode).

    If multiple items are maximal, the function returns the first one encountered
    (not necessarily the first one in the list).

    Args:
        a_list: A list with values.

    Returns:
        The most frequent value in the list
    """
    return max(set(a_list), key=a_list.count)
