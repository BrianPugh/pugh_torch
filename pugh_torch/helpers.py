import re


def camel_to_snake(s):
    """ Converts a camelCase or pascalCase string to snake_case

    Parameters
    ----------
    s : str
        Some camel/pascal case string to convert to snake case

    Returns
    -------
    str
        camel_case equivalent of the provided string.
    """

    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()
