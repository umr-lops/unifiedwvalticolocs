import logging

from yaml import CLoader as Loader
from yaml import load

logger = logging.getLogger("unifiedwvalticolocs.utils")
logger.addHandler(logging.NullHandler())


def get_conf_content(conf_path) -> dict:
    """

    Load the YAML configuration file from the specified path and return its content as a dictionary.

    Args:
        conf_path (str): The file path to the YAML configuration file.

    Returns:
        dict: The content of the YAML configuration file as a dictionary.

    """
    stream = open(conf_path)
    conf = load(stream, Loader=Loader)
    return conf
