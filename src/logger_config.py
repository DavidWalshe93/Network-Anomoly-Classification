"""
Author:         David Walshe
Date:           12/04/2020   
"""

import logging


def setup_logger(logger):
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler("run.log", mode="w")

    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")

    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.setLevel(logging.INFO)

    return logger
