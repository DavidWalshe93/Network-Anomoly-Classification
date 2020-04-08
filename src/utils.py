"""
Author:         David Walshe
Date:           09/04/2020   
"""


def refactor_names(names, features):
    for i, feature in enumerate(features):
        for j, name in enumerate(names):
            if name.find(f"x{i}") > -1:
                name = name.replace(f"x{i}_", f"[{feature}] ")
                name = name.replace("b'", "")
                name = name.replace("'", "")

                names[j] = name

    return names