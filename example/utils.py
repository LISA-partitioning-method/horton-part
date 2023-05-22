import os

__all__ = ["load_fchk"]


def load_fchk(name):
    """Load fchk file

    Parameters
    ----------
    name:
        Molecular name.
    """
    return os.path.join("data", "{}.fchk".format(name.lower()))
