import errno
import os


def make_directory(directory):
    if not os.path.isdir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if errno.EEXIST != e.errno:
                raise
