import os


def check_and_create_dir(path):
    if not os.path.exists(path):
        if not os.path.exists(os.path.dirname(path)):
            check_and_create_dir(os.path.dirname(path))
        print("create dir %s" % path)
        os.mkdir(path)
