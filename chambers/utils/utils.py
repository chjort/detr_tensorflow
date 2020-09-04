import datetime


def timestamp_now():
    dt = str(datetime.datetime.now())
    return "_".join(dt.split(" "))
