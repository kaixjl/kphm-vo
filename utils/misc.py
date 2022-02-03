import time
import datetime
import os

def mkdir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def logic_implicate(x, y):
    # type: (bool, bool) -> bool
    return (not x) or y

class Timer:
    def __init__(self):
        self.time_start = None
        self.datetime_start = None

    def __enter__(self):
        self.time_start = time.time()
        self.datetime_start = datetime.datetime.now()
        print("datetime now {}".format(self.datetime_start.strftime("%Y-%m-%d %H:%M:%S.%f")))
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.print_now()

    def print_now(self):
        print("time pass {}s".format(time.time() - self.time_start))
        datetime_now = datetime.datetime.now()
        print("datetime now {}".format(datetime_now.strftime("%Y-%m-%d %H:%M:%S.%f")))
        print("datetime pass {}".format(datetime_now - self.datetime_start))

    @property
    def StartTime(self):
        return self.time_start

    @property
    def StartDatetime(self):
        return self.datetime_start

    def time(self):
        return time.time()

    def now(self):
        return datetime.datetime.now()