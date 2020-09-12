import os
import time
import os.path as op


def mkdir(dir_path):
    """Create directory.
    
    Args:
        dir_path (str)
    """
    assert not op.exists(dir_path), ("Dir already exists!")
    os.makedirs(dir_path)


# ==========
# Time
# ==========


def get_timestr():
    """Return current time str."""
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


class Timer():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.accum_time = 0

    def restart(self):
        self.start_time = time.time()

    def accum(self):
        self.accum_time += time.time() - self.start_time

    def get_time(self):
        return time.time()

    def get_interval(self):
        return time.time() - self.start_time
    
    def get_accum(self):
        return self.accum_time


class Counter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.time = 0
        self.accum_volume = 0

    def accum(self, volume):
        self.time += 1
        self.accum_volume += volume

    def get_ave(self):
        return self.accum_volume / self.time
