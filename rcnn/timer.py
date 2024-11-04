import time
# import numpy as np

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()

    def get_time(self):
        return self.end_time - self.start_time

    def reset(self):
        self.start_time = None
        self.end_time = None

    def __str__(self):
        return str(self.get_time())
    

# timer = Timer()
# timer.start()
# time.sleep(2)
# timer.end()
# print(f"{timer}")