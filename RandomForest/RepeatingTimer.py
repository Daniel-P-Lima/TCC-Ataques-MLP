from threading import Timer

class RepeatingTimer:
    def __init__(self, timeout:float, func, args:list):
        self.timer = None
        self.timeout = timeout
        self.func = func
        self.args = args
    
    def _run(self):
        self.func(*self.args)
        self.start()
    
    def start(self):
        self.timer = Timer(self.timeout, self._run)
        self.timer.start()
    
    def cancel(self):
        if self.timer:
            self.timer.cancel()
