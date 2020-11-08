import datetime



class TimeHandler:
    def __init__(self, ):
        self.start_time = datetime.datetime.now()

    def getNow():
        return self.datetime.datetime.now()

    def getElapsedTime(self, elapsed_from='start', is_now=True):
        now = datetime.datetime.now()
        if elapsed_from == 'start':
            elapsed_time = now - self.start_time
        elif elapsed_from == 'prev':
            elapsed_time = now - self.prev_time
        self.prev_time = now 

        if is_now:
            return now, elapsed_time
        else:
            return elapsed_time
    
    
        
