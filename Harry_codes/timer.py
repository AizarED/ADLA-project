from time import time

class Timer():
    def __init__(self):
        self.profile = {}

    def start(self, event_name):
        if event_name not in self.profile:
            self.profile[event_name] = {}
        self.profile[event_name]['start'] = time()
        if 'calls' in self.profile[event_name]:
            self.profile[event_name]['calls'] += 1
        else:
            self.profile[event_name]['calls'] = 1

    def stop(self, event_name):
        try:
            assert event_name in self.profile
        except AssertionError as e:
            raise Exception('Make sure that you started an event with the same name')
        d = self.profile[event_name]
        new_time = round(time() - d['start'], 6)

        n_calls = d['calls']
        if n_calls < 1:
            prev_times = d['avg time']
        else:
            prev_times = 0
        avg_time = round(  ( ((n_calls - 1) * prev_times) + new_time) / n_calls, 6)
        self.profile[event_name] = {'avg time': avg_time, 'calls': n_calls}
        # implement max, min and sd of times

    def show(self):
        print('\nTIME PROFILE:')
        for event in self.profile:
           # print('\t{event}: {self.profile[event]}')
            print("\t{0}:{1}".format(event, self.profile[event]))
