class AccuracyRecorder():
    def __init__(self):
        self.ttrue_num = 0
        self.tfalse_num = 0
    
    def col_accuracy(self, total):
        return (self.ttrue_num+self.tfalse_num) / total
    
    def count_true(self):
        self.ttrue_num += 1
    
    def count_false(self):
        self.tfalse_num += 1
    
import random
import numpy as np
if __name__ == "__main__":
    recorders = []
    total = 0
    c = [0, 2, 1, 2, 0]
    p = [1, 2, 0, 2, 1]
    for i in range(3):
        recorders.append(AccuracyRecorder())
    for i in range(5):
        total += 1
        # c = random.randint(0, 4)
        # p = random.randint(0, 4)
        if p[i] == c[i]:
            recorders[p[i]].count_true()
            continue

        for j in range(len(recorders)):
            pass
            # if i = 
