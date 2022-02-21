#Adapted from https://github.com/silverbottlep/abid_challenge/blob/master/counting/evaluate.py
#Used https://stackoverflow.com/questions/11295171/reading-two-text-files-line-by-line-simultaneously 

import json
import numpy as np
#from itertools import zip

n = 0
num_classes = 5
perclass_correct = np.zeros(num_classes) 
perclass_dist = np.zeros(num_classes) 
perclass_N = np.zeros(num_classes)

with open('counting_result.txt') as f1, open('counting_label.txt') as f2:
    for x, y in zip(f1, f2):
        pred = int(x.strip())
        gt = int(y.strip()) 
        perclass_correct[gt] = perclass_correct[gt] + int(pred==gt) 
        perclass_dist[gt] = perclass_dist[gt] + np.power(pred-gt,2)
        perclass_N[gt] = perclass_N[gt] + 1
        n = n+1

print('Accuracy')
print('%d/%d (%f)' %(perclass_correct.sum(), perclass_N.sum(), perclass_correct.sum()/perclass_N.sum()))
print('RMSE(Root mean squared error)')
print(np.sqrt(perclass_dist.sum()/perclass_N.sum()))
print('Per class accuracy')
print(perclass_correct/perclass_N)
print('Per class RMSE')
print(np.sqrt(perclass_dist/perclass_N))
print('n =', n) 

