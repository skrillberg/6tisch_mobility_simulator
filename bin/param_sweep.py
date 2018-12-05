import RLServer_open_ev
import os
import sys

here = sys.path[0]
sys.path.insert(0, os.path.join(here, '..'))

batch_list=[16,32,64,128]
lr_list=[0.1,1,10,100]
clip_list = [1,10,100,1000]

b = 32
clip = 10 
for lr in lr_list:
	os.system('python RLServer_open_ev.py --batch_size {0} --lr_mult {1} --clip {2}'.format(b,lr,clip))

lr = 1
clip = 10
'''
for b in batch_list:
	os.system('python RLServer_open_ev.py --batch_size {0} --lr_mult {1} --clip {2}'.format(b,lr,clip))


'''