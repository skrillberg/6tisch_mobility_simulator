import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
files = ['44022a94-545c-4ffd-9a08-4aff653bb98d_b128_lr1.0_clip10.csv',
'b9024147-1924-4a0e-8b35-8ac31e802794_b64_lr1.0_clip10.csv',
'1b557bfe-1e05-4f5f-867f-f996e4938928_b32_lr1.0_clip10.csv',
'8e5ae770-b456-435b-9004-b29f97208622_b16_lr1.0_clip10.csv']

data = {}
for file in files:
	data[file]=pd.read_csv('gymSim/'+file)



def moving_avg(data):
	N=30
	cumsum, moving_aves = [0], []
	std = []
	for i, x in enumerate(data, 1):
	    cumsum.append(cumsum[i-1] + x)
	    if i>=N:
	        moving_ave = (cumsum[i] - cumsum[i-N])/N
	        #can do stuff with moving_ave here
	        moving_aves.append(moving_ave)

	return moving_aves
legends = []

for key in data:
	moving_ave = moving_avg(data[key]['0'])
	plt.plot(moving_ave)
	legends.append(key)
plt.ylim([-500,0])
plt.legend(legends)
plt.show()