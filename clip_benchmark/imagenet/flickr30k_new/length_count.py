import numpy as np

f = open('./flickr30k_test_karpathy.txt', 'r')
lines = f.readlines()[1:]
length = []

for line in lines:
    line = line.strip().split(',')
    caption = line[1]
    length.append(len(caption.split()))

length = np.array(length)

print(length.mean(), length.std())
