import os
import numpy as np
import sys
file = open(sys.argv[1],'r')
lines = file.readlines()
file.close()

lines = [line.strip() for line in lines]
# print(lines)
def func(string):
    ans = ""
    i = 0
    while i < len(string):
        if string[i] == '[':
            while(string[i] != ']'):
                i+=1
            i+=1
            ans += (" "+string[i])
            i+=1
        else:
            ans += string[i]
            i+=1
    return [float(j) for j in ans.split()]

results = [func(lines[i]) for i in range(len(lines))]

results = np.array(results)

print("Best Test:",round(np.mean(results[:,0], axis=0),5), round(np.std(results[:,0]),5))
print("Best Test final epoch:",np.mean(results[:,1], axis=0), np.std(results[:,1]))
print("Best Test cheat:",np.mean(results[:,2], axis=0), np.std(results[:,2]))

