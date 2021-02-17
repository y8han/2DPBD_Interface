import matplotlib.pyplot as plt
import numpy as np
import triangle as tr
import math
import sys
import os
import matplotlib

file_dir = sys.argv[1]
files = os.listdir(file_dir)
def norm(a, b):
  return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

for file in files:
  if file[-3:] == 'txt':
    points = []
    x = []
    y = []
    with open(file_dir + file, 'r') as f:
      data = f.readlines()
    inter = 0
    Intervel = 1
    previous_point = -1
    for line in data:
      odom = line.split()
      inter += 1
      if inter % Intervel == 0: #22 is one of the best parameters
        inter = 0
        points.append((float(odom[0]), float(odom[1])))
        x.append(float(odom[0]))
        y.append(float(odom[1]))
        if previous_point == -1:
          previous_point = [float(odom[0]),float(odom[1])]
        else:
          distance = norm([float(odom[0]),float(odom[1])], previous_point)
          previous_point = [float(odom[0]),float(odom[1])]
          print(distance)
          if distance <= 0.01:
            Intervel = 6
    #generate poly file that describes the outer geometric property of the mesh
    plt.plot(x, y)
    if os.path.exists(file_dir + file[0:-4] + '.poly'):
      os.remove(file_dir + file[0:-4] + '.poly')
    f = open(file_dir + file[0:-4] + '.poly', 'w')
    f.write(str(len(points)) + ' ' + '2 0 0\n')
    for i in range(len(points)):
      f.write(str(i+1) + ' ' + str(x[i]) + ' ' + str(y[i]) + '\n')
    f.write(str(len(points)) + ' 0\n')
    for i in range(len(points)):
      if i != len(points) - 1:
        f.write(str(i+1) + ' ' + str(i+1) + ' ' + str(i+2) + '\n')
      else:
        f.write(str(i+1) + ' ' + str(i+1) + ' ' + str(1) + '\n')
    f.write('0\n')
    plt.show()
