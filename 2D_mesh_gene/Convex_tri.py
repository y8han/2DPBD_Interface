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
  points = []
  x = []
  y = []
  with open(file_dir + file, 'r') as f:
    data = f.readlines()
  inter = 0
  for line in data:
    odom = line.split()
    inter += 1
    if inter % 1 == 0: #22 is one of the best parameters
      inter = 0
      points.append((float(odom[0]), float(odom[1])))
      x.append(float(odom[0]))
      y.append(float(odom[1]))
  points.append(points[0])
  plt.plot(x, y)
  plt.show()
  #A = dict(vertices=np.array(((0, 0), (1, 0), (1, 1), (0, 1))))
  A = dict(vertices=np.array(points))
  print("Number of contour points:", len(points))
  B = tr.triangulate(A, 'qa0.05')
  tr.compare(plt, A, B)
  print("Number of vertices:", len(B['vertices']))
  vertices = B['vertices']
  output_vertex = open(file_dir + file + '_vertex.txt', 'w')
  for point in vertices:
    output_vertex.write(str(point[0]))
    output_vertex.write(" ")
    output_vertex.write(str(point[1]))
    output_vertex.write("\n")
  output_vertex.close()
  print("Number of triangles:", len(B['triangles']))
  output_constraint = open(file_dir + file + '_constraint.txt', 'w')
  constraints = B['triangles']
  triangle = [0, 1, -1]
  for constraint in constraints:
    for i in triangle:
      dist = norm(vertices[constraint[i]], vertices[constraint[i + 1]])
      output_constraint.write(str(constraint[i]))
      output_constraint.write(" ")
      output_constraint.write(str(constraint[i + 1]))
      output_constraint.write(" ")
      output_constraint.write(str(dist))
      output_constraint.write("\n")
  output_constraint.close()
  plt.show()
  with open(file_dir + file + '_constraint.txt', 'r') as f:
    data = f.readlines()
  index = 0
  for line in data:
    odom = line.split()
    index += 1
    print(float(odom[2]))
