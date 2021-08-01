import taichi as ti
import math as math
import fcl
import numpy as np
import time
import os
import sys

ti.init(debug=False,arch=ti.cpu)
real = ti.f32 #data type f32 -> float in C

max_num_particles = 1000
dt = 1e-2#simulation time step(important) -> adjustable
dt_inv = 1 / dt
dim = 2
pbd_num_iters =30#Iteration number(important) -> adjustable
max_triangle = 10000

scalar = lambda: ti.var(dt=real) #2D dense tensor
vec = lambda: ti.Vector(dim, dt=real) #2*1 vector(each element in a tensor)
mat = lambda: ti.Matrix(dim, dim, dt=real) #2*2 matrix(each element in a tensor)

num_particles = ti.var(ti.i32, shape=())
paused = ti.var(ti.i32, shape=())
damping = ti.var(ti.f32, shape=())
stiffness = ti.var(ti.f32, shape=())
LidarMaxDistance = ti.var(ti.f32, shape=())
num_trian = ti.var(ti.i32, shape=())
Area_parameter = ti.var(ti.f32, shape=())
Momentum_Energy = ti.var(ti.f32, shape=())
Line_function_A = ti.var(ti.f32,shape=())
Line_function_B = ti.var(ti.f32,shape=())
Line_function_C = ti.var(ti.f32,shape=())
Line_function_D = ti.var(ti.f32,shape=())
PointP1_x = ti.var(ti.f32,shape=())
PointP2_x = ti.var(ti.f32,shape=())
clearance = ti.var(ti.f32,shape=())
test = ti.var(ti.f32,shape=())
maximum_constraints = 50

trian_volumn = scalar()
triangle_area = scalar()
volumn_constraint_num = ti.var(ti.i32)
volumn_constraint_list = scalar()

Delta_x_sequence = scalar()
Delta_y_sequence = scalar()

bottom_y = 0.0232
bottom_x = 1.00
epsolon = 0.0001 #digit accurary(important) -> adjustable
area_epsolon = 1e-11

x, v, old_x, star_x = vec(), vec(), vec(), vec()
actuation_type = scalar()
total_constraint = ti.var(ti.i32, shape=())
# rest_length[i, j] = 0 means i and j are not connected
mass = scalar()
rest_length = scalar()
position_delta_tmp = vec()
position_delta_sum = vec()
constraint_neighbors = ti.var(ti.i32)
constraint_num_neighbors = ti.var(ti.i32)

Gravity = -9.8
gravity = [0, Gravity] #direction
H_force = [0, 0] #another gr
previous_minimum_distance = -10000

@ti.layout  #Environment layout(placed in ti.layout) initialization of the dimensiond of each tensor variables(global)
def place():
    ti.root.dense(ti.ij, (max_num_particles, max_num_particles)).place(rest_length)
    ti.root.dense(ti.i, max_triangle).place(volumn_constraint_list)
    ti.root.dense(ti.ij, (max_triangle, 4)).place(trian_volumn)
    ti.root.dense(ti.i, max_triangle).place(triangle_area)
    ti.root.dense(ti.i, max_num_particles).place(mass, volumn_constraint_num, x, v, old_x, star_x, actuation_type, position_delta_tmp, position_delta_sum, Delta_x_sequence, Delta_y_sequence) #initialzation to zero
    nb_node = ti.root.dense(ti.i, max_num_particles)
    nb_node.place(constraint_num_neighbors)
    nb_node.dense(ti.j, maximum_constraints).place(constraint_neighbors)

@ti.kernel
def old_posi(n: ti.i32):
    for i in range(n):
        old_x[i] = x[i]

@ti.kernel
def star_posi(n: ti.i32):
    for i in range(n):
        star_x[i] = x[i]

@ti.kernel
def find_constraint(n: ti.i32):
    for i in range(n):
        nb_i = 0
        for j in range(n):
            if rest_length[i, j] != 0: #spring-constraint
                x_ij = x[i] - x[j]
                dist_diff = abs(x_ij.norm() - rest_length[i, j])
                if dist_diff >= epsolon:
                    constraint_neighbors[i, nb_i] = j
                    nb_i += 1
        constraint_num_neighbors[i] = nb_i

@ti.kernel
def find_area_constraint(n: ti.i32):
    for i in range(n):
        p1_index = trian_volumn[i, 0]
        p2_index = trian_volumn[i, 1]
        p3_index = trian_volumn[i, 2]
        p1 = x[int(p1_index)]
        p2 = x[int(p2_index)]
        p3 = x[int(p3_index)]
        p10 = p1.x
        p11 = p1.y
        p20 = p2.x
        p21 = p2.y
        p30 = p3.x
        p31 = p3.y
        area = 0.5 * ((p10 - p20)*(p11 - p31) - (p11 - p21)*(p10 - p30))
        if(abs(abs(area) - trian_volumn[i, 3]) > area_epsolon):
            volumn_constraint_list[i] = area
            volumn_constraint_num[int(p1_index)] += 1
            volumn_constraint_num[int(p2_index)] += 1
            volumn_constraint_num[int(p3_index)] += 1
        else:
            volumn_constraint_list[i] = 0.0


@ti.kernel
def substep(n: ti.i32): # Compute force and new velocity
    for i in range(n):
        if actuation_type[i] == 1:
            v[i] *= ti.exp(-dt * damping[None]) # damping
            total_force = ti.Vector(gravity) * mass[i] #gravity -> accelaration
            v[i] += dt * total_force / mass[i]  #dv = dt*a
        if actuation_type[i] == 2: #control points by the cylinder
            x[i].x += Delta_x_sequence[i]
            x[i].y += Delta_y_sequence[i]

@ti.kernel
def collision_check(n: ti.i32):# Collide with ground
    for i in range(n):
        p_x = x[i].x
        p_y = x[i].y
        if p_y < bottom_y:
            x[i].y = bottom_y
            v[i].y = 0
        if p_y > 1 - bottom_y:
            x[i].y = 1 - bottom_y
            v[i].y = 0
        # The cylinder is modelled as an obstacle
        project_t = (-Line_function_C[None] - p_x * Line_function_A[None] - p_y * Line_function_B[None]) / Line_function_D[None] ** 2
        project_x = p_x + project_t * Line_function_A[None]
        if project_x >= PointP2_x[None] and project_x < PointP1_x[None]:
            distance = np.abs(Line_function_A[None]*p_x+Line_function_B*p_y+Line_function_C[None])/Line_function_D[None]
            if distance <= clearance[None]:
                project_t = project_t / distance * (clearance[None] - distance)
                x[i].x = p_x - project_t * Line_function_A[None]
                x[i].y = p_y - project_t * Line_function_B[None]

@ti.kernel
def Position_update(n: ti.i32):# Compute new position
    for i in range(n):
        if actuation_type[i] == 1:
            x[i] += v[i] * dt

@ti.kernel
def stretch_constraint(n: ti.i32):
    for i in range(n):
        pos_i = x[i]
        posi_tmp = ti.Vector([0.0, 0.0])
        mass_i_inv = 1 / mass[i]
        for j in range(constraint_num_neighbors[i]):
            p_j = constraint_neighbors[i, j]
            pos_j = x[p_j]
            x_ij = pos_i - pos_j
            dist_diff = x_ij.norm() - rest_length[i, p_j]
            grad = x_ij.normalized()
            mass_j_inv = 1 / mass[p_j]
            mass_ij_inv = 1 / (mass_i_inv + mass_j_inv)
            position_delta = -stiffness[None] * mass_i_inv * mass_ij_inv * dist_diff * grad / constraint_num_neighbors[i]
            posi_tmp += position_delta
        position_delta_tmp[i] = posi_tmp

@ti.kernel
def area_constraint(n: ti.i32):
    for i in range(n):
        if(volumn_constraint_list[i] != 0.0):
            diff_volumn = abs(volumn_constraint_list[i]) - trian_volumn[i, 3]
            p1_index = trian_volumn[i, 0]
            p2_index = trian_volumn[i, 1]
            p3_index = trian_volumn[i, 2]
            #position of each particle
            p1 = x[int(p1_index)]
            p2 = x[int(p2_index)]
            p3 = x[int(p3_index)]
            p10 = p1.x
            p11 = p1.y
            p20 = p2.x
            p21 = p2.y
            p30 = p3.x
            p31 = p3.y
            if volumn_constraint_list[i] < 0:  #area smaller than 0
                grad_x1 = p31/2 - p21/2
                grad_y1 = p20/2 - p30/2
                grad_x2 = p11/2 - p31/2
                grad_y2 = p30/2 - p10/2
                grad_x3 = p21/2 - p11/2
                grad_y3 = p10/2 - p20/2
                grad_p1 = ti.Vector([grad_x1, grad_y1])
                tmp_p1 = grad_p1.norm() * grad_p1.norm()
                w_p1 = 1 / mass[int(p1_index)]
                grad_p2 = ti.Vector([grad_x2, grad_y2])
                tmp_p2 = grad_p2.norm() * grad_p2.norm()
                w_p2 = 1 / mass[int(p2_index)]
                grad_p3 = ti.Vector([grad_x3, grad_y3])
                tmp_p3 = grad_p3.norm() * grad_p3.norm()
                w_p3 = 1 / mass[int(p3_index)]
                denominator = w_p1 * tmp_p1 + w_p2 * tmp_p2 + w_p3 * tmp_p3
                constraint_lambda = diff_volumn / denominator
                delta_p1 = -constraint_lambda * w_p1 * grad_p1 / volumn_constraint_num[int(p1_index)]
                delta_p2 = -constraint_lambda * w_p2 * grad_p2 / volumn_constraint_num[int(p2_index)]
                delta_p3 = -constraint_lambda * w_p3 * grad_p3 / volumn_constraint_num[int(p3_index)]
                position_delta_tmp[int(p1_index)] += delta_p1
                position_delta_tmp[int(p2_index)] += delta_p2
                position_delta_tmp[int(p3_index)] += delta_p3
            else:
                grad_x1 = p21/2 - p31/2
                grad_y1 = p30/2 - p20/2
                grad_x2 = p31/2 - p11/2
                grad_y2 = p10/2 - p30/2
                grad_x3 = p11/2 - p21/2
                grad_y3 = p20/2 - p10/2
                grad_p1 = ti.Vector([grad_x1, grad_y1])
                tmp_p1 = grad_p1.norm() * grad_p1.norm()
                w_p1 = 1 / mass[int(p1_index)]
                grad_p2 = ti.Vector([grad_x2, grad_y2])
                tmp_p2 = grad_p2.norm() * grad_p2.norm()
                w_p2 = 1 / mass[int(p2_index)]
                grad_p3 = ti.Vector([grad_x3, grad_y3])
                tmp_p3 = grad_p3.norm() * grad_p3.norm()
                w_p3 = 1 / mass[int(p3_index)]
                denominator = w_p1 * tmp_p1 + w_p2 * tmp_p2 + w_p3 * tmp_p3
                constraint_lambda = diff_volumn / denominator
                delta_p1 = -constraint_lambda * w_p1 * grad_p1 / volumn_constraint_num[int(p1_index)]
                delta_p2 = -constraint_lambda * w_p2 * grad_p2 / volumn_constraint_num[int(p2_index)]
                delta_p3 = -constraint_lambda * w_p3 * grad_p3 / volumn_constraint_num[int(p3_index)]
                position_delta_tmp[int(p1_index)] += delta_p1
                position_delta_tmp[int(p2_index)] += delta_p2
                position_delta_tmp[int(p3_index)] += delta_p3

@ti.kernel
def apply_position_deltas(n: ti.i32):
    for i in range(n):
        if actuation_type[i] == 1:
            x[i] += position_delta_tmp[i]

@ti.kernel
def updata_velosity(n: ti.i32): #updata velosity after combining constraints
    for i in range(n):
        v[i] = (x[i] - old_x[i]) * dt_inv

@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32, type: ti.i32): # Taichi doesn't support using Matrices as kernel arguments yet
    actuation_type[num_particles[None]] = type
    if type == -1:
        mass[num_particles[None]] = 100000000
    else:
        mass[num_particles[None]] = 0.0001
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    num_particles[None] += 1

@ti.kernel
def new_costraint(p1_index: ti.i32, p2_index: ti.i32, dist: ti.f32):
    # Connect with existing particles
    rest_length[p1_index, p2_index] = dist
    rest_length[p2_index, p1_index] = dist

@ti.kernel
def new_trian(p1_index: ti.i32, p2_index: ti.i32, p3_index: ti.i32):
    new_trian_id = num_trian[None]
    trian_volumn[new_trian_id, 0] = float(p1_index)
    trian_volumn[new_trian_id, 1] = float(p2_index)
    trian_volumn[new_trian_id, 2] = float(p3_index)
    p1 = x[p1_index]
    p2 = x[p2_index]
    p3 = x[p3_index]
    p10 = p1.x
    p11 = p1.y
    p20 = p2.x
    p21 = p2.y
    p30 = p3.x
    p31 = p3.y
    area = 0.5 * abs((p10 - p20)*(p11 - p31) - (p11 - p21)*(p10 - p30))
    trian_volumn[new_trian_id, 3] = area
    num_trian[None] += 1

@ti.kernel
def Compute_area(p1_index: ti.i32, p2_index: ti.i32, p3_index: ti.i32, new_trian_id: ti.i32):
    trian_volumn[new_trian_id, 0] = float(p1_index)
    trian_volumn[new_trian_id, 1] = float(p2_index)
    trian_volumn[new_trian_id, 2] = float(p3_index)
    p1 = x[p1_index]
    p2 = x[p2_index]
    p3 = x[p3_index]
    p10 = p1.x
    p11 = p1.y
    p20 = p2.x
    p21 = p2.y
    p30 = p3.x
    p31 = p3.y
    area = 0.5 * abs((p10 - p20)*(p11 - p31) - (p11 - p21)*(p10 - p30))
    triangle_area[new_trian_id] = area

@ti.kernel
def move_obstacle(n: ti.i32, delta_x: ti.f32, delta_y: ti.f32):
    for i in range(n):
        x[i].x += delta_x
        x[i].y += delta_y

@ti.kernel
def paint():
    for i, j, k in pixels:
        pixels[i, j, k] =  255

@ti.kernel
def Compute_Momentum_Energy(start: ti.i32, end: ti.i32):
    for i in range(start, end):
        if actuation_type[i] != -1:
            Momentum_Energy[None] += 0.5 * mass[i] * (x[i] - star_x[i]).norm() * (x[i] - star_x[i]).norm()


def forward(n, number_triangles):
    #the first three steps -> only consider external force
    old_posi(n)
    substep(n)
    Position_update(n)
    #print(x.to_numpy()[0:n] - old_x.to_numpy()[0:n])
    collision_check(n)
    star_posi(n)
    #print(x.to_numpy()[0:n])
    #add constraints
    for i in range(pbd_num_iters):
        constraint_neighbors.fill(-1)
        find_constraint(n)
        volumn_constraint_num.fill(0)
        find_area_constraint(number_triangles)
        #print("This is ", i , "th iteration.")
        stretch_constraint(n)
        area_constraint(number_triangles)
        apply_position_deltas(n)
        collision_check(n)
    updata_velosity(n)

gui = ti.GUI('Mass Spring System', res=(640, 640), background_color=0xdddddd)
pixels = ti.field(ti.u8, shape=(640,640, 3))

def CheckRepeatMesh(i,j,k,lists):
    if [i,j,k] in lists:
        return False
    elif [i,k,j] in lists:
        return False
    elif [j,i,k] in lists:
        return False
    elif [j,k,i] in lists:
        return False
    elif [k,j,i] in lists:
        return False
    elif [j,i,j] in lists:
        return Fakse
    else:
        return True

def CheckCollison(rotate_direction, verts, tris, stick, rota_degree, trans_x, trans_y, tolerance):
    global previous_minimum_distance
    mesh = fcl.BVHModel()
    mesh.beginModel(len(verts), len(tris))
    mesh.addSubModel(verts, tris)
    mesh.endModel()
    mesh_obj = fcl.CollisionObject(mesh)
    objs = [mesh_obj, stick]
    manager = fcl.DynamicAABBTreeCollisionManager()
    manager.registerObjects(objs)
    manager.setup()
    crequest = fcl.CollisionRequest(enable_contact=True)
    drequest = fcl.DistanceRequest(enable_nearest_points=True)
    cdata = fcl.CollisionData(crequest, fcl.CollisionResult())
    ddata = fcl.DistanceData(drequest, fcl.DistanceResult())
    manager.collide(cdata, fcl.defaultCollisionCallback)
    manager.distance(ddata, fcl.defaultDistanceCallback)
    #print("Is collision: ", cdata.result.is_collision)
    minimum_distance = ddata.result.min_distance
    nearest_point = ddata.result.nearest_points[0][:2]
    nearest_point_id = -1
    for i in range(verts.shape[0]):
        if nearest_point[0] == verts[i][0] and nearest_point[1] == verts[i][1]:
            nearest_point_id = i
    # nearest_point_stick = ddata.result.nearest_points[1]
    # nearest_point_stick = Transform(rota_degree, trans_x, trans_y, nearest_point_stick)
    collision = -10000
    if previous_minimum_distance != -10000:
        if minimum_distance == -1:
            collision = -1
            print("The tip of the stick is inside the soft tissues.")
        elif minimum_distance >= tolerance:
            collision = 0
            print("Tip:No collision")
        # elif minimum_distance >= 0 and minimum_distance < tolerance and minimum_distance < previous_minimum_distance:
        elif minimum_distance >= 0 and minimum_distance < tolerance:
            print("Tip:Collision")
            collision = 1
        # if nearest_point in verts:
        #     print("nearest point of the stick:", nearest_point_stick)
    tmp = previous_minimum_distance
    previous_minimum_distance = minimum_distance
    return nearest_point, collision, minimum_distance, nearest_point_id

    #print(ddata.result.pos)
    # if cdata.result.is_collision:
    #     for contact in cdata.result.contacts:
    #         print("Contact pos:", contact.pos) #the contact point is defined as one of the
    #         print(contact.pos in verts)
    #         print("Contact nor:", contact.normal)
    #         print("Penetra depth:", contact.penetration_depth)

def BodyCheckCollison(rotate_direction, verts, tris, body, rota_degree, tolerance):
    global previous_minimum_distance
    mesh = fcl.BVHModel()
    mesh.beginModel(len(verts), len(tris))
    mesh.addSubModel(verts, tris)
    mesh.endModel()
    mesh_obj = fcl.CollisionObject(mesh)
    objs = [mesh_obj, body]
    manager = fcl.DynamicAABBTreeCollisionManager()
    manager.registerObjects(objs)
    manager.setup()
    crequest = fcl.CollisionRequest(enable_contact=True)
    drequest = fcl.DistanceRequest(enable_nearest_points=True)
    cdata = fcl.CollisionData(crequest, fcl.CollisionResult())
    ddata = fcl.DistanceData(drequest, fcl.DistanceResult())
    manager.collide(cdata, fcl.defaultCollisionCallback)
    manager.distance(ddata, fcl.defaultDistanceCallback)
    minimum_distance = ddata.result.min_distance
    nearest_point = ddata.result.nearest_points[0][:2]
    collision = -10000
    if previous_minimum_distance != -10000:
        if minimum_distance == -1:
            collision = -1
            print("The body of the stick is inside the soft tissues.")
        elif minimum_distance >= tolerance:
            collision = 0
            print("Body:No collision")
        elif minimum_distance >= 0 and minimum_distance < tolerance and minimum_distance < previous_minimum_distance:
        # elif minimum_distance >= 0 and minimum_distance < tolerance:
            print("Body:Collision")
            collision = 1
        # if nearest_point in verts:
        #     print("nearest point of the stick:", nearest_point_stick)
    tmp = previous_minimum_distance
    previous_minimum_distance = minimum_distance
    return nearest_point, collision, minimum_distance



def Transform(rota_degree, trans_x, trans_y, point):
    rotation = np.array([[np.cos(rota_degree/180*np.pi), -np.sin(rota_degree/180*np.pi), 0.0],
                         [np.sin(rota_degree/180*np.pi), np.cos(rota_degree/180*np.pi), 0.0],
                         [0.0, 0.0, 1.0]])
    translation = np.array([trans_x, trans_y, 0.0])
    transform_matrix = np.c_[rotation,translation]
    transform_matrix = np.row_stack((transform_matrix, np.array([0,0,0,1])))
    point_homo = np.r_[np.array(point),np.array([1,])]
    point_trans = transform_matrix @ point_homo
    point_trans = point_trans[:2]
    return point_trans

def lidar_configuration(trans_x, trans_y, radius):
    rotation = np.array([[1,0,0],[0,1,0],[0,0,1]])
    translation = np.array([trans_x, trans_y, 0.0])
    Transform = fcl.Transform(rotation, translation)
    lidar = fcl.CollisionObject(fcl.Cylinder(radius, 0), Transform)
    return lidar

def stick_configuration(rota_degree, trans_x, trans_y, new_trans_x, new_trans_y, length, width, top_left, top_right, bottom_left, bottom_right):
    rotation = np.array([[np.cos(rota_degree/180*np.pi), -np.sin(rota_degree/180*np.pi), 0.0],
                         [np.sin(rota_degree/180*np.pi), np.cos(rota_degree/180*np.pi), 0.0],
                         [0.0, 0.0, 1.0]])
    translation = np.array([trans_x, trans_y, 0.0])
    new_translation = np.array([new_trans_x, new_trans_y, 0.0])
    Transform = fcl.Transform(rotation, new_translation)
    #before transform
    stick = fcl.CollisionObject(fcl.Box(length, width, 0.0), Transform) #x,y,z length center at the origin
    #after transform
    transform_matrix = np.c_[rotation,translation]
    transform_matrix = np.row_stack((transform_matrix, np.array([0,0,0,1])))
    top_left_homo = np.r_[top_left,np.array([1,])]
    top_right_homo = np.r_[top_right,np.array([1,])]
    bottom_left_homo = np.r_[bottom_left,np.array([1,])]
    bottom_right_homo = np.r_[bottom_right,np.array([1,])]
    top_left_trans = transform_matrix @ top_left_homo
    top_left_trans = top_left_trans[:2]
    top_right_trans = transform_matrix @ top_right_homo
    top_right_trans = top_right_trans[:2]
    bottom_left_trans = transform_matrix @ bottom_left_homo
    bottom_left_trans = bottom_left_trans[:2]
    bottom_right_trans = transform_matrix @ bottom_right_homo
    bottom_right_trans = bottom_right_trans[:2]
    return transform_matrix, [top_left_trans, top_right_trans, bottom_right_trans, bottom_left_trans], stick

def body_configuration(rota_degree, new_trans_x, new_trans_y, length, width):
    rotation = np.array([[np.cos(rota_degree/180*np.pi), -np.sin(rota_degree/180*np.pi), 0.0],
                         [np.sin(rota_degree/180*np.pi), np.cos(rota_degree/180*np.pi), 0.0],
                         [0.0, 0.0, 1.0]])
    new_translation = np.array([new_trans_x, new_trans_y, 0.0])
    Transform = fcl.Transform(rotation, new_translation)
    #before transform
    body = fcl.CollisionObject(fcl.Box(length, width, 0.0), Transform) #x,y,z length center at the origin
    return body


def compute_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def Set_EndEffector(length):  #Adjust the position and angle of the end Effector
    while True:
        try:
            p1 = float(input("position_x:"))
            break
        except ValueError:
            print("That was no valid number.  Try again...")
    while True:
        try:
            p2 = float(input("position_y:"))
            break
        except ValueError:
            print("That was no valid number.  Try again...")
    while True:
        try:
            angle = float(input("angle:"))
            break
        except ValueError:
            print("That was no valid number.  Try again...")
    p2 = p2 - length / 2 * np.sin(angle/180*np.pi)
    p1 = p1 - length / 2 * np.cos(angle/180*np.pi)
    return p1,p2,angle

def Set_Center(n):
    while True:
        try:
            p1 = float(input("position_x:"))
            break
        except ValueError:
            print("That was no valid number.  Try again...")
    while True:
        try:
            p2 = float(input("position_y:"))
            break
        except ValueError:
            print("That was no valid number.  Try again...")
    X = x.to_numpy()[:n]
    center = np.mean(X,axis=0)
    tmp_x = p1 - center[0]
    tmp_y = p2 - center[1]
    move_obstacle(n, tmp_x, tmp_y)

def Set_Stiffness():
    while True:
        try:
            tmpstiff = float(input("stiffness:"))
            if abs(tmpstiff) <= 1 and tmpstiff > 0:
                break
            else:
                print("Stiffness should be smaller than 1 and strictly bigger than 0")
        except ValueError:
            print("That was no valid number.  Try again...")
    stiffness[None] = tmpstiff

def Set_Module(module_dict):
    print(module_dict)
    while True:
        try:
            index = int(input("Module index:"))
            break
        except ValueError:
            print("That was no valid number.  Try again...")
    return index

def Set_lidar(tran_x, tran_y, angle, length,stick_corners,n, connection_matrix, tri_mesh, FixedPointsLists, scale_ratio, scale_offset, Circle):
    while True:
        try:
            Lidar_number = int(input("Lidar number:"))
            break
        except ValueError:
            print("That was no valid number.  Try again...")
    Tip_x = tran_x + length / 2 * np.cos(angle/180*np.pi)
    Tip_y = tran_y + length / 2 * np.sin(angle/180*np.pi)
    Lidar_angle = angle + 90
    angle_lists = np.linspace(Lidar_angle, Lidar_angle - 180, num=Lidar_number)
    distance_lists = np.linspace(0, LidarMaxDistance[None], num=int(LidarMaxDistance[None]/0.01)) #steps accuracy: 0.01
    X=x.to_numpy()[:n]
    verts = np.c_[X,np.zeros(n)]#fcl -> 3_D field
    contact_lists = {}
    for Li_distance in distance_lists:
        stick_corners[0][0] = stick_corners[0][0] * scale_ratio + scale_offset
        stick_corners[1][0] = stick_corners[1][0] * scale_ratio + scale_offset
        stick_corners[2][0] = stick_corners[2][0] * scale_ratio + scale_offset
        stick_corners[3][0] = stick_corners[3][0] * scale_ratio + scale_offset
        paint()
        gui.set_image(pixels)
        gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
        gui.line(begin=(0.0, 1 - bottom_y - 0.001), end=(1.0, 1 - bottom_y - 0.001), color=0x0, radius=1)
        gui.line(begin=stick_corners[0],end=stick_corners[1],color=0x0, radius=1)
        gui.line(begin=stick_corners[1],end=stick_corners[2],color=0x0, radius=1)
        gui.line(begin=stick_corners[2],end=stick_corners[3],color=0x0, radius=1)
        gui.line(begin=stick_corners[3],end=stick_corners[0],color=0x0, radius=1)
        for i in range(n):
            if i in FixedPointsLists:
                tmp = X[i:i+1].copy()
                tmp[0][0] = tmp[0][0] * scale_ratio + scale_offset
                gui.circles(tmp, color=0xffaa77, radius=6)
            else:
                tmp = X[i:i+1].copy()
                tmp[0][0] = tmp[0][0] * scale_ratio + scale_offset
                gui.circles(tmp, color=0xffaa77, radius=3)
            for j in connection_matrix[i]:
                tmp_i = X[i].copy()
                tmp_i[0] = tmp_i[0] * scale_ratio + scale_offset
                tmp_j = X[j].copy()
                tmp_j[0] = tmp_j[0] * scale_ratio + scale_offset
                gui.line(begin=tmp_i, end=tmp_j, radius=2, color=0x445566)
        for Li_angle in angle_lists:
            tmp = -1
            if contact_lists.get(Li_angle, -1) != -1:  # check if each Lidar point collide with the environment
                tmp = Li_distance
                Li_distance = contact_lists.get(Li_angle)
            pos_x = Tip_x - Li_distance * (-np.cos(Li_angle/180*np.pi))
            pos_y = Tip_y + Li_distance * (np.sin(Li_angle/180*np.pi))
            lidar = lidar_configuration(pos_x, pos_y, radius = 0.001) # model a Lidar point as a circle with a set radius (return a fcl object)
            minimum_distance = checkLidarCollistion(lidar, verts, tri_mesh)
            if minimum_distance == -1: #lidar contact the obstacle
                contact_lists[Li_angle] = Li_distance
            if tmp != -1: #reset the Li_distance for other Lidar_points
                Li_distance = tmp
            gui.circles(np.array([[pos_x, pos_y]]), color=0xffaa77, radius=5) #draw circles
        if Circle is not None:
            gui.circles(Circle, color=0xffaa77, radius=10)
        gui.show()
        time.sleep(0.5)
    print(contact_lists)
    return contact_lists

# verts -> vertices position; tris -> vertices mesh structure
def checkLidarCollistion(lidar, verts, tris):
    mesh = fcl.BVHModel()
    mesh.beginModel(len(verts), len(tris))
    mesh.addSubModel(verts, tris)
    mesh.endModel()
    mesh_obj = fcl.CollisionObject(mesh)
    objs = [mesh_obj, lidar]
    manager = fcl.DynamicAABBTreeCollisionManager()
    manager.registerObjects(objs)
    manager.setup()
    crequest = fcl.CollisionRequest(enable_contact=True)
    drequest = fcl.DistanceRequest(enable_nearest_points=True)
    cdata = fcl.CollisionData(crequest, fcl.CollisionResult())
    ddata = fcl.DistanceData(drequest, fcl.DistanceResult())
    manager.collide(cdata, fcl.defaultCollisionCallback)
    manager.distance(ddata, fcl.defaultDistanceCallback)
    minimum_distance = ddata.result.min_distance
    return minimum_distance

def Set_lidarMaxDistance():
    while True:
        try:
            tmp = float(input("LidarMaxDistance:"))
            break
        except ValueError:
            print("That was no valid number.  Try again...")
    LidarMaxDistance[None] = tmp

def Set_Motion():
    while True:
        try:
            motion_mode = int(input("Enter Motion Mode:"))
            break
        except ValueError:
            print("That was no valid number.  Try again...")
    if motion_mode == 0: #Translation(along the stick)
        while True:
            try:
                Motion_distance = float(input("Enter the motion distance:"))
                break
            except ValueError:
                print("That was no valid number.  Try again...")
        return motion_mode, Motion_distance, True
    elif motion_mode == 1: #Rotation
        while True:
            try:
                Motion_angle = float(input("Enter the motion angle:"))
                break
            except ValueError:
                print("That was no valid number.  Try again...")
        return motion_mode, Motion_angle, True
    elif motion_mode == 2: #Translation(perpendicular to the stick)
        while True:
            try:
                Motion_distance = float(input("Enter the motion distance:"))
                break
            except ValueError:
                print("That was no valid number.  Try again...")
        return motion_mode, Motion_distance, True
    else:
        print("Invalid Input")
        return -1, -1, False

def Set_FixedPoints(stick_corners,n, connection_matrix, FixedPointsLists, Circle):
    FixedPoints = []
    ContinueSelect = True
    X = x.to_numpy()[:n]
    while ContinueSelect:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.LMB:
                SelectedPoint = Find_ClosePoints(e.pos[0], e.pos[1], X, n)
                FixedPoints.append(SelectedPoint)
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                ContinueSelect = False
        paint()
        gui.set_image(pixels)
        gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
        gui.line(begin=(0.0, 1 - bottom_y - 0.001), end=(1.0, 1 - bottom_y - 0.001), color=0x0, radius=1)
        gui.line(begin=stick_corners[0],end=stick_corners[1],color=0x0, radius=1)
        gui.line(begin=stick_corners[1],end=stick_corners[2],color=0x0, radius=1)
        gui.line(begin=stick_corners[2],end=stick_corners[3],color=0x0, radius=1)
        gui.line(begin=stick_corners[3],end=stick_corners[0],color=0x0, radius=1)
        for i in range(n):
            if i in FixedPointsLists or i in FixedPoints:
                gui.circles(X[i:i + 1], color=0xffaa77, radius=6)
            else:
                gui.circles(X[i:i+1], color=0xffaa77, radius=3)
            for j in connection_matrix[i]:
                gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
        if Circle is not None:
            gui.circles(Circle, color=0xffaa77, radius=10)
        gui.show()
    return FixedPoints

def Set_circle(stick_corners,n, connection_matrix, FixedPointsLists):
    ContinueSelect = True
    circle = np.ones([1,2])
    X = x.to_numpy()[:n]
    while ContinueSelect:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.LMB:
                circle[0][0] = e.pos[0]
                circle[0][1] = e.pos[1]
                ContinueSelect = False
        paint()
        gui.set_image(pixels)
        gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
        gui.line(begin=(0.0, 1 - bottom_y - 0.001), end=(1.0, 1 - bottom_y - 0.001), color=0x0, radius=1)
        gui.line(begin=stick_corners[0],end=stick_corners[1],color=0x0, radius=1)
        gui.line(begin=stick_corners[1],end=stick_corners[2],color=0x0, radius=1)
        gui.line(begin=stick_corners[2],end=stick_corners[3],color=0x0, radius=1)
        gui.line(begin=stick_corners[3],end=stick_corners[0],color=0x0, radius=1)
        for i in range(n):
            if i in FixedPointsLists:
                gui.circles(X[i:i + 1], color=0xffaa77, radius=6)
            else:
                gui.circles(X[i:i+1], color=0xffaa77, radius=3)
            for j in connection_matrix[i]:
                gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
        gui.show()
    return circle

def Remove_FixedPoints(stick_corners,n, connection_matrix, FixedPointsLists, Circle):
    RemovedPoints = []
    ContinueSelect = True
    X = x.to_numpy()[:n]
    while ContinueSelect:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.LMB:
                SelectedPoint = Find_ClosePoints(e.pos[0], e.pos[1], X, n)
                RemovedPoints.append(SelectedPoint)
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                ContinueSelect = False
        paint()
        gui.set_image(pixels)
        gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
        gui.line(begin=(0.0, 1 - bottom_y - 0.001), end=(1.0, 1 - bottom_y - 0.001), color=0x0, radius=1)
        gui.line(begin=stick_corners[0],end=stick_corners[1],color=0x0, radius=1)
        gui.line(begin=stick_corners[1],end=stick_corners[2],color=0x0, radius=1)
        gui.line(begin=stick_corners[2],end=stick_corners[3],color=0x0, radius=1)
        gui.line(begin=stick_corners[3],end=stick_corners[0],color=0x0, radius=1)
        for i in range(n):
            if i in FixedPointsLists and i not in RemovedPoints:
                gui.circles(X[i:i + 1], color=0xffaa77, radius=6)
            else:
                gui.circles(X[i:i + 1], color=0xffaa77, radius=3)
            for j in connection_matrix[i]:
                gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
        if Circle is not None:
            gui.circles(Circle, color=0xffaa77, radius=10)
        gui.show()
    return RemovedPoints

def Find_ClosePoints(p1_x, p1_y, X, n):
    min_distance = 10000
    closedPoint = -1
    for i in range(n):
        distance = (p1_x - X[i][0]) ** 2 + (p1_y - X[i][1]) ** 2
        if distance <= min_distance:
            closedPoint = i
            min_distance = distance
    return closedPoint

def compute_Rotationdirection(rotate_direction, transform_matrix, length, width, nearest_point):
    top_left = np.array([-length / 2, width / 2, 0])
    top_right = np.array([length / 2, width / 2, 0])
    bottom_left = np.array([-length / 2, -width / 2, 0])
    bottom_right = np.array([length / 2, -width / 2, 0])
    top_left_homo = np.r_[top_left,np.array([1,])]
    top_right_homo = np.r_[top_right,np.array([1,])]
    bottom_left_homo = np.r_[bottom_left,np.array([1,])]
    bottom_right_homo = np.r_[bottom_right,np.array([1,])]
    top_left_trans = transform_matrix @ top_left_homo
    top_left_trans = top_left_trans[:2]
    top_left = compute_distance(top_left_trans, nearest_point)
    top_right_trans = transform_matrix @ top_right_homo
    top_right_trans = top_right_trans[:2]
    top_right = compute_distance(top_right_trans, nearest_point)
    bottom_left_trans = transform_matrix @ bottom_left_homo
    bottom_left_trans = bottom_left_trans[:2]
    bottom_left = compute_distance(bottom_left_trans, nearest_point)
    bottom_right_trans = transform_matrix @ bottom_right_homo
    bottom_right_trans = bottom_right_trans[:2]
    bottom_right = compute_distance(bottom_right_trans, nearest_point)
    if top_right < top_left:
        if rotate_direction == 'clock-wise':
            direction = bottom_right_trans - top_right_trans
        elif rotate_direction == 'counter-clock-wise':
            direction = top_right_trans- bottom_right_trans
    else:
        if rotate_direction == 'clock-wise':
            direction = top_left_trans - bottom_left_trans
        elif rotate_direction == 'counter-clock-wise':
            direction = bottom_left_trans- top_left_trans
    return direction

def compute_Translationdirection(angle, motion_value):
    if motion_value >= 0:
        direction = [np.cos(angle/180*np.pi), np.sin(angle/180*np.pi)]
        return direction
    else:
        direction = [-np.cos(angle/180*np.pi), -np.sin(angle/180*np.pi)]
        return direction

def compute_PerpendicularDirection(angle, motion_value):
    if motion_value >= 0:
        direction = [np.cos((angle+90)/180*np.pi), np.sin((angle+90)/180*np.pi)]
        return direction
    else:
        direction = [np.cos((angle-90)/180*np.pi), np.sin((angle-90)/180*np.pi)]
        return direction

### Function: Convert each tri relation into three spring constraints
def tria2constraint(constraints, X):
    new_constraints = []
    for i in constraints:
        for j in range(len(i)):
            pos_j = X[i[j]]
            for z in range(j + 1, len(i)):
                new_constraint = []
                pos_z = X[i[z]]
                pos_jz = pos_j - pos_z
                new_constraint.append(i[j])
                new_constraint.append(i[z])
                new_constraint.append(np.linalg.norm(pos_jz))
                new_constraints.append(new_constraint)
    return new_constraints

def tria2Energy(constraints, X):
    new_constraints = []
    for i in constraints:
        for j in range(len(i)):
            pos_j = X[i[j]]
            for z in range(j + 1, len(i)):
                new_constraint = []
                pos_z = X[i[z]]
                pos_jz = pos_j - pos_z
                new_constraint.append(i[j])
                new_constraint.append(i[z])
                new_constraint.append(pos_jz[0])
                new_constraint.append(pos_jz[1])
                new_constraints.append(new_constraint)
    return new_constraints


def ComputerEnergyConstant(NumberParticle, rest_X):
    NumberConstraints = len(rest_X)
    K = stiffness[None] * np.eye(NumberConstraints)
    M = np.zeros([NumberParticle, NumberConstraints])
    d = np.zeros([NumberConstraints, 2])
    for i in range(NumberConstraints):
        p1 = rest_X[i][0]
        p2 = rest_X[i][1]
        M[p1,i] = 1
        M[p2,i] = -1
        d[i][0] = rest_X[i][2]
        d[i][1] = rest_X[i][3]
    s = np.eye(NumberConstraints)
    N = M @ K @ s.T
    s = s @ K @ s.T
    constant = 0.5 * np.trace(d.T @ s @ d)
    M_ = M @ K @ M.T
    N = N @ d
    return M_, N, constant

def ComputerEnergy(tissue, M, N, constant, X, triangle_current, triangle_rest, number_bottom, number_upper, n):
    Momentum_Energy[None] = 0
    if tissue == 'Bottom':
        Compute_Momentum_Energy(0, number_bottom)
    elif tissue == 'Upper':
        Compute_Momentum_Energy(number_upper, n)
    else:
        print("Wrong")
    Area_energy = 0
    for i in range(triangle_current.shape[0]):
        Area_energy += dt ** 2 *0.5 * Area_parameter[None] * (triangle_current[i] - triangle_rest[i]) ** 2
    Spring_energy = dt ** 2 * (0.5 * np.trace(X.T @ M @ X) - np.trace(X.T @ N) + constant)
    return Area_energy + Spring_energy, Momentum_Energy[None], Momentum_Energy[None] + Area_energy + Spring_energy

def output_file(file_name, A, S, All):
    np.save(file_name + '_Potential.npy',A)
    np.save(file_name + '_Momentum.npy',S)
    np.save(file_name + '_Implicit.npy',All)

stiffness[None] = 0.15 #adjustable
damping[None] = 8 #8 is the most suitable
LidarMaxDistance[None] = 0.1 #default
Area_parameter[None] = 1
test[None] = 0

def main(Bottom_dir, Upper_dir, file_dir, interval, video_save):
    offset = 1
    #Read all mesh points from node/ele files  Bottom part
    points = []
    with open(Bottom_dir + 'Bottom.1.node', 'r') as f:
        data = f.readlines()
    for line in data[1: len(data) - 1]:
        odom = line.split()
        points.append([float(odom[1]), float(odom[2])])
    # Read all constraints(triangle)
    constraints = []
    with open(Bottom_dir + 'Bottom.1.ele', 'r') as f:
        data = f.readlines()
    for line in data[1: len(data) - 1]:
        odom = line.split()
        constraints.append([int(odom[1]) - offset, int(odom[2])- offset, int(odom[3]) - offset])

    num_particles[None] = 0
    num_trian[None] = 0
    FixedPointsLists = []
    # Set the initial fixed points based on the y-value
    for i in points:
        if i[1] >= 0.023 and i[1] <= 0.026:
            FixedPointsLists.append(points.index(i))
            new_particle(i[0], i[1], -1)
        elif i[1] >= 0.97 and i[1] <= 0.99:
            FixedPointsLists.append(points.index(i))
            new_particle(i[0], i[1], -1)
        else:
            new_particle(i[0], i[1], 1)

    n = num_particles[None]
    X = x.to_numpy()[:n]
    # Input: raw triangles
    # Output: Constraint & rest length
    Bottom_constraints_energy = tria2Energy(constraints, X)  # normal constraint (should be used for changing stiffness)
    Number_bottom_points = len(points)
    Number_bottom_triangle = len(constraints)
    Bottom_M, Bottom_N, Bottom_constannt = ComputerEnergyConstant(Number_bottom_points, Bottom_constraints_energy)  #Energy constants
    Bottom_points = points.copy()

    #Read all mesh points from node/ele files  Upper part
    with open(Upper_dir + 'Upper.1.node', 'r') as f:
            data = f.readlines()
    for line in data[1: len(data) - 1]:
        odom = line.split()
        points.append([float(odom[1]), float(odom[2])])
    # Read all constraints(triangle)
    with open(Upper_dir + 'Upper.1.ele', 'r') as f:
            data = f.readlines()
    Upper_constraints = []
    for line in data[1: len(data) - 1]:
        odom = line.split()
        Upper_constraints.append([int(odom[1]) - offset, int(odom[2]) - offset, int(odom[3]) - offset])
        constraints.append([int(odom[1]) + Number_bottom_points - offset, int(odom[2])  + Number_bottom_points - offset, int(odom[3])  + Number_bottom_points - offset])
    Number_upper_points = len(points) - Number_bottom_points
    Number_upper_triangle = len(constraints) - Number_bottom_triangle
    for i in points:
        if i not in Bottom_points:
            if i[1] >= 0.023 and i[1] <= 0.027:
                FixedPointsLists.append(points.index(i))
                new_particle(i[0], i[1], -1)
            elif i[1] >= 0.97 and i[1] <= 0.99:
                FixedPointsLists.append(points.index(i))
                new_particle(i[0], i[1], -1)
            else:
                new_particle(i[0], i[1], 1)
    n = num_particles[None]
    X = x.to_numpy()[:n]
    X_upper = x.to_numpy()[Number_bottom_points:n]
    Upper_constraints_energy = tria2Energy(Upper_constraints, X_upper)  # normal constraint
    Upper_M, Upper_N, Upper_constannt = ComputerEnergyConstant(Number_upper_points, Upper_constraints_energy)  #Energy constants
    # Input: raw triangles
    # Output: Constraint & rest length, that is needed to be used in simulation
    new_constraints = tria2constraint(constraints, X)

    for triangle_vertices in constraints:
        new_trian(triangle_vertices[0], triangle_vertices[1], triangle_vertices[2])
    number_tri = num_trian[None]
    volume_start = trian_volumn.to_numpy()[:number_tri]
    volume_sum = 0
    for i in range(number_tri):
        volume_sum += volume_start[i, 3] #rest_area
    print("Initial area is: ", volume_sum)

    tri_matrix = np.zeros((n,n))
    for i in new_constraints:
        new_costraint(int(i[0]), int(i[1]), float(i[2]))
        tri_matrix[int(i[0]),int(i[1])]=1

    tri_mesh = [] #triangle_mesh
    for i in range(n):
        for j in range(n):
            if tri_matrix[i,j] == 1:
                for k in range(j+1,n):
                    if tri_matrix[i,k] == 1 and tri_matrix[j,k] == 1:
                        if CheckRepeatMesh(i,j,k,tri_mesh): #True -> add False -> Remove
                            tri_mesh.append([i,j,k])

    connection_matrix = [] # mainly used for the visualization purpose
    for i in range(n):
        tmp = []
        for j in range(i + 1, n):
            if rest_length[i, j] != 0:
                tmp.append(j)
        connection_matrix.append(tmp) #spring connection
    tri_mesh = np.array(tri_mesh)
    # actuation_type_tmp_history = np.ones((max_num_particles,),dtype=float)
    # System Setup
    index = 0
    omega = 0.4 #unit:degree
    speed = 0.001 #normalized between [0,1]
    initial_angle = 0
    tolerance = 0.02 #stick and obstacle  0.025 for task2 0.03 for task1
    clearance[None] = 0.01
    scale = 1  #response intensity
    length = 0.8 #fixed or adjustable
    length_tip = 0.05
    stick_offset = (length - length_tip) / 2
    body_offset = length_tip / 2
    width = 0.005 #fixed
    trans_x = 0.15 #initial postion
    trans_y = 0.5 #initial position
    Motion_switch_on = False
    Motion_Index = -1
    Motion_value = -1
    Store = False
    Pause = False
    Collision_Enter = False
    Draw_circle = False
    Circle = None

    top_left = np.array([-length / 2, width / 2, 0])
    top_right = np.array([length / 2, width / 2, 0])
    bottom_left = np.array([-length / 2, -width / 2, 0])
    bottom_right = np.array([length / 2, -width / 2, 0])
    scale_ratio = 1  # This parameter is used to make sure the object deformation is always shown with the window
    # It should be noted that if scale_ratio is not set to be 1, the end_effecter position is not set as expected!
    scale_offset = (1 - scale_ratio) / 2
    rotate_direction = 'counter-clock-wise' #or 'clock-wise'
    Module = {'EndEffector':0, 'Extension':1 ,'Obstacles':2, 'stiffness':3, 'LidarSwitch':4, 'LidarMaxDistance':5,'FixedPoints':6, 'Motion':7, 'Length': 8, 'FixedPointsRemove': 9, 'Fixed_circle': 10, 'Store': 11} #Mode
    filename = ''
    Image_store = False
    Lists_bottom_potential = []
    Lists_bottom_momentum = []
    Lists_bottom_implicit = []
    Lists_upper_potential = []
    Lists_upper_momentum = []
    Lists_upper_implicit= []
    # System Setup Finish

    # Simulation starts
    while True:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == gui.SPACE and not Motion_switch_on:
                #paused[None] = not paused[None]
                Module_index = Set_Module(Module)
                # Set the end effector position (tip position)
                if(Module_index == Module['EndEffector']):
                    print("Set End Effector")
                    trans_x, trans_y, initial_angle = Set_EndEffector(length)
                # Extend Set the end effector position
                # disable
                # if(Module_index == Module['Extension']):
                #     print("Set Extension")
                # disable
                # Set the center point of the obstacle
                # disable
                # if(Module_index == Module['Obstacles']):
                #     print("Set center point")
                #     Set_Center(n)
                # disable
                # Set the stiffness
                if(Module_index == Module['stiffness']):
                    print("Set stiffness")
                    Set_Stiffness()
                    Bottom_M, Bottom_N, Bottom_constannt = ComputerEnergyConstant(Number_bottom_points, Bottom_constraints_energy)  #Energy constants
                    Upper_M, Upper_N, Upper_constannt = ComputerEnergyConstant(Number_upper_points, Upper_constraints_energy)  #Energy constants
                # Launch Lidar beam
                if(Module_index == Module['LidarSwitch']):
                    print("Lidar Switch on")
                    Set_lidar(trans_x, trans_y, initial_angle, length, stick_corners, n, connection_matrix, tri_mesh, FixedPointsLists, scale_ratio, scale_offset, Circle)
                # Set Lidar Max distance
                if(Module_index == Module['LidarMaxDistance']):
                    print("Set Lidar max distance")
                    Set_lidarMaxDistance()
                # Set the fixed particles (add to the initial list)
                if(Module_index == Module['FixedPoints']):
                    print("Select Fixed points")
                    FixedPointsLists.extend(Set_FixedPoints(stick_corners, n, connection_matrix, FixedPointsLists, Circle))
                # Set the motion mode
                if(Module_index == Module['Motion']):
                    print("Motion command:")
                    print("Translation(along orientation):0, Rotation:1, Translation(perpendicular to orientation):2")
                    Motion_Index, Motion_value, Motion_switch_on = Set_Motion() #motion_index: 0 -> Translation 1-> Rotation
                # Set the length of the cylinder
                if(Module_index == Module['Length']):
                    print("Set Length")
                    length = float(input("Length:"))
                    top_left = np.array([-length / 2, width / 2, 0])
                    top_right = np.array([length / 2, width / 2, 0])
                    bottom_left = np.array([-length / 2, -width / 2, 0])
                    bottom_right = np.array([length / 2, -width / 2, 0])
                # Remove selected fixed particles
                if(Module_index == Module['FixedPointsRemove']):
                    print("Remove fixed points")
                    FixedPointsLists = list(set(FixedPointsLists) - set(Remove_FixedPoints(stick_corners, n, connection_matrix, FixedPointsLists, Circle)))
                # Set the circle
                if(Module_index == Module['Fixed_circle']):
                    print("Select circle point")
                    Circle = Set_circle(stick_corners, n, connection_matrix, FixedPointsLists)
                if(Module_index == Module['Store']):
                    Store = True
        collision = -10
        for step in range(1):
            if not Pause:  # Continue simulation
                forward(n, number_tri)
            else:  #compute energy (cylinder get inside the tissue)
                X=x.to_numpy()[:n]
                while True:
                    paint()
                    gui.set_image(pixels)
                    filename = f'final.png'
                    filename = file_dir + filename
                    gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
                    gui.line(begin=(0.0, 1 - bottom_y - 0.001), end=(1.0, 1 - bottom_y - 0.001), color=0x0, radius=1)
                    gui.line(begin=stick_corners[0],end=stick_corners[1],color=0x0, radius=1)
                    gui.line(begin=stick_corners[1],end=stick_corners[2],color=0x0, radius=1)
                    gui.line(begin=stick_corners[2],end=stick_corners[3],color=0x0, radius=1)
                    gui.line(begin=stick_corners[3],end=stick_corners[0],color=0x0, radius=1)
                    for i in range(n):
                        if i in FixedPointsLists:
                            tmp = X[i:i+1].copy()
                            tmp[0][0] = tmp[0][0] * scale_ratio + scale_offset
                            gui.circles(tmp, color=0xffaa77, radius=6)
                        else:
                            tmp = X[i:i+1].copy()
                            tmp[0][0] = tmp[0][0] * scale_ratio + scale_offset
                            gui.circles(tmp, color=0xffaa77, radius=3)
                        for j in connection_matrix[i]:
                            tmp_i = X[i].copy()
                            tmp_i[0] = tmp_i[0] * scale_ratio + scale_offset
                            tmp_j = X[j].copy()
                            tmp_j[0] = tmp_j[0] * scale_ratio + scale_offset
                            gui.line(begin=tmp_i, end=tmp_j, radius=2, color=0x445566)
                    if Draw_circle:
                        gui.circles(Circle, color=0xffaa77, radius=10)
                    gui.show(filename)
                    output_file(file_dir + 'Bottom', Lists_bottom_potential, Lists_bottom_momentum, Lists_bottom_implicit)
                    output_file(file_dir + 'Upper', Lists_upper_potential, Lists_upper_momentum, Lists_upper_implicit)
                    sys.exit(0)
            X = x.to_numpy()[:n]
            X_bottom = X[:Number_bottom_points]
            X_upper = X[Number_bottom_points:n]
            tri_index = 0
            for triangle_vertices in constraints:
                Compute_area(triangle_vertices[0], triangle_vertices[1], triangle_vertices[2], tri_index)
                tri_index += 1
            Triangle_bottom = triangle_area.to_numpy()[:Number_bottom_triangle] #current
            Rest_bottom = trian_volumn.to_numpy()[:Number_bottom_triangle][:,3] #rest
            Triangle_upper = triangle_area.to_numpy()[Number_bottom_triangle:number_tri] #current
            Rest_upper = trian_volumn.to_numpy()[Number_bottom_triangle:number_tri][:,3] #rest
            PotentialEnergy_bottom, MomentumEnergy_bottom, ImplitEnergy_bottom = ComputerEnergy('Bottom', Bottom_M, Bottom_N, Bottom_constannt, X_bottom, Triangle_bottom, Rest_bottom, Number_bottom_points, Number_upper_points, n)  #Spring + triangle + gravity
            PotentialEnergy_upper, MomentumEnergy_upper, ImplitEnergy_upper = ComputerEnergy('Upper', Upper_M, Upper_N, Upper_constannt, X_upper, Triangle_upper, Rest_upper, Number_bottom_points, Number_upper_points, n) #Spring + triangle + gravity

            if Motion_switch_on: #Store the energy data
                Lists_bottom_potential.append(PotentialEnergy_bottom)
                Lists_bottom_momentum.append(MomentumEnergy_bottom)
                Lists_bottom_implicit.append(ImplitEnergy_bottom)
                Lists_upper_potential.append(PotentialEnergy_upper)
                Lists_upper_momentum.append(MomentumEnergy_upper)
                Lists_upper_implicit.append(ImplitEnergy_upper)

            verts = np.c_[X,np.zeros(n)] #fcl used for 3_D field -> add the zero components to -1 axis
            if Motion_switch_on:
                if Motion_Index == 1:   #Rotation
                    if Motion_value < 0:
                        if omega*index <= abs(Motion_value):
                            index += 1
                            initial_angle -= omega
                            rotate_direction = 'clock-wise'
                        else:
                            Motion_switch_on = False
                            index = 0
                    else:
                        if omega*index <= abs(Motion_value):
                            index += 1
                            initial_angle += omega
                            rotate_direction = 'counter-clock-wise'
                        else:
                            Motion_switch_on = False
                            index = 0
                elif Motion_Index == 0: #Translation along the orientation(angle remains the same)
                    if Motion_value < 0:
                        if speed*index <= abs(Motion_value):
                            index += 1
                            trans_x -= speed * np.cos(angle/180*np.pi)
                            trans_y -= speed * np.sin(angle/180*np.pi)
                            if video_save:
                                if index % interval == 0:
                                    paint()
                                    gui.set_image(pixels)
                                    filename = f'frame_{index:05d}.png'   # create filename with suffix png
                                    filename = file_dir + filename
                                    Image_store = True
                                else:
                                    Image_store = False
                        else:
                            Motion_switch_on = False
                            index = 0
                    else:
                        if speed*index <= abs(Motion_value):
                            index += 1
                            trans_x += speed * np.cos(angle/180*np.pi)
                            trans_y += speed * np.sin(angle/180*np.pi)
                            if video_save:
                                if index % interval == 0:
                                    filename = f'frame_{index:05d}.png'   # create filename with suffix png
                                    filename = file_dir + filename
                                    Image_store = True
                                else:
                                    Image_store = False
                        else:
                            Motion_switch_on = False
                            index = 0
                elif Motion_Index == 2: #Translation perpendicular to the orientation(angle remains the same)
                    if Motion_value < 0:
                        if speed*index <= abs(Motion_value):
                            index += 1
                            trans_x += speed * np.cos((angle-90)/180*np.pi)
                            trans_y += speed * np.sin((angle-90)/180*np.pi)
                            if video_save:
                                if index % interval == 0:
                                    filename = f'frame_{index:05d}.png'   # create filename with suffix png
                                    filename = file_dir + filename
                                    Image_store = True
                                else:
                                    Image_store = False
                        else:
                            Motion_switch_on = False
                            index = 0
                    else:
                        if speed*index <= abs(Motion_value):
                            index += 1
                            trans_x += speed * np.cos((angle+90)/180*np.pi)
                            trans_y += speed * np.sin((angle+90)/180*np.pi)
                            # trans_x += speed * np.cos(angle+90/180*np.pi)
                            # trans_y += speed * np.sin(angle+90/180*np.pi) #original version, maybe something wrong
                            if video_save:
                                if index % interval == 0:
                                    filename = f'frame_{index:05d}.png'   # create filename with suffix png
                                    filename = file_dir + filename
                                    Image_store = True
                                else:
                                    Image_store = False
                        else:
                            Motion_switch_on = False
                            index = 0
            angle = initial_angle
            new_trans_x = trans_x + stick_offset * np.cos(angle/180*np.pi)  #tip of stick
            new_trans_y = trans_y + stick_offset * np.sin(angle/180*np.pi)  #tip of stick
            # newnew_trans_x = trans_x - body_offset * np.cos(angle/180*np.pi)  #The left part of the stick
            # newnew_trans_y = trans_y - body_offset * np.sin(angle/180*np.pi)
            if Motion_Index == 0: # Assume the collision should happen along the tip point
                transform_matrix, stick_corners, stick = stick_configuration(angle, trans_x, trans_y, new_trans_x, new_trans_y, length_tip, width, top_left, top_right, bottom_left, bottom_right)
            else: # Assume the collision should happen on the body side
                transform_matrix, stick_corners, stick = stick_configuration(angle, trans_x, trans_y, trans_x, trans_y, length, width, top_left, top_right, bottom_left, bottom_right)
            # Which side is close to the tissue
            # This can be computed efficiently based on the distance between the particle and line (not hard)
            # Line_function_A[None] = stick_corners[1][1] - stick_corners[0][1]
            # Line_function_B[None] = stick_corners[0][0] - stick_corners[1][0]
            # Line_function_C[None] = stick_corners[1][0]*stick_corners[0][1] - stick_corners[0][0]*stick_corners[1][1]
            Line_function_A[None] = stick_corners[2][1] - stick_corners[3][1]
            Line_function_B[None] = stick_corners[3][0] - stick_corners[2][0]
            Line_function_C[None] = stick_corners[2][0]*stick_corners[3][1] - stick_corners[3][0]*stick_corners[2][1]
            Line_function_D[None] = np.sqrt(Line_function_A[None]**2 + Line_function_B[None]**2)
            # if stick_corners[0][0] >= stick_corners[1][0]:
            if stick_corners[3][0] >= stick_corners[2][0]:
                PointP1_x[None] = stick_corners[3][0]
                PointP2_x[None] = stick_corners[2][0]
            else:
                PointP1_x[None] = stick_corners[2][0]
                PointP2_x[None] = stick_corners[3][0]
            # body = body_configuration(angle, newnew_trans_x, newnew_trans_y, length_tip, width)
            # nearest_point_body, collision_body, delta_body = BodyCheckCollison(rotate_direction, verts, tri_mesh, body, angle, tolerance) #argv1 & argv2 -> mesh argv3 -> stick
            # Very useful add: check whether they are on the same side (direction and tissue's collision point)
            # deserve to have a try if time permit
            # or switch the position of the rectangle.(I think it is better one(two tips, two sides))

            # nearest_point record the the contact particle position on the mesh (id -> particle index)
            nearest_point, collision, delta, nearest_point_id = CheckCollison(rotate_direction, verts, tri_mesh, stick, angle, trans_x, trans_y, tolerance) #argv1 & argv2 -> mesh argv3 -> stick 
            if collision == -1 and Collision_Enter:
                Pause = True
            #Rotation collision and translation collision should use different strategies
        stick_corners[0][0] = stick_corners[0][0] * scale_ratio + scale_offset
        stick_corners[1][0] = stick_corners[1][0] * scale_ratio + scale_offset
        stick_corners[2][0] = stick_corners[2][0] * scale_ratio + scale_offset
        stick_corners[3][0] = stick_corners[3][0] * scale_ratio + scale_offset
        paint()
        gui.set_image(pixels)
        gui.line(begin=stick_corners[0],end=stick_corners[1],color=0x0, radius=1)
        gui.line(begin=stick_corners[1],end=stick_corners[2],color=0x0, radius=1)
        gui.line(begin=stick_corners[2],end=stick_corners[3],color=0x0, radius=1)
        gui.line(begin=stick_corners[3],end=stick_corners[0],color=0x0, radius=1)
        actuation_type_tmp = np.ones((max_num_particles,),dtype=float)
        Delta_x_se = np.zeros((max_num_particles,),dtype=float)
        Delta_y_se = np.zeros((max_num_particles,),dtype=float)
        for i in range(n):
            if collision == 1:  #interaction of the cylinder and mass
                Collision_Enter = True
                distance_to_point = compute_distance(X[i:i+1][0], nearest_point)
                # if X[i:i+1][0][0] == nearest_point[0] and X[i:i+1][0][1] == nearest_point[1]: #control point
                region_radius = 0.25
                distance_scale = (region_radius - distance_to_point) / region_radius
                Check = False
                # Based on the region_radius, the particle on the other tissue may also be affected
                if nearest_point_id < Number_bottom_points: #bottom tissue
                    if i < Number_bottom_points:
                        Check = True
                else: #upper tissue
                    if i >= Number_bottom_points:
                        Check = True
                if distance_scale >= 0 and Check: # The particle i is within the radius -> compute the position delta
                    actuation_type_tmp[i] = 2
                    # actuation_type_tmp_history[i] = 2
                    # The position delta can be considered as: 1). compute the direction, 2). compute the distance
                    #direction is computed in three different ways:rotation or translation
                    if Motion_Index == 1: #rotation
                        direction = compute_Rotationdirection(rotate_direction, transform_matrix, length, width, nearest_point)
                    elif Motion_Index == 0: #translation(along the orientation)
                        direction = compute_Translationdirection(angle, Motion_value)
                    elif Motion_Index == 2: #translation(perpendicular to the orientation)
                        direction = compute_PerpendicularDirection(angle, Motion_value)
                    # Compute the position change for each particle caused by the cylinder motion
                    delta_x = scale * distance_scale * delta * direction[0] / np.sqrt(direction[0] ** 2 + direction[1] ** 2)
                    delta_y = scale * distance_scale * delta * direction[1] / np.sqrt(direction[0] ** 2 + direction[1] ** 2)
                    Delta_x_se[i] = delta_x
                    Delta_y_se[i] = delta_y
            #one extra iteration
            # elif collision == 0 and Collision_Enter:
            #     actuation_type_tmp[i] = actuation_type_tmp_history[i]
            #     Collision_Enter = False
            if i in FixedPointsLists:
                actuation_type_tmp[i] = -1
        actuation_type.from_numpy(actuation_type_tmp)
        Delta_x_sequence.from_numpy(Delta_x_se)  #
        Delta_y_sequence.from_numpy(Delta_y_se)
        gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
        gui.line(begin=(0.0, 1-bottom_y-0.001), end=(1.0, 1-bottom_y-0.001), color=0x0, radius=1)
        for i in range(n):
            if i in FixedPointsLists:
                tmp = X[i:i+1].copy()
                tmp[0][0] = tmp[0][0] * scale_ratio + scale_offset
                gui.circles(tmp, color=0xffaa77, radius=6)
            else:
                tmp = X[i:i+1].copy()
                tmp[0][0] = tmp[0][0] * scale_ratio + scale_offset
                gui.circles(tmp, color=0xffaa77, radius=3)
            for j in connection_matrix[i]:
                tmp_i = X[i].copy()
                tmp_i[0] = tmp_i[0] * scale_ratio + scale_offset
                tmp_j = X[j].copy()
                tmp_j[0] = tmp_j[0] * scale_ratio + scale_offset
                gui.line(begin=tmp_i, end=tmp_j, radius=2, color=0x445566)
        if Circle is not None:
            gui.circles(Circle, color=0xffaa77, radius=10)
        if Motion_switch_on and Image_store:
            gui.show(filename)
        if Store:
            filename = f'final.png'  # create filename with suffix png
            filename = file_dir + filename
            gui.show(filename)
            output_file(file_dir + 'Bottom', Lists_bottom_potential, Lists_bottom_momentum, Lists_bottom_implicit)
            output_file(file_dir + 'Upper', Lists_upper_potential, Lists_upper_momentum, Lists_upper_implicit)
            sys.exit(0)
        else:
            gui.show()  # export and show in GUI

if __name__ == '__main__':
    Bottom_dir = sys.argv[1]
    Upper_dir = sys.argv[2]
    file_dir = sys.argv[3]
    if file_dir[-1] != '/':
        file_dir += '/'
    interval = int(sys.argv[4])
    if sys.argv[5] == "True":
        video_save = True
    elif sys.argv[5] == "False":
        video_save = False
    else:
        print("Check parameters!")
        sys.exit(0)
    main(Bottom_dir, Upper_dir, file_dir, interval, video_save)


