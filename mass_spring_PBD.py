import taichi as ti
import math as math
import fcl
import numpy as np

ti.init(debug=False,arch=ti.cpu)
real = ti.f32 #data type f32 -> float in C

max_num_particles = 1000
lambda_epsilon = 0.0 #user specified relaxation parameter(important) -> adjustable
dt = 1e-2#simulation time step(important) -> adjustable
dt_inv = 1 / dt
dx = 0.02
dim = 2
pbd_num_iters = 30#Iteration number(important) -> adjustable

scalar = lambda: ti.var(dt=real) #2D dense tensor
vec = lambda: ti.Vector(dim, dt=real) #2*1 vector(each element in a tensor)
mat = lambda: ti.Matrix(dim, dim, dt=real) #2*2 matrix(each element in a tensor)

num_particles = ti.var(ti.i32, shape=())
paused = ti.var(ti.i32, shape=())
damping = ti.var(ti.f32, shape=())

particle_mass = 1 #mi
particle_mass_inv = 1 / particle_mass # 1 / mi
particle_mass_invv = 1 / (particle_mass_inv + particle_mass_inv + lambda_epsilon)
maximum_constraints = 50

bottom_y = 0.02
bottom_x = 0.98
epsolon = 0.0001 #digit accurary(important) -> adjustable

x, v, old_x = vec(), vec(), vec()
actuation_type = scalar()
total_constraint = ti.var(ti.i32, shape=())
# rest_length[i, j] = 0 means i and j are not connected
rest_length = scalar()
position_delta_tmp = vec()
position_delta_sum = vec()
constraint_neighbors = ti.var(ti.i32)
constraint_num_neighbors = ti.var(ti.i32)

gravity = [0, 0] #direction
H_force = [0, 0] #another gr
previous_minimum_distance = -10000

@ti.layout  #Environment layout(placed in ti.layout) initialization of the dimensiond of each tensor variables(global)
def place():
    ti.root.dense(ti.ij, (max_num_particles, max_num_particles)).place(rest_length)
    ti.root.dense(ti.i, max_num_particles).place(x, v, old_x, actuation_type, position_delta_tmp, position_delta_sum) #initialzation to zero
    nb_node = ti.root.dense(ti.i, max_num_particles)
    nb_node.place(constraint_num_neighbors)
    nb_node.dense(ti.j, maximum_constraints).place(constraint_neighbors)

@ti.kernel
def old_posi(n: ti.i32):
    for i in range(n):
        old_x[i] = x[i]

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
def substep(n: ti.i32): # Compute force and new velocity
    for i in range(n):
        v[i] *= ti.exp(-dt * damping[None]) # damping
        total_force = ti.Vector(gravity) * particle_mass #gravity -> accelaration
        if actuation_type[i] == 1:
            total_force = ti.Vector(H_force) * particle_mass
        #if actuation_type[i] == 2: #control points by the stick

        v[i] += dt * total_force / particle_mass

@ti.kernel
def collision_check(n: ti.i32):# Collide with ground
    for i in range(n):
        if x[i].y < bottom_y:
            x[i].y = bottom_y
            v[i].y = 0
        if x[i].y > 1 - bottom_y:
            x[i].y = 1 - bottom_y
            v[i].y = 0
        if x[i].x > bottom_x:
            x[i].x = bottom_x
            v[i].x = 0
        if x[i].x < 1 - bottom_x:
            x[i].x = bottom_x
            v[i].x = 0

@ti.kernel
def Position_update(n: ti.i32):# Compute new position
    for i in range(n):
        x[i] += v[i] * dt

@ti.kernel
def stretch_constraint(n: ti.i32):
    for i in range(n):
        pos_i = x[i]
        posi_tmp = ti.Vector([0.0, 0.0])
        for j in range(constraint_num_neighbors[i]):
            p_j = constraint_neighbors[i, j]
            pos_j = x[p_j]
            x_ij = pos_i - pos_j
            dist_diff = x_ij.norm() - rest_length[i, p_j]
            grad = x_ij.normalized()
            position_delta = -particle_mass * particle_mass_invv * dist_diff * grad / constraint_num_neighbors[i]
            posi_tmp += position_delta
        position_delta_tmp[i] = posi_tmp

@ti.kernel
def apply_position_deltas(n: ti.i32):
    for i in range(n):
        x[i] += position_delta_tmp[i]

@ti.kernel
def updata_velosity(n: ti.i32): #updata velosity after combining constraints
    for i in range(n):
        v[i] = (x[i] - old_x[i]) * dt_inv

@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32): # Taichi doesn't support using Matrices as kernel arguments yet
    actuation_type[num_particles[None]] = 1
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    num_particles[None] += 1

@ti.kernel
def new_costraint(p1_index: ti.i32, p2_index: ti.i32, dist: ti.f32):
    # Connect with existing particles
    rest_length[p1_index, p2_index] = dist
    rest_length[p2_index, p1_index] = dist

def check_single_particle():
    cons = rest_length.to_numpy()
    invalid_particle = []
    for i in range(num_particles[None]):
        sum = 0
        for j in range(num_particles[None]):
            sum += cons[i ,j]
        if sum == 0:
            invalid_particle.append(i)
    return invalid_particle

def forward(n):
    #the first three steps -> only consider external force
    old_posi(n)
    substep(n)
    Position_update(n)
    #print(x.to_numpy()[0:n] - old_x.to_numpy()[0:n])
    collision_check(n)
    #print(x.to_numpy()[0:n])
    #add constraints
    for i in range(pbd_num_iters):
        constraint_neighbors.fill(-1)
        find_constraint(n)
        #print("This is ", i , "th iteration.")
        stretch_constraint(n)
        apply_position_deltas(n)
        collision_check(n)
    updata_velosity(n)

gui = ti.GUI('Mass Spring System', res=(640, 640), background_color=0xdddddd)

damping[None] = 8 #8 is the most suitable

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
    nearest_point_stick = ddata.result.nearest_points[1]
    nearest_point_stick = Transform(rota_degree, trans_x, trans_y, nearest_point_stick)
    collision = -1
    if previous_minimum_distance != -10000:
        if minimum_distance == -1:
            collision = -1
            print("The tip of the stick is inside the soft tissues.")
        elif minimum_distance >= tolerance:
            collision = 0
            print("No collision")
        elif minimum_distance >= 0 and minimum_distance < tolerance and minimum_distance < previous_minimum_distance:
            print("Collision")
            collision = 1
        # if nearest_point in verts:
        #     print("nearest point of the stick:", nearest_point_stick)
    previous_minimum_distance = minimum_distance
    return nearest_point, collision
    #print(ddata.result.pos)
    # if cdata.result.is_collision:
    #     for contact in cdata.result.contacts:
    #         print("Contact pos:", contact.pos) #the contact point is defined as one of the
    #         print(contact.pos in verts)
    #         print("Contact nor:", contact.normal)
    #         print("Penetra depth:", contact.penetration_depth)

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

def stick_configuration(rota_degree, trans_x, trans_y, length, width, top_left, top_right, bottom_left, bottom_right):
    rotation = np.array([[np.cos(rota_degree/180*np.pi), -np.sin(rota_degree/180*np.pi), 0.0],
                         [np.sin(rota_degree/180*np.pi), np.cos(rota_degree/180*np.pi), 0.0],
                         [0.0, 0.0, 1.0]])
    translation = np.array([trans_x, trans_y, 0.0])
    Transform = fcl.Transform(rotation, translation)
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

def compute_direction(transform_matrix, length, width, nearest_point):
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
    
    distance(top_left_trans, nearest_point)

    top_right_trans = transform_matrix @ top_right_homo
    top_right_trans = top_right_trans[:2]
    distance(top_right_trans, nearest_point)
    bottom_left_trans = transform_matrix @ bottom_left_homo
    bottom_left_trans = bottom_left_trans[:2]
    distance(bottom_left_trans, nearest_point)
    bottom_right_trans = transform_matrix @ bottom_right_homo
    bottom_right_trans = bottom_right_trans[:2]
    distance(bottom_right_trans, nearest_point)

def main():
    #Read all mesh points from txt.file
    points = []
    with open('./Contour_points/vertex.txt', 'r') as f:
        data = f.readlines()
    for line in data:
        odom = line.split()
        points.append([float(odom[0]), float(odom[1])])
    constraints = []
    with open('./Contour_points/constraint.txt', 'r') as f:
        data = f.readlines()
    for line in data:
        odom = line.split()
        constraints.append([float(odom[0]), float(odom[1]), float(odom[2])])
    num_particles[None] = 0
    for i in points:
        new_particle(i[0], i[1])
    n = num_particles[None]
    tri_matrix = np.zeros((n,n))
    for i in constraints:
        new_costraint(int(i[0]), int(i[1]), i[2])
        tri_matrix[int(i[0]),int(i[1])]=1
    tri_mesh = [] #triangle_mesh
    for i in range(n):
        for j in range(n):
            if tri_matrix[i,j] == 1:
                for k in range(j+1,n):
                    if tri_matrix[i,k] == 1 and tri_matrix[j,k] == 1:
                        if CheckRepeatMesh(i,j,k,tri_mesh): #True -> add False -> Remove
                            tri_mesh.append([i,j,k])
    tri_mesh = np.array(tri_mesh)
    single_particle_list = check_single_particle()
    index = 0
    omega = 0.5
    tolerance = 0.02
    length = 0.5
    width = 0.02
    top_left = np.array([-length / 2, width / 2, 0])
    top_right = np.array([length / 2, width / 2, 0])
    bottom_left = np.array([-length / 2, -width / 2, 0])
    bottom_right = np.array([length / 2, -width / 2, 0])
    rotate_direction = 'counter-clock-wise'
    while True:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == gui.SPACE:
                paused[None] = not paused[None]
            elif e.key == ti.GUI.LMB:
                print(e.pos[0], e.pos[1])

        collision = -10
        if not paused[None]:
            index += 1
            for step in range(1):
                forward(n)
                X = x.to_numpy()[:n]
                verts = np.c_[X,np.zeros(n)]
                if rotate_direction == 'counter-clock-wise':
                    angle = 45 + omega*index
                else:
                    angle = 45 - omega*index
                transform_matrix, stick_corners, stick = stick_configuration(angle, 0.3, 0.3, length, width, top_left, top_right, bottom_left, bottom_right)
                nearest_point, collision = CheckCollison(rotate_direction, verts, tri_mesh, stick, angle, 0.3, 0.3, tolerance) #argv1 & argv2 -> mesh argv3 -> stick

        gui.line(begin=stick_corners[0],end=stick_corners[1],color=0x0, radius=1)
        gui.line(begin=stick_corners[1],end=stick_corners[2],color=0x0, radius=1)
        gui.line(begin=stick_corners[2],end=stick_corners[3],color=0x0, radius=1)
        gui.line(begin=stick_corners[3],end=stick_corners[0],color=0x0, radius=1)
        actuation_type_tmp = np.ones((max_num_particles,),dtype=float)
        for i in range(n):
            if i not in single_particle_list:
                if collision == 1:
                    if X[i:i+1][0][0] == nearest_point[0]: #control point
                        actuation_type_tmp[i] = 2
                        compute_direction(transform_matrix, length, width, nearest_point)
                        gui.circles(X[i:i+1], color=0xffaa77, radius=12)
                    else:
                        gui.circles(X[i:i+1], color=0xffaa77, radius=5)
                else:
                    gui.circles(X[i:i+1], color=0xffaa77, radius=5)
        actuation_type.from_numpy(actuation_type_tmp)

        gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
        for i in range(num_particles[None]):
            for j in range(i + 1, num_particles[None]):
                if rest_length[i, j] != 0:
                    gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)

        gui.text(content=f'C: clear all; Space: pause', pos=(0, 0.95), color=0x0)
        gui.show()

if __name__ == '__main__':
    main()
