import taichi as ti
import math as math

ti.init(debug=False,arch=ti.cpu)
real = ti.f32 #data type f32 -> float in C

max_num_particles = 1000
dt = 1e-3
dx = 0.02
dim = 2

scalar = lambda: ti.var(dt=real) #2D dense tensor
vec = lambda: ti.Vector(dim, dt=real) #2*1 vector(each element in a tensor)
mat = lambda: ti.Matrix(dim, dim, dt=real) #2*2 matrix(each element in a tensor)

num_particles = ti.var(ti.i32, shape=())
spring_stiffness = ti.var(ti.f32, shape=())
paused = ti.var(ti.i32, shape=())
damping = ti.var(ti.f32, shape=())

particle_mass = 1
bottom_y = 0.05
bottom_x = 0.95

x, v = vec(), vec()
actuation_type = scalar()
# rest_length[i, j] = 0 means i and j are not connected
rest_length = scalar()


connection_radius = 0.03
gravity = [0, -9] #direction
H_force = [0, 0] #another gr

@ti.layout  #Environment layout(placed in ti.layout) initialization of the dimensiond of each tensor variables(global)
def place():
    ti.root.dense(ti.ij, (max_num_particles, max_num_particles)).place(rest_length)
    ti.root.dense(ti.i, max_num_particles).place(x, v, actuation_type) #initialzation to zero

@ti.kernel
def substep(n: ti.i32, t: ti.i32): # Compute force and new velocity
    for i in range(n):
        v[i] *= ti.exp(-dt * damping[None]) # damping
        total_force = ti.Vector(gravity) * particle_mass #gravity -> accelaration
        if actuation_type[i] == 1:
            #total_force = ti.Vector([9.8 * t, 0], real) * particle_mass
            total_force = H_force * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                total_force += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()
                # if i == 96:
                #     print("x[i]:", x[i], "x[j]:", x[j])
                #     print(x_ij.norm(), rest_length[i, j], x_ij.normalized())
                #     print("this is",i, "and", j, ":", -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized())
        v[i] += dt * total_force / particle_mass
        # if v[i].norm() >= 1:
        #     print(i, v[i])
        #     print(i, "total_force:", total_force)
        #     print("norm:", total_force.normalized())

@ti.kernel
def collision_check(n: ti.i32):# Collide with ground
    for i in range(n):
        if x[i].y < bottom_y:
            x[i].y = bottom_y
            v[i].y = 0
        if x[i].x > bottom_x:
            x[i].x = bottom_x
            v[i].x = 0

@ti.kernel
def Position_update(n: ti.i32):# Compute new position
    for i in range(num_particles[None]):
        x[i] += v[i] * dt
        if v[i].norm() >= 1:
            print(i, v[i])

@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32): # Taichi doesn't support using Matrices as kernel arguments yet
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    num_particles[None] += 1

@ti.kernel
def new_costraint(p1_index: ti.i32, p2_index: ti.i32, dist: ti.f32):
    # Connect with existing particles
    rest_length[p1_index, p2_index] = dist
    rest_length[p2_index, p1_index] = dist
# @ti.kernel
# def new_particle(pos_x: ti.f32, pos_y: ti.f32): # Taichi doesn't support using Matrices as kernel arguments yet
#     new_particle_id = num_particles[None]
#     x[new_particle_id] = [pos_x, pos_y]
#     v[new_particle_id] = [0, 0]
#     num_particles[None] += 1
#
#     # Connect with existing particles
#     for i in range(new_particle_id):
#         dist = (x[new_particle_id] - x[i]).norm()
#         if dist < connection_radius:
#             rest_length[i, new_particle_id] = dist
#             rest_length[new_particle_id, i] = dist
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

def forward(n, time_step):
    #print("next_step is subsetp")
    substep(n, time_step)
    collision_check(n)
    #print("next_step is position_update")
    Position_update(n)

gui = ti.GUI('Mass Spring System', res=(640, 640), background_color=0xdddddd)

class Scene:
    def __init__(self, resolution):
        self.n_particles = 0
        self.resolution = resolution
        self.actuation = []
        self.x = []

    def add_line(self, x_start, x_end, y):
        w = x_end - x_start
        w_count = int(w / dx) * self.resolution  # N particles in each grid(N = 1,2,3,4,5,..)
        if w_count == 0: #tangent line
            self.x.append([
                (x_start + x_end) / 2,
                y
            ])
            self.n_particles += 1
            self.actuation.append(0)
        else:
            real_dx = w / w_count  # step
            flag = False
            for i in range(w_count):
                if flag == False and 0.2 < x_start < 0.4 and 0.4 < y < 0.6:
                    flag = True
                    self.actuation.append(1)
                else:
                    self.actuation.append(0)
                self.x.append([  # each particle's position
                    x_start + (i + 0.5) * real_dx,  # 0.5 * real_dx is very small
                    y
                ])
                self.n_particles += 1

    def finalize(self):
        global max_num_particles
        max_num_particles = self.n_particles + 50 #the true number of particles
        #50 for manually added particles by mouse-click
        print('maximum number of particles:', max_num_particles)

def x_position_circle(center_x, center_y, radius, y):
    x_start = center_x - math.sqrt(abs(radius ** 2 - (y - center_y) ** 2))
    x_end = center_y + math.sqrt(abs(radius ** 2 - (y - center_y) ** 2))
    return x_start, x_end

def robot_circle(scene, center_x, center_y, radius): #the confiurtion of the soft robot?
    h = 2 * radius
    h_count = int(h / dx) * scene.resolution
    real_dy = h / h_count
    for j in range(h_count + 1):
        y = center_y - radius + j * real_dy
        x_start, x_end = x_position_circle(center_x, center_y, radius, y)
        scene.add_line(x_start, x_end, y)

spring_stiffness[None] = 50000
damping[None] = 20

def main():
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
    for i in constraints:
        new_costraint(int(i[0]), int(i[1]), i[2])
    single_particle_list = check_single_particle()
    # p1 = [0.3, 0.5]
    # p2 = [0.5, 0.7]
    # p3 = [0.7, 0.5]
    # p4 = [0.5, 0.3]
    # new_particle(p4[0], p4[1])
    # new_particle(p2[0], p2[1])
    # new_particle(p3[0], p3[1])
    # new_particle(p1[0], p1[1])
    # actuation_type[0] = 1
    # for i in range(3):
    #     actuation_type[1 + i] = 0
    time_step = 0
    index = 0
    X = x.to_numpy()
    for i in range(num_particles[None]):
        if X[i][0] <= 0 or x[i][0] >= 1:
            print(X[i])
    print("********")
    while True:
        # for e in gui.get_events(ti.GUI.PRESS):
        #     if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
        #         exit()
        #     elif e.key == gui.SPACE:
        #         paused[None] = not paused[None]
        #     elif e.key == ti.GUI.LMB:
        #         new_particle(e.pos[0], e.pos[1])
        #     elif e.key == 'c':
        #         num_particles[None] = 0
        #         rest_length.fill(0)
        #     elif e.key == 's':
        #         if gui.is_pressed('Shift'):
        #             spring_stiffness[None] /= 1.1
        #         else:
        #             spring_stiffness[None] *= 1.1
        #     elif e.key == 'd':
        #         if gui.is_pressed('Shift'):
        #             damping[None] /= 1.1
        #         else:
        #             damping[None] *= 1.1
        index += 1
        if index % 2 == 0:
            if time_step <= 14:
                time_step += 1
        if not paused[None]:
            for step in range(10):
                n = num_particles[None]
                forward(n, time_step)
                X = x.to_numpy()
                # for i in range(num_particles[None]):
                #     if X[i][0] <= 0 or x[i][0] >= 1:
                #         print(X[i])
                # print("******")
        X = x.to_numpy()
        for i in range(n):
            # if scene.actuation[i] == 1:
            #     gui.circles(X[i:i+1], color=0x33bb76, radius=5)
            # else:
            if i not in single_particle_list:
                gui.circles(X[i:i+1], color=0xffaa77, radius=5)
        gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
        for i in range(num_particles[None]):
            for j in range(i + 1, num_particles[None]):
                if rest_length[i, j] != 0:
                    gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)

        gui.text(content=f'C: clear all; Space: pause', pos=(0, 0.95), color=0x0)
        gui.text(content=f'S: Spring stiffness {spring_stiffness[None]:.1f}', pos=(0, 0.9), color=0x0)
        gui.text(content=f'D: damping {damping[None]:.2f}', pos=(0, 0.85), color=0x0)
        gui.show()

if __name__ == '__main__':
    main()
