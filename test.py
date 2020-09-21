import fcl
import numpy as np

rota_degree = 0
rotation = np.array([[np.cos(rota_degree/180*np.pi), -np.sin(rota_degree/180*np.pi), 0.0],
                    [np.sin(rota_degree/180*np.pi), np.cos(rota_degree/180*np.pi), 0.0],
                    [0.0, 0.0, 1.0]])
translation = np.array([0, 0, 0.0])
Transform = fcl.Transform(rotation, translation)
box1 = fcl.CollisionObject(fcl.Box(1, 1, 0.0), Transform) #x,y,z length center at the origin
#box.setTranslation(np.array([0.0, 1.05000, 0.0])) #useful
#box.setRotation(rotation) #useful
#other options: setTransform setQuatRotation


rota_degree = 0
rotation = np.array([[np.cos(rota_degree/180*np.pi), -np.sin(rota_degree/180*np.pi), 0.0],
                     [np.sin(rota_degree/180*np.pi), np.cos(rota_degree/180*np.pi), 0.0],
                     [0.0, 0.0, 1.0]])
translation = np.array([1.4999, 0, 0.0])
Transform = fcl.Transform(rotation, translation)
box2 = fcl.CollisionObject(fcl.Box(3, 3, 0.0), Transform) #x,y,z length center at the origin
cylinder = fcl.CollisionObject(fcl.Cylinder(1, 0.0),Transform) #radius = 1
objs = [box1,cylinder]

manager = fcl.DynamicAABBTreeCollisionManager()
manager.registerObjects(objs)
manager.setup()


crequest = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
cdata = fcl.CollisionData(crequest, fcl.CollisionResult())

manager.collide(cdata, fcl.defaultCollisionCallback)
print(cdata.result.contacts)
for contact in cdata.result.contacts:
    print(contact.pos)
    print(contact.normal)
    print(contact.penetration_depth)
