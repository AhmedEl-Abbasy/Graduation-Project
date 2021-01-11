import glob
import os
import sys
import random
import time
import numpy as np
import cv2



try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

IM_WIDTH = 681
IM_HEIGHT = 696
def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape(IM_HEIGHT,IM_WIDTH,4)
    i3 = i2[:,:,:3]
    cv2.imshow("",i3)
    cv2.waitKey(1)
    return i3/255.0

actor_list = []
try:
    client = carla.Client("127.0.0.1",2000)
    client.set_timeout(2.0)
    
    world = client.get_world()
    
    blueprint_library = world.get_blueprint_library()
    
    bp = blueprint_library.filter("model3")[0]
    #spawn_point = random.choice(world.get_map().get_spawn_points())
    spawn_point = carla.Transform(carla.Location(x=35.2, y=7.2, z=5), carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
    vehicle = world.spawn_actor(bp,spawn_point)
    vehicle.set_autopilot(True)
    #vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=0.0))
    actor_list.append(vehicle)
    
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
    cam_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
    cam_bp.set_attribute("fov","90")
    spawn_cam_point = carla.Transform(carla.Location(x=-5,z=3))
    camera = world.spawn_actor(cam_bp,spawn_cam_point,attach_to = vehicle)
    actor_list.append(camera)
    camera.listen(lambda data: process_img(data))

    time.sleep(10)
finally:
    for actor in actor_list:
        actor.destroy()
    print("All Cleared~")
