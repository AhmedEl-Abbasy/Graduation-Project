import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
    
import carla
from carla import ColorConverter as cc
import numpy as np
import cv2
import argparse
import collections
import datetime
import time
import math
import random
import weakref

"""
Keys Used :
    reverse
    manual
    gearup
    geardown
    autopilot
    light
    inlight
    leftblinker
    rightblinker
    highbeamlight
    up
    down
    right
    left
    center
    
class attributes:
    World(client.get_world())
    CameraManager(self.player)
    VehicleManager(world,start_in_autopilot_stats)
    
    
functions attributes:
    World:
        restart()
        destroy()
    CameraManager:
        toggle_camera()
        set_sensor(cam_index)
        next_sensor()
        toggle_recording()
        _parse_image(weak_self, image)
    VehicleManager:
        parse_events(client, world, key, val)
        _parse_vehicle_keys(key,val)
"""

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================
class World(object):
    def __init__(self, carla_world):
        self.world = carla_world
        self.actor_role_name = "Zero"
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
            
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_index = 0
        self._actor_filter = 'vehicle.*'
        self._gamma = 2.2
        self.restart()
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
            
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            #spawn_point = self.player.get_transform()
            #spawn_point.location.z += 2.0
            #spawn_point.rotation.roll = 0.0
            #spawn_point.rotation.pitch = 0.0
            spawn_point = carla.Transform(carla.Location(x=35.2, y=7.2, z=2), carla.Rotation(pitch=0.0, yaw=0, roll=0.0))
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            #spawn_points = self.map.get_spawn_points()
            #spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            spawn_point = carla.Transform(carla.Location(x=35.2, y=7.2, z=2), carla.Rotation(pitch=0.0, yaw=0, roll=0.0))
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        #self.collision_sensor = CollisionSensor(self.player)
        #self.lane_invasion_sensor = LaneInvasionSensor(self.player)
        #self.gnss_sensor = GnssSensor(self.player)
        #self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        #actor_type = get_actor_display_name(self.player)

    #def toggle_radar(self):
        #if self.radar_sensor is None:
            #self.radar_sensor = RadarSensor(self.player)
        #elif self.radar_sensor.sensor is not None:
            #self.radar_sensor.sensor.destroy()
            #self.radar_sensor = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor
            #self.collision_sensor.sensor,
            #self.lane_invasion_sensor.sensor,
            #self.gnss_sensor.sensor,
            #self.imu_sensor.sensor
            ]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()

# ==============================================================================
# -- CAMERAManager --------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', f'{IM_WIDTH}')
                bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(2.2))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
            
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min((IM_WIDTH, IM_HEIGHT)) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * IM_WIDTH, 0.5 * IM_HEIGHT)
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (IM_WIDTH, IM_HEIGHT, 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
        if self.recording:
            #image.save_to_disk('_out/%08d' % image.frame)
            out.write(array)
            
# ==============================================================================
# -- Vehicle Controller---------------------------------------------------------
# ==============================================================================

class VehicleManager(object):
    def __init__(self, world,start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl()
        self._lights = carla.VehicleLightState.NONE
        world.player.set_autopilot(self._autopilot_enabled)
        world.player.set_light_state(self._lights)
        self._steer_cache = 0.0
        
    def parse_events(self,client,world,key,val):
        current_lights = self._lights
        if key == "reverse":
            self._control.gear = 1 if self._control.reverse else -1
        elif key == "manual":
            self._control.manual_gear_shift = not self._control.manual_gear_shift
            self._control.gear = world.player.get_control().gear
        elif self._control.manual_gear_shift and key == "gearup":
            self._control.gear = max(-1, self._control.gear - 1)
        elif self._control.manual_gear_shift and key == "geardown":
            self._control.gear = self._control.gear + 1
        elif key == "autopilot":
            self._autopilot_enabled = not self._autopilot_enabled
            world.player.set_autopilot(self._autopilot_enabled)
        elif key == "highbeamlight":
            current_lights ^= carla.VehicleLightState.HighBeam
        elif key == "light":
            # closed -> position -> low beam -> fog
            if not self._lights & carla.VehicleLightState.Position:
                current_lights |= carla.VehicleLightState.Position
            else:
                current_lights |= carla.VehicleLightState.LowBeam
            if self._lights & carla.VehicleLightState.LowBeam:
                current_lights |= carla.VehicleLightState.Fog
            if self._lights & carla.VehicleLightState.Fog:
                current_lights ^= carla.VehicleLightState.Position
                current_lights ^= carla.VehicleLightState.LowBeam
                current_lights ^= carla.VehicleLightState.Fog
        elif key == "inlight":
            current_lights ^= carla.VehicleLightState.Interior
        elif key == "leftblinker":
            current_lights ^= carla.VehicleLightState.LeftBlinker
        elif key == "rightblinker":
            current_lights ^= carla.VehicleLightState.RightBlinker
                    
        if not self._autopilot_enabled:
            self._parse_vehicle_keys(key,val)
            self._control.reverse = self._control.gear < 0
            # Set automatic control-related vehicle lights
            if self._control.brake:
                current_lights |= carla.VehicleLightState.Brake
            else: # Remove the Brake flag
                current_lights &= ~carla.VehicleLightState.Brake
            if self._control.reverse:
                current_lights |= carla.VehicleLightState.Reverse
            else: # Remove the Reverse flag
                current_lights &= ~carla.VehicleLightState.Reverse
            if current_lights != self._lights: # Change the light state only if necessary
                self._lights = current_lights
                world.player.set_light_state(carla.VehicleLightState(self._lights))
            world.player.apply_control(self._control)
        
    def _parse_vehicle_keys(self, Dir, Val):
        if Dir == "up":
            self._control.throttle = min(self._control.throttle + Val, 1)
            self._control.brake = 0
        if Dir =="down":
            self._control.throttle = 0
            self._control.brake = min(self._control.brake + Val, 1)
        if Dir =="left":
            self._steer_cache -= Val
        elif Dir =="right":
            self._steer_cache = Val
        elif Dir == "center":
            self._steer_cache = 0
        
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        
    def _is_quit_shortcut(key):
        return key == "EXIT"

# ==============================================================================
# -- Main-----------------------------------------------------------------------
# ==============================================================================

IM_WIDTH = 1280
IM_HEIGHT = 720
start_in_autopilot_stats = 0
world = None
IP = "127.0.0.1"
PORT = 2000
out = cv2.VideoWriter('Car_Manager.avi',cv2.VideoWriter_fourcc('M','J','P','G'),24,(IM_WIDTH,IM_HEIGHT))
try:
    client = carla.Client(IP,PORT)
    client.set_timeout(2.0)
    world = World(client.get_world())
    controller = VehicleManager(world,start_in_autopilot_stats)
    world.camera_manager.toggle_recording()
    controller.parse_events(client,world,"up",.7)
    time.sleep(2)
    controller.parse_events(client,world,"left",1)
    time.sleep(1)
    controller.parse_events(client,world,"right",.7)
    time.sleep(1)
    controller.parse_events(client,world,"center",1)
    time.sleep(2)
    controller.parse_events(client,world,"down",.7)
    time.sleep(1)
    controller.parse_events(client,world,"reverse",1)
    time.sleep(1)
    controller.parse_events(client,world,"up",.7)
    time.sleep(5)
    controller.parse_events(client,world,"left",.5)
    time.sleep(1)
    controller.parse_events(client,world,"right",.5)
    time.sleep(2)
    controller.parse_events(client,world,"down",.7)
    time.sleep(2)
    #world.player.apply_control(carla.VehicleControl(throttle=1.0,steer=0.0))
    out.release()
    cv2.destroyAllWindows()

finally:
    if world is not None:
        world.destroy()
    print("All Cleared~")