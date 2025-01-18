# from lane_detection import LaneDetection
# from waypoint_prediction import waypoint_prediction, target_speed_prediction
# from lateral_control import LateralController
# from longitudinal_control import LongitudinalController
# import matplotlib.pyplot as plt
# import numpy as np
# import pyglet
# from pyglet import gl
# from pyglet.window import key

# # action variables

# # init carla environement

# # define variables
# steps = 0

# # init modules of the pipeline
# LD_module = LaneDetection()
# LatC_module = LateralController()
# LongC_module = LongitudinalController()

# # init extra plot
# fig = plt.figure()
# plt.ion()
# plt.show()

# while True:
#     # perform step

#     # lane detection
#     lane1, lane2 = LD_module.lane_detection(s)

#     # waypoint and target_speed prediction
#     waypoints = waypoint_prediction(lane1, lane2)
#     target_speed = target_speed_prediction(waypoints, max_speed=60, exp_constant=4.5)

#     # control
#     a[0] = LatC_module.stanley(waypoints, speed)
#     a[1], a[2] = LongC_module.control(speed, target_speed)

#     # outputs during training
#     if steps % 2 == 0:
#         print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
#         print("speed {:+0.2f} targetspeed {:+0.2f}".format(speed, target_speed))

#         #LD_module.plot_state_lane(s, steps, fig, waypoints=waypoints)
#         LongC_module.plot_speed(speed, target_speed, steps, fig)
#     steps += 1

#     # check if stop

from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
from lateral_control import LateralController
from longitudinal_control import LongitudinalController

import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key
import carla
import pygame
import random
import cv2 as cv

raw = np.empty((320, 240, 3))

def process_image(display, image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))  

    array = array[:, :, :3]
    rgb_array = array[:, :, ::-1]
    rgb_array = rgb_array.swapaxes(0, 1)

    global raw
    raw = rgb_array

# action variables
steer = 0
throttle = 0
brake = 0

# init carla environement
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(4.0)
world = client.load_world('Town02') 

vehicle_blueprint = world.get_blueprint_library().filter('model3')[0]
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
vehicle.set_autopilot(False)    

pygame.init()
display = pygame.display.set_mode(
            (640, 240))

# define variables
steps = 0
control = carla.VehicleControl()
clock = pygame.time.Clock()
done = False

# init modules of the pipeline
LD_module = LaneDetection()
LatC_module = LateralController()
LongC_module = LongitudinalController()


# init extra plot
fig1 = plt.figure(1)
# plt.ion()
# plt.show()

fig2 = plt.figure(2)
plt.ion()
plt.show()

# cv.namedWindow("graycut", cv.WINDOW_AUTOSIZE)
# cv.namedWindow("grads", cv.WINDOW_AUTOSIZE)

camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))  # Adjust camera position

camera_bp.set_attribute('image_size_x', '320')
camera_bp.set_attribute('image_size_y', '240')
camera_bp.set_attribute('fov', '90')

camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
camera.listen(lambda image: process_image(display, image))

auto = -1
while not done:
    # perform step
    keys = pygame.key.get_pressed() 

    if keys[pygame.K_UP] or keys[pygame.K_w]:
        control.throttle = min(control.throttle + 0.05, 1.0)
    else:
        control.throttle = 0.0

    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        control.brake = min(control.brake + 0.2, 1.0)
    else:
        control.brake = 0.0

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        control.steer = max(control.steer - 0.05, -1.0)
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        control.steer = min(control.steer + 0.05, 1.0)
    else:
        control.steer = 0.0
    
    if keys[pygame.K_q]:
        control.gear = 1 if control.reverse else -1

    if keys[pygame.K_a]:
        auto = auto * -1
    
    control.reverse = control.gear < 0

    control.hand_brake = keys[pygame.K_SPACE]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    # lane detection
    f2b = LD_module.front2bev(raw)
    lane1, lane2, lanePointsImage = LD_module.lane_detection(f2b)

    # waypoint and target_speed prediction
    waypoints = waypoint_prediction(lane1, lane2)
    target_speed = target_speed_prediction(waypoints)
    # print(target_speed)

    velocity = vehicle.get_velocity()
    # print(velocity)
    speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    speed_km_h = speed_m_s * 3.6

    if (auto < 0):
        control.steer = LatC_module.stanley(waypoints, speed_km_h)
        control.throttle, control.brake = LongC_module.control(speed_km_h, target_speed)
    vehicle.apply_control(control)

    # outputs during training
    if steps % 1 == 0:
        # print("\naction " + str(["{:+0.2f}".format(x) for x in ]))
        LD_module.plot_state_lane(f2b, steps, fig1, waypoints=waypoints)
        LongC_module.plot_speed(speed_km_h, target_speed, steps, fig2)

    steps += 1
    # check if stop
    concat = np.vstack((raw, f2b.swapaxes(0, 1)))
    img_surface = pygame.surfarray.make_surface(concat)
    display.blit(img_surface, (0, 0))

    world.tick()

    pygame.display.flip()
    pygame.display.update()
    clock.tick(30)
