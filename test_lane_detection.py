from lane_detection import LaneDetection
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key
import carla
import pygame
import random
import cv2 as cv

import DDRNet as net
import torch
from PIL import Image
from torchvision import transforms as tf

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

# init extra plot
fig = plt.figure()
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

device = torch.device('cuda')
model = net.DDRNet(num_class=19, use_aux=False).to(device)
model.load_state_dict(torch.load("ddrnet-23-slim.pth")['state_dict'])
model.eval()

colormap = [color for color in net.colormap.values()]   
colormap = torch.tensor(colormap).to(device)
obstacles = np.array([11, 12, 13, 14, 15, 17, 18])

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
    
    control.reverse = control.gear < 0

    control.hand_brake = keys[pygame.K_SPACE]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    f2b = LD_module.front2bev(raw)
    cutGray = LD_module.cut_gray(f2b)
    # cv.imshow("graycut", cutGray)
    grads = LD_module.edge_detection(cutGray)
    # normalizedGrads = cv.normalize(grads.astype(np.uint8), None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U).swapaxes(0, 1)
    # cv.imwrite("imageformask.png", normalizedGrads.swapaxes(0, 1))
    # bigger = cv.resize(grads, (640, 240))

    # cv.imshow("grads_hough", bigger)

    # # lane detection
    spline1, spline2, lanePointsImage = LD_module.lane_detection(f2b)

    # input = torch.tensor(raw.swapaxes(0, 1).copy()).unsqueeze(0)
    # # print(input.shape)
    # input = input.permute(0,3,1,2).float().to(device)
    # # print(input.shape)

    # segmented_pred = model(input)   

    # segmented_pred_obstacles = segmented_pred[:, (11, 12, 13, 14, 15, 17, 18), :, :].max(dim=1)[1]
    # print(np.unique(segmented_pred_obstacles.cpu().numpy()))
    
    # segmented_colors = colormap[segmented_pred_obstacles].cpu().numpy()

    # image = np.transpose(np.squeeze(input.cpu().numpy(), axis=0), (1, 2, 0))


    # for i in range(segmented_colors.shape[0]):
    #     # save_path = os.path.join(config.save_dir, img_names[i])
    #     # save_suffix = img_names[i].split('.')[-1]
    #     print(segmented_colors[i].shape)
    #     print(image.shape)
    #     pred = Image.fromarray(segmented_colors[i].astype(np.uint8))
        
     
    #     pred.save(str(i) + ".png")
        
    #     image = Image.fromarray(image.astype(np.uint8))
    #     image = Image.blend(image, pred, 0.3)
    #     image.save("blend" + str(i) + ".png")

    if steps % 2 == 0:
        # print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        LD_module.plot_state_lane(f2b, steps, fig)
    steps += 1

    # print(raw.shape, lanePointsImage.shape)
    concat = np.vstack((raw, lanePointsImage.swapaxes(0, 1)))
    img_surface = pygame.surfarray.make_surface(concat)
    display.blit(img_surface, (0, 0))
    pygame.display.flip()
    pygame.display.update()

    vehicle.apply_control(control)
    world.tick()

    clock.tick(30)

    # check if stop
