"""
Collect paired (RGB, semantic-label) frames from CARLA in synchronous mode.

All tunable values live in configs/config.yaml under the `collect:` section,
so changing map/weather/output does not modify this file.
"""
import carla
import random
import os
import queue

from config import load_config, repo_path

cfg = load_config('collect')

MAP_NAME = cfg['map']
WEATHER = getattr(carla.WeatherParameters, cfg['weather'])
SAVE_SUBDIR = cfg['out']
NUM_FRAMES = cfg['frames']

# 1. connect
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)

print(f"Loading map: {MAP_NAME} ...")
world = client.load_world(MAP_NAME)
world.set_weather(WEATHER)
print("Map loaded, weather applied")

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

# output folders (under repo-root/dataset/<SAVE_SUBDIR>/...)
base = repo_path('dataset', SAVE_SUBDIR)
os.makedirs(os.path.join(base, 'rgb'), exist_ok=True)
os.makedirs(os.path.join(base, 'label_raw'), exist_ok=True)
os.makedirs(os.path.join(base, 'label_vis'), exist_ok=True)

vehicle = None
rgb_camera = None
seg_camera = None
original_settings = world.get_settings()

try:
    blueprint_library = world.get_blueprint_library()

    # 2. synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 3. spawn vehicle + autopilot
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True, traffic_manager.get_port())
    print("Vehicle spawned:", vehicle.id)

    # 4. RGB + semantic cameras (same transform / resolution / fov)
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    w, h, fov = str(cfg['image_width']), str(cfg['image_height']), str(cfg['fov'])

    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', w)
    rgb_bp.set_attribute('image_size_y', h)
    rgb_bp.set_attribute('fov', fov)
    rgb_camera = world.spawn_actor(rgb_bp, camera_transform, attach_to=vehicle)

    seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    seg_bp.set_attribute('image_size_x', w)
    seg_bp.set_attribute('image_size_y', h)
    seg_bp.set_attribute('fov', fov)
    seg_camera = world.spawn_actor(seg_bp, camera_transform, attach_to=vehicle)
    print("Two cameras attached")

    # 5. queues
    rgb_queue = queue.Queue()
    seg_queue = queue.Queue()
    rgb_camera.listen(rgb_queue.put)
    seg_camera.listen(seg_queue.put)

    # 6. warm-up (not saved)
    for _ in range(20):
        world.tick()
        rgb_queue.get()
        seg_queue.get()

    # 7. main collection loop
    for i in range(NUM_FRAMES):
        world.tick()
        rgb_image = rgb_queue.get()
        seg_image = seg_queue.get()
        assert rgb_image.frame == seg_image.frame

        rgb_image.save_to_disk(os.path.join(base, 'rgb', '%06d.png' % rgb_image.frame))
        seg_image.save_to_disk(os.path.join(base, 'label_raw', '%06d.png' % seg_image.frame),
                               carla.ColorConverter.Raw)
        seg_image.save_to_disk(os.path.join(base, 'label_vis', '%06d.png' % seg_image.frame),
                               carla.ColorConverter.CityScapesPalette)
        if i % 50 == 0:
            print(f"progress: {i}/{NUM_FRAMES}")

    print(f"Done. Saved {NUM_FRAMES} pairs to {SAVE_SUBDIR}.")

finally:
    if rgb_camera is not None:
        rgb_camera.stop()
        rgb_camera.destroy()
    if seg_camera is not None:
        seg_camera.stop()
        seg_camera.destroy()
    if vehicle is not None:
        vehicle.destroy()
    traffic_manager.set_synchronous_mode(False)
    world.apply_settings(original_settings)
    print("Cleanup done, server settings restored")
