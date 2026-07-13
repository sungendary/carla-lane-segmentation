"""
Collect paired (RGB, semantic-label) frames from CARLA in synchronous mode.

A speed filter skips frames where the vehicle is (near-)stationary, so that
red-light / traffic stops do not flood the set with near-identical frames.
The simulation still ticks during stops; only saving is skipped.

Two ways to run one set:
  A) from config.yaml (no args):   python src/collect_data.py
  B) from CLI args (batch runner, one process per set):
       python src/collect_data.py --map Town01 --weather ClearNoon \
                                  --out town01_clearnoon --frames 1500
"""
import argparse
import math
import os
import random
import queue

import carla
from config import load_config, repo_path

# minimum speed (m/s) required to save a frame. ~1.0 m/s = 3.6 km/h:
# filters out red-light/jam stops while keeping genuine slow driving.
MIN_SPEED = 1.0
# safety cap so a long stop can't loop forever: at most this many ticks
# per target frame before we give up waiting.
MAX_TICK_FACTOR = 15


def speed_of(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def collect_one(client, map_name, weather_name, out_subdir, num_frames,
                image_width=800, image_height=600, fov=90):
    """Collect one map/weather set on an already-connected client."""
    weather = getattr(carla.WeatherParameters, weather_name)

    print(f"[collect] loading {map_name} / {weather_name} -> {out_subdir}")
    world = client.load_world(map_name)
    world.set_weather(weather)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    base = repo_path('dataset', out_subdir)
    os.makedirs(os.path.join(base, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(base, 'label_raw'), exist_ok=True)
    os.makedirs(os.path.join(base, 'label_vis'), exist_ok=True)

    vehicle = None
    rgb_camera = None
    seg_camera = None
    original_settings = world.get_settings()

    try:
        blueprint_library = world.get_blueprint_library()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True, traffic_manager.get_port())

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        w, h, f = str(image_width), str(image_height), str(fov)

        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', w)
        rgb_bp.set_attribute('image_size_y', h)
        rgb_bp.set_attribute('fov', f)
        rgb_camera = world.spawn_actor(rgb_bp, camera_transform, attach_to=vehicle)

        seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', w)
        seg_bp.set_attribute('image_size_y', h)
        seg_bp.set_attribute('fov', f)
        seg_camera = world.spawn_actor(seg_bp, camera_transform, attach_to=vehicle)

        rgb_queue = queue.Queue()
        seg_queue = queue.Queue()
        rgb_camera.listen(rgb_queue.put)
        seg_camera.listen(seg_queue.put)

        for _ in range(20):          # warm-up
            world.tick()
            rgb_queue.get()
            seg_queue.get()

        # collect until num_frames MOVING frames are saved (or tick cap hit)
        saved = 0
        ticks = 0
        max_ticks = num_frames * MAX_TICK_FACTOR
        skipped_stationary = 0

        while saved < num_frames and ticks < max_ticks:
            world.tick()
            ticks += 1
            rgb_image = rgb_queue.get()
            seg_image = seg_queue.get()
            assert rgb_image.frame == seg_image.frame

            # speed filter: skip near-stationary frames
            if speed_of(vehicle) < MIN_SPEED:
                skipped_stationary += 1
                continue

            rgb_image.save_to_disk(os.path.join(base, 'rgb', '%06d.png' % rgb_image.frame))
            seg_image.save_to_disk(os.path.join(base, 'label_raw', '%06d.png' % seg_image.frame),
                                   carla.ColorConverter.Raw)
            seg_image.save_to_disk(os.path.join(base, 'label_vis', '%06d.png' % seg_image.frame),
                                   carla.ColorConverter.CityScapesPalette)
            saved += 1
            if saved % 100 == 0:
                print(f"  {out_subdir}: {saved}/{num_frames} "
                      f"(skipped {skipped_stationary} stationary)")

        print(f"[collect] done {out_subdir}: {saved} frames "
              f"(skipped {skipped_stationary} stationary, {ticks} ticks)")
        if saved < num_frames:
            print(f"[collect] WARNING {out_subdir}: hit tick cap, "
                  f"only {saved}/{num_frames}")
        return saved

    finally:
        for sensor in (rgb_camera, seg_camera):
            if sensor is not None:
                try:
                    sensor.stop()
                except Exception:
                    pass
        try:
            world.tick()
        except Exception:
            pass
        for actor in (rgb_camera, seg_camera, vehicle):
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
        try:
            traffic_manager.set_synchronous_mode(False)
            world.apply_settings(original_settings)
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--map')
    p.add_argument('--weather')
    p.add_argument('--out')
    p.add_argument('--frames', type=int)
    p.add_argument('--width', type=int, default=800)
    p.add_argument('--height', type=int, default=600)
    p.add_argument('--fov', type=int, default=90)
    return p.parse_args()


def main():
    args = parse_args()
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)

    if args.map:
        collect_one(client, args.map, args.weather, args.out, args.frames,
                    args.width, args.height, args.fov)
    else:
        cfg = load_config('collect')
        collect_one(client, cfg['map'], cfg['weather'], cfg['out'], cfg['frames'],
                    cfg['image_width'], cfg['image_height'], cfg['fov'])
    print("Cleanup done, server settings restored")


if __name__ == '__main__':
    main()