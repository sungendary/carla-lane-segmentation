import carla
import random
import os
import queue

# ===== 여기 세 개만 바꿔가며 여러 번 실행한다 =====
MAP_NAME = 'Town03'
# WEATHER = carla.WeatherParameters.ClearNoon # 맑은 낮
# WEATHER = carla.WeatherParameters.WetCloudyNoon   # 흐리고 젖은 낮
# WEATHER = carla.WeatherParameters.HardRainNoon    # 폭우 낮
WEATHER = carla.WeatherParameters.ClearSunset     # 맑은 석양
SAVE_SUBDIR = 'town03_sunset'
NUM_FRAMES = 200
# ================================================

# 1. 서버 접속
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)   # 맵 로딩은 시간이 걸려서 넉넉히

# 맵 로드 (월드가 새로 생성되므로 접속 직후에)
print(f"맵 로딩: {MAP_NAME} ...")
world = client.load_world(MAP_NAME)
world.set_weather(WEATHER)
print("맵 로딩 완료, 날씨 적용 완료")

# 트래픽 매니저 동기 모드
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

# 조건별 하위 폴더에 저장
base = os.path.join('dataset', SAVE_SUBDIR)
os.makedirs(os.path.join(base, 'rgb'), exist_ok=True)
os.makedirs(os.path.join(base, 'label_raw'), exist_ok=True)
os.makedirs(os.path.join(base, 'label_vis'), exist_ok=True)

vehicle = None
rgb_camera = None
seg_camera = None
original_settings = world.get_settings()

try:
    blueprint_library = world.get_blueprint_library()

    # 2. 동기 모드
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 3. 차량 스폰 + 오토파일럿
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True, traffic_manager.get_port())
    print("차량 스폰 완료:", vehicle.id)

    # 4. RGB + 세그멘테이션 카메라 (같은 위치·화각)
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', '800')
    rgb_bp.set_attribute('image_size_y', '600')
    rgb_bp.set_attribute('fov', '90')
    rgb_camera = world.spawn_actor(rgb_bp, camera_transform, attach_to=vehicle)

    seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    seg_bp.set_attribute('image_size_x', '800')
    seg_bp.set_attribute('image_size_y', '600')
    seg_bp.set_attribute('fov', '90')
    seg_camera = world.spawn_actor(seg_bp, camera_transform, attach_to=vehicle)
    print("카메라 2대 부착 완료")

    # 5. 큐 두 개
    rgb_queue = queue.Queue()
    seg_queue = queue.Queue()
    rgb_camera.listen(rgb_queue.put)
    seg_camera.listen(seg_queue.put)

    # 6. 워밍업
    for _ in range(20):
        world.tick()
        rgb_queue.get()
        seg_queue.get()

    # 7. 본 수집 루프
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
            print(f"진행: {i}/{NUM_FRAMES}")

    print(f"완료. {SAVE_SUBDIR}에 {NUM_FRAMES}쌍 저장.")

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
    print("정리 완료, 서버 설정 복구됨")