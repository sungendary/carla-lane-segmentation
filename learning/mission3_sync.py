import carla
import random
import os
import queue

# 1. 서버 접속
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# 트래픽 매니저를 동기 모드로 설정
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

os.makedirs('output_sync', exist_ok=True)

vehicle = None
camera = None
original_settings = world.get_settings()   # 원래 설정 백업

try:
    blueprint_library = world.get_blueprint_library()

    # 2. 동기 모드 켜기
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05       # 한 tick = 0.05초 (20 FPS)
    world.apply_settings(settings)

    # 3. 차량 스폰
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True, traffic_manager.get_port())
    print("차량 스폰 완료:", vehicle.id)

    # 4. 카메라 스폰
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    print("카메라 부착 완료:", camera.id)

    # 5. 큐 만들고 콜백이 이미지를 큐에 넣게 등록
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    # 워밍업: 차량이 자리잡고 움직이기 시작하도록 20스텝 진행 (저장 안 함)
    for _ in range(20):
        world.tick()
        image_queue.get()   # 큐가 쌓이지 않게 꺼내서 버림

    # 6. 메인 루프: 내가 직접 한 스텝씩 진행
    for frame in range(200):          # 200 스텝만
        world.tick()                  # 세계를 한 칸 전진
        image = image_queue.get()     # 이 tick의 이미지가 올 때까지 대기 후 꺼냄
        image.save_to_disk('output_sync/%06d.png' % image.frame)
        print("프레임 저장:", image.frame)

    print("완료. 총 200프레임 저장.")

finally:
    # 7. 뒷정리 + 반드시 원래 설정(async)으로 복구
    if camera is not None:
        camera.stop()
        camera.destroy()
    if vehicle is not None:
        vehicle.destroy()
    traffic_manager.set_synchronous_mode(False)
    world.apply_settings(original_settings)   # 이게 없으면 서버가 계속 멈춰 있음
    print("정리 완료, 서버 설정 복구됨")