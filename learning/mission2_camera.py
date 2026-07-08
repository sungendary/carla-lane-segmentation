import carla
import random
import os

# 1. 서버 접속
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# 저장 폴더 준비
os.makedirs('output', exist_ok=True)

vehicle = None
camera = None
try:
    blueprint_library = world.get_blueprint_library()

    # 2. 차량 스폰
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("차량 스폰 완료:", vehicle.id)

    # 3. 카메라 블루프린트 설정
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')

    # 4. 카메라를 차량에 붙여서 스폰 (차 앞쪽 1.5m, 높이 2.4m)
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    print("카메라 부착 완료:", camera.id)

    # 5. 콜백 등록: 이미지가 올 때마다 저장
    camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))

    # 6. 차량 자동주행 켜기 (움직여야 그림이 바뀐다)
    vehicle.set_autopilot(True)

    input("엔터를 누르면 종료합니다. 그동안 output 폴더에 이미지가 쌓입니다...")

finally:
    # 7. 뒷정리 (센서 먼저 멈추고 지운 뒤 차량 삭제)
    if camera is not None:
        camera.stop()
        camera.destroy()
        print("카메라 삭제 완료")
    if vehicle is not None:
        vehicle.destroy()
        print("차량 삭제 완료")