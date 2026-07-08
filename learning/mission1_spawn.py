import carla
import random

# 1. 서버 접속
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# 2. 월드 가져오기
world = client.get_world()

vehicle = None
try:
    # 3. 차량 블루프린트 고르기
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    # 4. 스폰 위치 고르기
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)

    # 5. 차량 스폰
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("차량 스폰 완료:", vehicle.id)
    print("스폰 위치:", vehicle.get_transform().location)   # 이 줄 추가

    # 6. 스펙테이터를 차량 위로 이동
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(
        carla.Transform(
            transform.location + carla.Location(z=30),
            carla.Rotation(pitch=-90)
        )
    )

    # 차량이 보이도록 잠시 대기
    input("엔터를 누르면 차량을 삭제하고 종료합니다...")

finally:
    # 7. 뒷정리
    if vehicle is not None:
        vehicle.destroy()
        print("차량 삭제 완료")
