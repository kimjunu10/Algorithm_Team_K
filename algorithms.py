import math
from itertools import permutations
import numpy as np

# 서울 중심 좌표와 반지름 설정
# 서울의 중심 좌표(위도, 경도)와 반경(15km)을 설정하여 기준으로 사용합니다.
SEOUL_CENTER_LAT = 37.544444  # 서울 중심 위도
SEOUL_CENTER_LNG = 127.009417  # 서울 중심 경도
SEOUL_RADIUS_KM = 15  # 서울 반경 (km)

# 두 좌표가 서울 내부에 있는지 확인
def is_in_seoul(lat, lng):
    """
    주어진 좌표(lat, lng)가 서울 중심 반경 내에 있는지 확인합니다.
    Haversine 공식을 사용해 두 좌표 간의 거리를 계산하고, 
    계산된 거리가 서울 반경보다 작거나 같은지 판단합니다.
    """
    R = 6371.0  # 지구 반지름 (km) - Haversine 공식에서 사용되는 기본 값
    # 좌표를 라디안으로 변환
    # 위도와 경도를 Haversine 계산에 사용하기 위해 라디안 값으로 변환합니다.
    lat1, lng1 = math.radians(SEOUL_CENTER_LAT), math.radians(SEOUL_CENTER_LNG)  # 서울 중심 좌표 (라디안)
    lat2, lng2 = math.radians(lat), math.radians(lng)  # 입력받은 좌표 (라디안)

    dlat = lat2 - lat1  # 위도 차이 계산
    dlng = lng2 - lng1  # 경도 차이 계산

    # Haversine 공식으로 두 좌표 간의 거리 계산
    # 두 좌표의 중심각(두 점 사이의 구면 거리의 각도)을 계산합니다.
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    # atan2로 구면상 두 점 간의 실제 거리 비율을 계산
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 거리 = 지구 반지름 * 중심각
    # 서울 중심과 입력된 좌표 간의 거리를 계산하여 반경과 비교합니다.
    return R * c <= SEOUL_RADIUS_KM

# 각 포인트의 서울 내부 여부를 확인하여 캐싱
def classify_points(points):
    """
    입력된 각 포인트(위도, 경도, 이름 등)에 대해 서울 내부 여부를 판단합니다.
    """
    return [
        # 각 포인트에 대해 `is_in_seoul` 함수를 호출하여 결과를 'in_seoul' 키에 추가
        {**point, 'in_seoul': is_in_seoul(point['lat'], point['lng'])}
        for point in points
    ]

# Haversine 공식을 사용하여 두 좌표 간의 거리 계산
def haversine(lat1, lng1, lat2, lng2):
    """
    두 좌표 간의 직선 거리를 Haversine 공식을 사용하여 계산합니다.
    구면 거리 계산을 통해 두 점 간의 거리(km)를 반환합니다.
    """
    R = 6371.0  # 지구 반지름 (km)
    # 좌표를 라디안으로 변환
    # 입력된 위도와 경도를 모두 라디안 값으로 변환합니다.
    phi1, phi2 = math.radians(lat1), math.radians(lat2)  # 두 지점의 위도 (라디안)
    delta_phi = math.radians(lat2 - lat1)  # 위도 차이 (라디안)
    delta_lambda = math.radians(lng2 - lng1)  # 경도 차이 (라디안)

    # Haversine 공식에 필요한 값을 계산
    # 두 좌표 사이의 구면 거리 비율을 계산
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    # 구면 거리의 각도를 계산하여 두 점 간의 거리(km)를 구합니다.
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # 두 좌표 간의 거리(km)를 반환

# 거리와 속도를 조정하여 이동 시간 계산
def calculate_speed_and_adjusted_distance(distance_km, in_seoul1, in_seoul2):
    """
    거리(distance_km)와 서울 내부 여부(in_seoul1, in_seoul2)를 기반으로 이동 속도와 조정된 거리를 계산합니다.
    - 서울 내부(서울 반경 내)에 있을 경우 더 느린 속도가 적용됩니다.
    - 거리에 따라 추가적인 거리 보정을 적용하여 현실적인 거리를 반영합니다.
    """
    if in_seoul1 or in_seoul2:  # 서울 내부일 경우
        if distance_km < 20:
            speed_kmh = 30  # 속도: 30km/h
        elif distance_km < 30:
            speed_kmh = 35  # 속도: 35km/h
        elif distance_km < 40:
            speed_kmh = 40  # 속도: 40km/h
        else:
            speed_kmh = 70  # 속도: 70km/h
    else:  # 서울 외부일 경우
        if distance_km < 5:
            speed_kmh = 35  # 속도: 35km/h
        elif distance_km < 20:
            speed_kmh = 45  # 속도: 45km/h
        elif distance_km < 50:
            speed_kmh = 55  # 속도: 55km/h
        else:
            speed_kmh = 70  # 속도: 70km/h

    ''' 
    - 거리와 조건에 따라 일정한 거리 값을 추가로 보정합니다. 
    '''
    if 15 <= distance_km < 40:
        adjusted_distance_km = distance_km + 5  # 5km 추가 보정
    elif 40 <= distance_km < 80:
        adjusted_distance_km = distance_km + 10  # 10km 추가 보정
    elif distance_km < 100:
        adjusted_distance_km = distance_km + 15  # 15km 추가 보정
    elif distance_km >= 100:
        adjusted_distance_km = distance_km + 30  # 30km 추가 보정
    else:
        adjusted_distance_km = distance_km  # 거리 보정 없음

    return speed_kmh, adjusted_distance_km  # 계산된 속도(km/h)와 보정된 거리(km)를 반환


# 두 좌표 간의 이동 시간 및 조정된 거리 계산
def calculate_time_and_distance(from_point, to_point):
    """
    두 지점 간의 거리와 이동 시간을 계산합니다.
    Haversine 공식으로 거리 계산 후, 속도 및 거리 보정을 적용하여 이동 시간을 계산합니다.
    """
    # Haversine 공식을 사용해 직선 거리를 계산합니다.
    distance_km = haversine(from_point['lat'], from_point['lng'], to_point['lat'], to_point['lng'])
    # 서울 내부 여부와 거리를 기반으로 속도와 보정 거리를 계산합니다.
    speed_kmh, adjusted_distance_km = calculate_speed_and_adjusted_distance(
        distance_km, from_point['in_seoul'], to_point['in_seoul']
    )
    # 이동 시간 계산: 시간 = 거리 / 속도 * 60 (분 단위로 변환)
    travel_time_minutes = (adjusted_distance_km / speed_kmh) * 60
    return adjusted_distance_km, travel_time_minutes  # 보정 거리와 이동 시간을 반환

# 가중치 계산 함수
def calculate_weight(from_point, to_point):
    """
    두 좌표(from_point, to_point) 간의 가중치를 계산합니다.
    가중치는 거리와 이동 시간의 조합으로 결정됩니다.
    """
    # 거리 및 시간 계산
    adjusted_distance_km, travel_time_minutes = calculate_time_and_distance(from_point, to_point)
    
    # 가중치 계산: 거리(60%)와 시간(40%) 비율로 가중치를 조합합니다.
    weight = adjusted_distance_km * 0.6 + travel_time_minutes * 0.4

    # 디버깅 정보를 출력합니다.
    # 시작 지점, 도착 지점, 거리, 시간, 계산된 가중치를 확인할 수 있습니다.
    print(f"From: {from_point['name']} -> To: {to_point['name']}")
    print(f"  거리(km): {adjusted_distance_km:.2f}, 시간(분): {travel_time_minutes:.2f}, 가중치: {weight:.2f}")
    
    return weight  # 계산된 가중치를 반환

def create_weighted_distance_matrix(points):
    n = len(points)
    distance_matrix = np.zeros((n, n))
    distance_details = np.zeros((n, n, 2))  # 거리와 시간을 저장할 추가 행렬

    for i in range(n):
        for j in range(n):
            if i != j:
                weight = calculate_weight(points[i], points[j])
                distance, time = calculate_time_and_distance(points[i], points[j])
                distance_matrix[i][j] = weight
                distance_details[i][j] = [distance, time]  # [거리, 시간] 저장
            else:
                distance_matrix[i][j] = float('inf')
                distance_details[i][j] = [float('inf'), float('inf')]

        # 가중치 행렬 출력
    print("\n가중치 행렬:")
    print(distance_matrix)
    
    return distance_matrix, distance_details


# 각 구간의 이동 시간 계산
def calculate_segment_times(points, path):
    """
    최적 경로의 각 구간에 대한 이동 시간을 계산하여 반환합니다.
    이동 시간은 각 경로 구간의 시작 지점과 도착 지점 간의 이동 시간을 기준으로 계산됩니다.
    """
    segment_times = []  # 구간별 이동 시간을 저장할 리스트
    for i in range(len(path) - 1):  # 경로에서 각 구간을 순회
        from_point = points[path[i]]  # 시작 지점
        to_point = points[path[i + 1]]  # 도착 지점
        _, travel_time_minutes = calculate_time_and_distance(from_point, to_point)  # 이동 시간 계산

        
        if travel_time_minutes >= 60:  
            hours = int(travel_time_minutes // 60)  # 시간 부분
            minutes = int(travel_time_minutes % 60)  # 분 부분
            segment_times.append(f"{hours}시간 {minutes}분")  # 형식: 'X시간 Y분'
        else:  
            segment_times.append(f"{int(travel_time_minutes)}분")  # 형식: 'X분'
    return segment_times  # 각 구간의 이동 시간을 반환

# TSP 문제 해결
def solve_tsp(distances, points, distance_details):
    """
    TSP(Traveling Salesperson Problem)를 해결하여 최적 경로를 찾습니다.
    모든 가능한 경로를 탐색하여 가장 낮은 가중치를 가지는 경로를 선택합니다.
    """
    n = len(distances)  # 포인트의 개수
    best_path = None  # 최적 경로를 저장할 변수
    min_distance = float('inf')  # 최소 가중치 초기값 (무한대로 설정)
    min_first_leg_distance = float('inf')  # 최소 첫 번째 구간 거리 초기값 (무한대로 설정)
    best_segment_distances = []  # 최적 경로의 각 구간 거리 저장
    best_total_distance = 0  # 최적 경로의 총 거리 저장

    for perm in permutations(range(1, n)):  # 첫 번째 포인트는 항상 시작점(0번 노드)로 고정
        path = [0] + list(perm) + [0]  # 경로: 시작 -> 방문지들 -> 시작으로 돌아옴
        total_weight = 0
        total_distance = 0
        segment_distances = []

        for i in range(len(path) - 1):
            from_idx, to_idx = path[i], path[i + 1]
            weight = distances[from_idx][to_idx]  # 미리 계산된 가중치
            distance = distance_details[from_idx][to_idx][0]  # 미리 계산된 거리
            total_weight += weight
            total_distance += distance
            segment_distances.append({
                "start": points[from_idx]["name"],
                "end": points[to_idx]["name"],
                "distance_km": distance
            })

        # 첫 번째 구간 거리 계산 (출발지 → 첫 번째 목적지)
        first_leg_distance = distance_details[path[0]][path[1]][0]

        # 최적 경로 갱신 조건
        if total_weight < min_distance:
            min_distance = total_weight
            min_first_leg_distance = first_leg_distance
            best_path = path
            best_segment_distances = segment_distances
            best_total_distance = total_distance
        elif total_weight == min_distance:
            # 첫 번째 구간 거리 기준으로 선택
            if first_leg_distance < min_first_leg_distance:
                min_first_leg_distance = first_leg_distance
                best_path = path
                best_segment_distances = segment_distances
                best_total_distance = total_distance

    return best_path, min_distance, best_segment_distances, best_total_distance


# 경로를 따라 정렬된 좌표 반환
def get_ordered_coordinates(points, path):
    """
    최적 경로를 따라 정렬된 포인트 좌표를 반환합니다.
    """
    return [points[i] for i in path]  # 최적 경로 순서에 따라 포인트 좌표를 정렬하여 반환
