from flask import Flask, request, jsonify
from flask_cors import CORS
from algorithms import create_weighted_distance_matrix, solve_tsp, calculate_segment_times, get_ordered_coordinates, classify_points
import requests

# Flask 애플리케이션 생성
app = Flask(__name__)

# Flask 애플리케이션에 CORS 설정 적용
CORS(app)

# Kakao Mobility API 키 설정
API_KEY = "d53b4ecaa23c32b09206f7c531b46a55"

# 경로 계산 API 엔드포인트 정의
@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.json
        origin = data.get('start')  # 출발지 정보
        destinations = data.get('destinations', [])  # 목적지 리스트 정보 (기본값: 빈 리스트)

        if not origin or not destinations:
            return jsonify({
                "status": "error",
                "message": "출발지와 목적지가 필요합니다."
            }), 400

        # 출발지와 목적지 정보를 합쳐 하나의 리스트(points)로 만듭니다.
        points = [origin] + destinations

        # 각 포인트에 'in_seoul' 여부 추가 (서울 내외부 여부 확인)
        points = classify_points(points)

        # 가중치 행렬과 거리/시간 행렬 생성
        weight_matrix, distance_details = create_weighted_distance_matrix(points)

        # TSP 문제 해결
        best_path, min_weight, segment_distances, total_distance = solve_tsp(weight_matrix, points, distance_details)

        # 최적 경로에 따라 정렬된 지점 정보를 가져옵니다.
        ordered_points = get_ordered_coordinates(points, best_path)

        # Kakao Mobility API 요청을 위한 데이터 포맷 변환
        waypoints = [{"name": point['name'], "x": point['lng'], "y": point['lat']} for point in ordered_points]

        # Kakao API 요청에 필요한 데이터 구성
        payload = {
            "origin": {"x": ordered_points[0]['lng'], "y": ordered_points[0]['lat']},
            "destination": {"x": ordered_points[-1]['lng'], "y": ordered_points[-1]['lat']},
            "waypoints": waypoints[1:-1],
            "priority": "RECOMMEND"
        }

        # Kakao API 요청 헤더 설정
        headers = {
            "Authorization": f"KakaoAK {API_KEY}",
            "Content-Type": "application/json"
        }

        # Kakao Mobility API에 POST 요청을 보냅니다.
        response = requests.post(
            "https://apis-navi.kakaomobility.com/v1/waypoints/directions",
            json=payload,
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()
            routes = result.get('routes', [])

            if routes:
                vertexes = []
                for section in routes[0].get('sections', []):
                    for road in section.get('roads', []):
                        vertexes += road.get('vertexes', [])

                # 최종 응답 데이터 JSON 형태로 반환
                return jsonify({
                    "status": "success",
                    "path": ordered_points,
                    "segment_times": calculate_segment_times(points, best_path),
                    "segment_distances": segment_distances,
                    "total_distance": total_distance,
                    "total_weight": min_weight,
                    "route": [
                        {"lat": vertexes[i + 1], "lng": vertexes[i]}
                        for i in range(0, len(vertexes), 2)
                    ]
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "경로를 찾을 수 없습니다."
                })
        else:
            return jsonify({
                "status": "error",
                "message": response.text
            }), response.status_code

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Flask 애플리케이션 실행
if __name__ == '__main__':
    app.run(debug=True)
