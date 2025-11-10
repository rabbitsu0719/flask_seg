import cv2
import numpy as np

# 1. 이미지 로드
image_path = 'car2.png'
image = cv2.imread(image_path)

# 이미지가 제대로 로드되었는지 확인
if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

# 2. 그레이스케일 변환
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. 이진화 (역이진 + Otsu)
_, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 4. 윤곽선 찾기 (외곽선만)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # 5. 가장 큰 윤곽선 선택
    largest_contour = max(contours, key=cv2.contourArea)

    # 6. 빈 마스크 생성 (원본 이미지 크기, 단일 채널)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 7. 가장 큰 윤곽선을 흰색(255)으로 마스크에 그림
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=-1)

    # 8. 원본 이미지에 마스크 적용하여 객체만 추출
    segmented_object = cv2.bitwise_and(image, image, mask=mask)

    # 9. 객체 배경을 투명하게
    b, g, r =cv2.bitwise_and(image, image, mask=mask)
    alpha = mask
    rgba_image = cv2.merge((b, g, r, alpha))

    # 10. 도로 이미지 불러오기
    road_path = 'road.png'
    road_image = cv2.imread(road_path, cv2.IMREAD_COLOR)

    # 자동차 이미지 크기에 맞게 도로 이미지 크기 조정
     if road_image.shape[:2] != image.shape[:2]:
       road_image = cv2.resize(road_image, (image.shape[1], image.shape[0]))

    # 도로 이미지를 BGRA로 변환하고 알파 채널을 불투명으로 설정
    road_bgra = cv2.cvtColor(road_image, cv2.COLOR_BGR2BGRA)
    road_bgra[:, :, 3] = 255  # 알파 채널 255 (불투명)

    # 자동차 알파 채널을 기준으로 합성: 자동차는 불투명, 나머지는 도로 배경
    final_image = np.where(rgba_image[:, :, 3:] == 255, rgba_image, road_bgra)   

    # 결과 출력
    cv2.imshow('Original Image', image)
    cv2.imshow('Gray Image', gray_image)
    cv2.imshow('Thresholded Image', thresh)
    cv2.imshow('Mask', mask)
    cv2.imshow('Segmented Object', segmented_object)
    cv2.imshow('Segmented Object with Transparency', rgba_image)
    cv2.imshow('Car on the road', final_image)
    cv2.imshow('car_on_the_road.png', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("윤곽선이 감지되지 않았습니다.")