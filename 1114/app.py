from flask import Flask, request, send_file, jsonify, render_template
from PIL import Image
import io, threading
import numpy as np
import cv2

# SAM: 두 가지 방식 중 하나 사용
# 1) Predictor (네가 받은 코드 스타일)
from ultralytics.models.sam import Predictor as SAMPredictor
# 2) 간단 API: from ultralytics import SAM; model = SAM("sam_b.pt"); results = model.predict(img)

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route("/")
def home():
    return render_template("index.html")


class State:
    def __init__(self):
        self.image = None          # PIL RGBA
        self.masks = []            # list[np.ndarray bool(H,W)]
        self.predictor = None      # SAMPredictor
        self.lock = threading.Lock()

state = State()

# ---------- 유틸 ----------
def ensure_predictor():
    if state.predictor is None:
        # sam_b.pt는 처음에 자동 다운로드됨 (네트워크 필요)
        state.predictor = SAMPredictor(overrides=dict(model="sam_b.pt"))

def np_bool(mask_np):
    return mask_np.astype(bool)

def dilate_mask(mask_np, kernel_size=7):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask_np.astype(np.uint8), kernel, iterations=1).astype(bool)

def smooth_mask(mask_np, kernel_size=5):
    # float32로 블러 → 임계값
    blurred = cv2.GaussianBlur(mask_np.astype(np.float32), (kernel_size, kernel_size), 0)
    return (blurred > 0.5).astype(bool)

def overlay_preview_rgba(base_rgba: Image.Image, mask_np: np.ndarray, color=(255,0,0,120)):
    img = base_rgba.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = Image.new("RGBA", img.size, color)
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
    overlay.paste(draw, mask=mask_img)
    return Image.alpha_composite(img, overlay)

# ---------- 라우트 ----------
@app.route("/upload", methods=["POST"])
def upload_image():
    # file 필드에서 이미지 받기
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "image 파일이 필요합니다."}), 400

    try:
        up = Image.open(file.stream).convert("RGBA")
    except Exception as e:
        return jsonify({"error": f"이미지 로드 실패: {e}"}), 415

    with state.lock:
        state.image = up
        state.masks.clear()
        ensure_predictor()
        # Predictor는 RGB 배열을 기준으로 set_image 필요
        img_np = np.array(up.convert("RGB"))
        state.predictor.set_image(img_np)

    return jsonify({"message": "이미지 업로드 완료"}), 200

@app.route("/click", methods=["POST"])
def click_point():
    """
    body: {"x": int, "y": int}
    """
    data = request.get_json(silent=True) or {}
    if "x" not in data or "y" not in data:
        return jsonify({"error": "x,y 좌표가 필요합니다."}), 400

    with state.lock:
        if state.image is None or state.predictor is None:
            return jsonify({"error": "먼저 /upload 로 이미지를 올려주세요."}), 400

        x, y = int(data["x"]), int(data["y"])
        points = np.array([[x, y]])
        labels = np.array([1], dtype=np.int32)  # 1 = foreground

        try:
            results = state.predictor(points=points, point_labels=labels, multimask_output=False)
        except Exception as e:
            return jsonify({"error": f"SAM 추론 오류: {e}"}), 500

        if not results or len(results) == 0 or results[0].masks is None:
            return jsonify({"error": "마스크 생성 실패"}), 400

        # (1, H, W) 중 첫 번째
        mask_np = results[0].masks.data.cpu().numpy()[0] > 0.5
        # 음식 사진 경계 보정
        mask_np = dilate_mask(mask_np, kernel_size=7)
        mask_np = smooth_mask(mask_np, kernel_size=5)

        state.masks.append(mask_np)

        # 프리뷰는 합성 결과를 즉시 반환(선택)
        preview = overlay_preview_rgba(state.image, mask_np)
        buf = io.BytesIO()
        preview.save(buf, "PNG"); buf.seek(0)

    return send_file(buf, mimetype="image/png")

@app.route("/delete", methods=["POST"])
def delete_mask():
    """
    누적된 마스크를 합쳐 배경을 투명 처리
    """
    with state.lock:
        if state.image is None:
            return jsonify({"error": "이미지가 없습니다."}), 400

        if not state.masks:
            return jsonify({"error": "적용할 마스크가 없습니다. /click 으로 먼저 선택하세요."}), 400

        h, w = state.image.height, state.image.width
        combined = np.zeros((h, w), dtype=bool)
        for m in state.masks:
            combined = np.logical_or(combined, m)

        # 최종 부드럽게
        combined = dilate_mask(combined, kernel_size=5)
        combined = smooth_mask(combined, kernel_size=5)

        img_np = np.array(state.image)  # RGBA
        # 전경만 남기고 배경을 투명 처리하려면 "반전"이 아님에 주의.
        # 요청 의도가 "선택영역 삭제"라면 아래 한 줄을 반전하세요:
        # combined = np.logical_not(combined)
        # 여기선 "선택영역만 남기기(배경 제거)"로 구현:
        alpha = img_np[:, :, 3]
        alpha[~combined] = 0
        img_np[:, :, 3] = cv2.GaussianBlur(alpha, (5,5), 0)  # 경계 soft
        state.image = Image.fromarray(img_np)
        state.masks.clear()

    return jsonify({"message": "배경 제거 적용 완료"}), 200

@app.route("/save", methods=["GET"])
def save_image():
    if state.image is None:
        return jsonify({"error": "이미지가 없습니다."}), 400
    path = "saved_output.png"
    with state.lock:
        # 마지막으로 알파를 한 번 더 부드럽게
        np_img = np.array(state.image)
        np_img[:,:,3] = cv2.GaussianBlur(np_img[:,:,3], (5,5), 0)
        Image.fromarray(np_img).save(path)
    return jsonify({"message": path}), 200

@app.route("/download", methods=["GET"])
def down_image():
    if state.image is None:
        return jsonify({"error": "이미지가 없습니다."}), 400
    with state.lock:
        np_img = np.array(state.image)
        np_img[:,:,3] = cv2.GaussianBlur(np_img[:,:,3], (5,5), 0)
        out = Image.fromarray(np_img)
        buf = io.BytesIO()
        out.save(buf, "PNG"); buf.seek(0)
    return send_file(buf, mimetype="image/png", as_attachment=True, download_name="edited_image.png")

@app.route("/preview", methods=["GET"])
def preview_image():
    # 현재 이미지를 확인하기 위한 엔드포인트(브라우저에서 열어보기)
    if state.image is None:
        return jsonify({"error": "이미지가 없습니다."}), 400
    with state.lock:
        buf = io.BytesIO()
        state.image.save(buf, "PNG"); buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
