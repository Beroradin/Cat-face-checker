import streamlit as st
import cv2
import av
import mediapipe as mp
import queue
import os
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ==============================================================================
# 0. SUPRIME LOGS DO TENSORFLOW
# ==============================================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Detector de Gatinhos 😺",
    page_icon="😺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. RECURSOS
# ==============================================================================
IMG_SUS = (
    "sus.png"
    if os.path.exists("sus.png")
    else "https://raw.githubusercontent.com/mathig/cat-face-detector/main/sus.png"
)
IMG_HEHE = (
    "hehe.jpeg"
    if os.path.exists("hehe.jpeg")
    else "https://i.imgur.com/tI6M8fA.jpg"
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Landmarks de interesse para a detecção de língua
_TONGUE_LANDMARKS = [13, 14, 78, 308]


# ==============================================================================
# 3. LÓGICA (PROCESSADOR DE VÍDEO)
# ==============================================================================
def _mouth_ratio(landmarks) -> float:
    """Calcula a razão vertical/horizontal da boca."""
    up, down = landmarks[13], landmarks[14]
    left, right = landmarks[78], landmarks[308]
    h_dist = abs(left.x - right.x)
    if h_dist < 1e-6:
        return 0.0
    return abs(up.y - down.y) / h_dist


class VideoProcessor:
    THRESHOLD = 0.35
    # Envia no máximo 1 status a cada N frames para não inundar a fila
    SEND_EVERY_N_FRAMES = 3

    def __init__(self):
        # ✅ FaceMesh criado DENTRO da thread do processador (thread-safe)
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # maxsize evita memory leak se a UI não consumir rápido
        self.result_queue: queue.Queue[str] = queue.Queue(maxsize=5)
        self._frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        results = self._face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        status = "SUS"
        color = (0, 0, 255)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            ratio = _mouth_ratio(lm)

            # Desenhar landmarks de referência
            for idx in _TONGUE_LANDMARKS:
                pt = lm[idx]
                cv2.circle(img, (int(pt.x * w), int(pt.y * h)), 3, (255, 255, 0), -1)

            if ratio > self.THRESHOLD:
                status = "HEHE"
                color = (0, 255, 0)

        # HUD no vídeo
        text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
        cv2.putText(
            img,
            status,
            ((w - text_size[0]) // 2, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            color,
            4,
        )

        # ✅ Debounce: só envia a cada N frames
        self._frame_count += 1
        if self._frame_count >= self.SEND_EVERY_N_FRAMES:
            self._frame_count = 0
            # put_nowait + maxsize: descarta se a fila estiver cheia (não bloqueia)
            try:
                self.result_queue.put_nowait(status)
            except queue.Full:
                # Descarta o mais antigo e coloca o novo
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                self.result_queue.put_nowait(status)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==============================================================================
# 4. INTERFACE PRINCIPAL
# ==============================================================================
def main():
    st.title("😺 Detector de Gatinhos")

    col_cam, col_img = st.columns([2, 1])

    with col_cam:
        ctx = webrtc_streamer(
            key="cat-face",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "facingMode": "user",
                    "width": {"ideal": 480},
                    "height": {"ideal": 640},
                },
                "audio": False,
            },
            video_processor_factory=VideoProcessor,
            async_processing=True,
        )

    with col_img:
        st.markdown("### Estado Atual:")
        image_placeholder = st.empty()
        image_placeholder.image(IMG_SUS, caption="Aguardando...", use_container_width=True)

    # ✅ Loop com condição de saída limpa (sem while True infinito)
    if ctx.state.playing and ctx.video_processor:
        processor = ctx.video_processor
        last_status = None

        while ctx.state.playing:
            try:
                # Drena a fila inteira e pega só o último status (mais recente)
                status = None
                while True:
                    try:
                        status = processor.result_queue.get_nowait()
                    except queue.Empty:
                        break

                # Só atualiza a UI se o status mudou
                if status is not None and status != last_status:
                    last_status = status
                    if status == "HEHE":
                        image_placeholder.image(
                            IMG_HEHE, caption="MUITO HEHE!", use_container_width=True
                        )
                    else:
                        image_placeholder.image(
                            IMG_SUS, caption="Suspeito...", use_container_width=True
                        )

            except Exception:
                break

            time.sleep(0.1)  # 10 FPS para a UI é mais que suficiente


if __name__ == "__main__":
    main()
