import streamlit as st
import cv2
import av
import mediapipe as mp
import queue
import os
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ==============================================================================
# 0. CORREÇÃO DE LOGS (Remove avisos chatos do TensorFlow)
# ==============================================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Gato Detector 😺",
    page_icon="😺",
    layout="wide", # Layout wide facilita colocar coisas lado a lado
    initial_sidebar_state="collapsed"
)

# CSS Mobile Friendly
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. RECURSOS
# ==============================================================================

# URLs/Arquivos
IMG_SUS = "sus.png" if os.path.exists("sus.png") else "https://raw.githubusercontent.com/mathig/cat-face-detector/main/sus.png"
IMG_HEHE = "hehe.jpeg" if os.path.exists("hehe.jpeg") else "https://i.imgur.com/tI6M8fA.jpg"

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh_detector = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==============================================================================
# 3. LÓGICA (PROCESSADOR DE VÍDEO)
# ==============================================================================

def detect_tongue(landmarks):
    up = landmarks[13]
    down = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]
    v_dist = abs(up.y - down.y)
    h_dist = abs(left.x - right.x)
    if h_dist == 0: return 0.0
    return v_dist / h_dist

class VideoProcessor:
    def __init__(self):
        # Fila para enviar dados para a UI principal
        self.result_queue = queue.Queue()
        self.last_status = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh_detector.process(img_rgb)
        
        status = "SUS"
        color = (0, 0, 255)
        val = 0.0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                val = detect_tongue(face_landmarks.landmark)
                
                # Desenhar pontos
                for idx in [13, 14, 78, 308]:
                    p = face_landmarks.landmark[idx]
                    cv2.circle(img, (int(p.x*w), int(p.y*h)), 3, (255, 255, 0), -1)
                
                # Lógica HEHE vs SUS
                if val > 0.35:
                    status = "HEHE"
                    color = (0, 255, 0)
                
                # Enviar para a fila APENAS se mudou (para não travar a UI)
                # ou enviar sempre e a UI que se vire (aqui enviaremos sempre que detectar)
                self.result_queue.put(status)

                # HUD Opcional (Texto na tela do vídeo)
                text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(img, status, (text_x, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

        else:
            # Se não achou rosto, manda SUS por padrão
            self.result_queue.put("SUS")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==============================================================================
# 4. INTERFACE PRINCIPAL
# ==============================================================================

def main():
    st.title("😺 Gato Detector")

    # Layout de Colunas: [Vídeo (maior)] | [Imagem Reativa (menor)]
    col_cam, col_img = st.columns([2, 1])

    with col_cam:
        ctx = webrtc_streamer(
            key="cat-face-mobile",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
            video_processor_factory=VideoProcessor,
            async_processing=True,
        )

    # Placeholder para a imagem na coluna da direita
    with col_img:
        st.write("### Estado Atual:")
        image_placeholder = st.empty()
        # Imagem inicial padrão
        image_placeholder.image(IMG_SUS, caption="Aguardando...", width="stretch")

    # Loop para atualizar a UI fora da thread do vídeo
    if ctx.state.playing:
        while True:
            # Tenta pegar o status do processador
            if ctx.video_processor:
                try:
                    # Pega o último status da fila (sem bloquear muito tempo)
                    status = ctx.video_processor.result_queue.get(timeout=1.0)
                    
                    if status == "HEHE":
                        # Substitui 'use_container_width' por 'width="stretch"' conforme o erro pediu
                        image_placeholder.image(IMG_HEHE, caption="MUITO HEHE!", width="stretch")
                    else:
                        image_placeholder.image(IMG_SUS, caption="Suspeito...", width="stretch")
                    
                except queue.Empty:
                    pass
            time.sleep(0.05) # Pequena pausa para não explodir a CPU

if __name__ == "__main__":
    main()
