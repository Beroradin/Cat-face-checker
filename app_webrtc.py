import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA (Mobile Friendly)
# ==============================================================================
st.set_page_config(
    page_title="Gato Detector 😺",
    page_icon="😺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS para esconder elementos desnecessários e melhorar mobile
st.markdown("""
<style>
    /* Esconder menu hamburger do Streamlit para app parecer mais nativo */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Área principal com menos padding em mobile */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Estilo dos status */
    .status-text {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .he-he { background-color: #dbf6e6; color: #2e7d32; border: 2px solid #2e7d32; }
    .sus { background-color: #ffebee; color: #c62828; border: 2px solid #c62828; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CONSTANTES E RECURSOS
# ==============================================================================

# URLs para imagens (Prioridade: Arquivo Local > URL Repositório)
import os

# Se o arquivo existe localmente (no Cloud ou PC), usa ele.
if os.path.exists("sus.png"):
    IMG_SUS = "sus.png"
else:
    # Fallback caso algo dê errado no upload
    IMG_SUS = "https://raw.githubusercontent.com/mathig/cat-face-detector/main/sus.png"

if os.path.exists("hehe.jpeg"):
    IMG_HEHE = "hehe.jpeg"
else:
    IMG_HEHE = "https://i.imgur.com/tI6M8fA.jpg"

# Configuração do MediaPipe (Carregado uma única vez)
mp_face_mesh = mp.solutions.face_mesh
face_mesh_detector = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Config STUN/TURN para WebRTC funcionar em 4G/5G
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==============================================================================
# 3. LÓGICA DE PROCESSAMENTO
# ==============================================================================

def detect_tongue(landmarks):
    """Calcula se a língua está pra fora baseado na abertura vertical da boca"""
    # Landmarks: 13 (Lábio Sup), 14 (Lábio Inf), 78 (Canto Esq), 308 (Canto Dir)
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
        self.threshold = 0.3 # Valor padrão
        self.last_val = 0.0

    def recv(self, frame):
        # Converter av.VideoFrame para ndarray
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Espelhar (Selfie mode)
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # 2. Processar MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh_detector.process(img_rgb)
        
        status = "SUS"
        color = (0, 0, 255) # Red
        val = 0.0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calcular abertura
                val = detect_tongue(face_landmarks.landmark)
                self.last_val = val
                
                # Desenhar pontos da boca (feedback visual rápido)
                # Lábios
                for idx in [13, 14, 78, 308]:
                    p = face_landmarks.landmark[idx]
                    cv2.circle(img, (int(p.x*w), int(p.y*h)), 3, (255, 255, 0), -1)
                
                # Lógica de decisão
                # Nota: Lemos o threshold definido na UI através de um truque ou fixo.
                # Como passar dados da UI para o loop de vídeo é complexo sem queue,
                # vamos usar um valor fixo seguro ou tentar ler de uma global (não ideal mas funciona simples)
                if val > 0.35: # Hardcoded safe value for mobile loop
                    status = "HEHE"
                    color = (0, 255, 0) # Green
                
                # HUD na tela
                # Barra de progresso visual
                bar_x, bar_y, bar_w, bar_h = 20, 50, 200, 20
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), -1)
                fill_w = int(min(val / 0.6, 1.0) * bar_w)
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
                cv2.putText(img, f"BOCA: {val:.2f}", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Texto Grande no centro inferior
                text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(img, status, (text_x, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==============================================================================
# 4. INTERFACE
# ==============================================================================

def main():
    st.title("😺 Gato Detector Mobile")
    st.info("💡 Para usar no celular: Use o Chrome ou Safari e permita o acesso à câmera.")

    # Container centralizado
    with st.container():
        # WebRTC Streamer
        # key: importante para manter estado
        # mode: SENDRECV para enviar vídeo e receber processado
        # media_stream_constraints: facingMode='user' força câmera frontal no mobile
        webrtc_streamer(
            key="cat-face-mobile",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
            video_processor_factory=VideoProcessor,
            async_processing=True,
        )
    
    st.write("---")
    
    # Imagens de Referência (estáticas, abaixo do vídeo)
    col1, col2 = st.columns(2)
    with col1:
        st.image(IMG_SUS, caption="Modo SUS (Boca Fechada)", use_container_width=True)
    with col2:
        st.image(IMG_HEHE, caption="Modo HEHE (Língua Fora)", use_container_width=True)

    with st.expander("🛠️ Dicas de Solução de Problemas"):
        st.write("""
        1. **iOS (iPhone)**: Use o Safari. O Chrome no iOS às vezes bloqueia WebRTC.
        2. **Android**: Chrome ou Firefox funcionam bem.
        3. **Tela Preta**: Verifique se deu permissão de câmera. Recarregue a página.
        4. **Lentidão**: WebRTC depende da sua conexão de internet (upload). Use Wi-Fi se possível.
        """)

if __name__ == "__main__":
    main()
