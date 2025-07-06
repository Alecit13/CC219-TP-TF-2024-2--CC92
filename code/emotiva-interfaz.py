import streamlit as st
import torch
from torchvision import transforms, models
from facenet_pytorch import MTCNN
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# ========== CONFIGURACIÓN GENERAL ==========
st.set_page_config(
    page_title="EMOTIVA - Dashboard Emocional",
    layout="wide",
    page_icon="📊"
)

st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .block-container {padding-top: 1rem;}
    .css-1d391kg {padding-top: 2rem;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ========== MAPEO EMOCIONES ==========
idx_to_emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ========== MODELO ==========
@st.cache_resource
def load_model():
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 7)
    model.load_state_dict(torch.load("best_efficientnet_emotion.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
mtcnn = MTCNN(keep_all=True, device="cpu")

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ========== SIDEBAR ==========
st.sidebar.image("https://i.ibb.co/wRd5cPK/emotiva-logo.png", width=180)
st.sidebar.title("📤 Subir Imagen")
image_file = st.sidebar.file_uploader("Selecciona una imagen del aula", type=["jpg", "jpeg", "png"])

# ========== FUNCIONES DE PROCESAMIENTO ==========
def predict_emotions(image):
    boxes, _ = mtcnn.detect(image)
    emotion_counts = {emo: 0 for emo in idx_to_emotion}
    cropped_faces = []

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = image.crop((x1, y1, x2, y2))
            input_tensor = transform(face).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                pred = output.argmax().item()
                emotion = idx_to_emotion[pred]
                emotion_counts[emotion] += 1
                cropped_faces.append((box, emotion))
    return cropped_faces, emotion_counts

def generar_recomendaciones(emociones):
    total = sum(emociones.values())
    if total == 0:
        return ["No se detectaron rostros en la imagen."]

    recomendaciones = []
    proporciones = {emo: count / total for emo, count in emociones.items()}

    if proporciones['Sad'] > 0.25 or proporciones['Angry'] > 0.2:
        recomendaciones.append("⚠️ Se detectó una alta proporción de emociones negativas. Considere aplicar una pausa activa o una actividad lúdica para mejorar el ambiente emocional.")
    if proporciones['Happy'] > 0.4:
        recomendaciones.append("✅ El aula muestra una buena disposición emocional. Puedes aprovechar este momento para introducir contenido clave.")
    if proporciones['Neutral'] > 0.5:
        recomendaciones.append("ℹ️ Hay predominancia de emociones neutrales. Es buen momento para usar dinámicas o preguntas interactivas para estimular el interés.")
    if proporciones['Fear'] > 0.2 or proporciones['Disgust'] > 0.2:
        recomendaciones.append("🚨 Emociones como miedo o desagrado podrían indicar incomodidad o estrés. Evalúe si el entorno es acogedor y seguro.")
    if not recomendaciones:
        recomendaciones.append("✔️ Las emociones están distribuidas de forma equilibrada. Continúa observando y adaptando según la evolución del grupo.")

    return recomendaciones

# ========== ESTADÍSTICAS ADICIONALES ==========
def estadisticas_adicionales(emo_counts):
    total = sum(emo_counts.values())
    emociones_positivas = ['Happy', 'Surprise']
    emociones_negativas = ['Angry', 'Sad', 'Fear', 'Disgust']
    positivas = sum(emo_counts[e] for e in emociones_positivas)
    negativas = sum(emo_counts[e] for e in emociones_negativas)
    diversidad = len([e for e in emo_counts.values() if e > 0])
    max_emocion = max(emo_counts, key=emo_counts.get)
    homogeneidad = emo_counts[max_emocion] / total if total > 0 else 0
    return {
        "positividad": round((positivas / total) * 100, 1) if total > 0 else 0,
        "diversidad": diversidad,
        "homogeneidad": round(homogeneidad * 100, 1)
    }

# ========== SECCIÓN PRINCIPAL ==========
st.title("📊 EMOTIVA Dashboard Emocional")
st.markdown("Monitoreo grupal de emociones en el aula a partir de una imagen.")

if image_file is not None:
    img = Image.open(image_file).convert("RGB")
    faces, emo_counts = predict_emotions(img)

    # Primer grid: 2 columnas x 2 filas (4 espacios)
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.subheader("🧠 Imagen Analizada")
            img_draw = img.copy()
            draw = ImageDraw.Draw(img_draw)
            for box, emotion in faces:
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
                draw.text((x1, y1 - 10), emotion, fill="black")
            st.image(img_draw, caption="Resultado del análisis", use_container_width=True)

    with col2:
        with st.container():
            st.subheader("📈 Distribución de emociones")
            labels = list(emo_counts.keys())
            values = list(emo_counts.values())
            fig, ax = plt.subplots()
            ax.barh(labels, values, color="skyblue")
            ax.set_xlabel("Cantidad")
            st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        with st.container():
            st.subheader("📊 Estadísticas Generales")
            total = sum(emo_counts.values())
            st.markdown(f"**Total de rostros detectados:** {total}")
            if total > 0:
                max_emotion = max(emo_counts, key=emo_counts.get)
                st.markdown(f"**Emoción predominante:** {max_emotion}")
                stats = estadisticas_adicionales(emo_counts)
                st.markdown(f"**Índice de positividad:** {stats['positividad']}%")
                st.markdown(f"**Diversidad emocional:** {stats['diversidad']} emociones distintas")
                st.markdown(f"**Homogeneidad emocional:** {stats['homogeneidad']}%")

    with col4:
        with st.container():
            st.subheader("🟢 Gráfico Circular")
            fig2, ax2 = plt.subplots()
            ax2.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')
            st.pyplot(fig2)

    # Segunda fila, fila completa con recomendaciones basadas en estadísticas
    st.markdown("---")
    st.subheader("📌 Recomendaciones pedagógicas")
    for rec in generar_recomendaciones(emo_counts):
        st.markdown(f"- {rec}")

else:
    st.info("🔍 Sube una imagen desde el panel lateral para comenzar el análisis.")
