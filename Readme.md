# EMOTIVA - Reconocimiento de Emociones Faciales en el Aula

**Aplicaciones de Data Science - Trabajo Final**
**Carrera de Ciencias de la Computación | 1ACC0219**
**Profesor:** Richard Fernando Fernández Vásquez

---

## 🚀 Objetivo del Trabajo

Desarrollar una solución basada en visión computacional y aprendizaje profundo capaz de detectar emociones en tiempo real dentro de un aula educativa. El sistema, denominado **EMOTIVA**, analiza expresiones faciales a partir de imágenes captadas por cámaras, y presenta un resumen emocional agregado para el docente, permitiéndole adaptar sus estrategias pedagógicas en base al estado emocional de sus estudiantes.

---

## 👥 Alumnos Participantes

- Jimena Alexsandra Quintana Noa - U20201F576  
- Freddy Alejandro Cuadros Contreras - U20221C488  
- Joaquín Sebastián Ruiz Ramírez - U20201F678

---

## 🔍 Descripción del Dataset

El proyecto hace uso del conjunto de datos **FER2013 (Facial Expression Recognition 2013)**, disponible en Kaggle. Este dataset contiene **35,887 imágenes** de rostros en escala de grises de 48x48 px clasificadas en 7 emociones:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

Las imágenes fueron recolectadas mediante búsqueda web automatizada y luego etiquetadas manualmente. El dataset está dividido en particiones de entrenamiento y prueba, y fue procesado utilizando `torchvision.datasets.ImageFolder` para adaptarse al modelo de clasificación de emociones.

Puedes ver más detalles en el archivo `EDA_APDA.ipynb`.

---

## 🔬 Conclusiones del Proyecto

- **Precisión alcanzada:** El modelo basado en EfficientNet-B2 alcanzó una precisión global del **72%**, destacando en emociones como *happy* y *surprise*, con dificultades en *fear* y *sad*.
- **Viabilidad real:** EMOTIVA demostró ser viable como sistema de retroalimentación emocional colectiva sin comprometer la privacidad individual.
- **Aplicabilidad:** Aporta un enfoque novedoso en la mejora del clima emocional en aulas, fortaleciendo la comunicación docente-estudiante.
- **Ética:** Se prioriza la anonimización, el consentimiento informado y la no conservación de imágenes faciales.
- **Futuro:** Se recomienda incorporar datasets más realistas (AffectNet, RAF-DB), capas densas adicionales, regularización agresiva, y técnicas de aumentación de datos.

---

## 🔗 Enlaces Relevantes

- [FER2013 Dataset en Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- [Colab del modelo EMOTIVA](https://colab.research.google.com/drive/...) *(Reemplazar con enlace real)*
- [Presentación EMOTIVA](https://docs.google.com/presentation/d/...) *(Reemplazar con enlace real)*

---

## ✉️ Licencia

Este proyecto se encuentra bajo la licencia **MIT**. Puedes reutilizar, modificar o adaptar el código con fines educativos o experimentales, brindando crédito a sus autores. No se autoriza el uso comercial sin previa autorización.

---

Para dudas o contacto, puedes escribirnos a cualquiera de nuestros correos institucionales.

---

*EMOTIVA - 2025. Una iniciativa para entender el aprendizaje desde las emociones.*
