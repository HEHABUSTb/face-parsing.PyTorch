
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import cv2

# 1. Инициализируем FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # or 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=0)

# 2. Загружаем и анализируем два изображения
img1 = cv2.imread(r"D:\AI\ArcFace\Cropped_100\1138_ashley_barnes_Front_Default_face2.png")
img2 = cv2.imread(r"D:\AI\ArcFace\Cropped_100\1138_ashley_barnes_Front_Default_face1.png")

face1 = app.get(img1)[0]  # Получаем первое лицо
face2 = app.get(img2)[0]

# 3. Сравниваем эмбеддинги
emb1 = face1.embedding
emb2 = face2.embedding

# 4. Расчет косинусного расстояния
distance = cosine(emb1, emb2)

print(f"Cosine distance: {distance:.4f}")
if distance < 0.4:
    print("👬 Похожи (вероятно один человек)")
else:
    print("❌ Не похожи (разные)")
