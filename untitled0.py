import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import sys  

# Modeli yükleyin
model_path = 'C:/Staj-projects/fire_smoke_project/fire-flame.pt'  # Model dosyanızın tam yolu
model = torch.load(model_path)
model.eval()
model.cuda()  # Eğer CUDA destekli bir GPU kullanıyorsanız

# Görüntü işleme dönüşümleri
transformer = transforms.Compose([
    transforms.Resize(225),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Video dosyasını açın
video_path = 'C:/Staj-projects/fire_smoke_project/fire.mp4'  # İşlemek istediğiniz video dosyasının yolu
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Video dosyası açılamadı.")
    sys.exit()  # Programı sonlandırmak için sys.exit() kullanın

# Video özelliklerini alın
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Çıktı video dosyasını oluşturun
output_path = 'output_video.mp4'  # Çıktı video dosyasının yolu
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = video.read()
    if not ret:
        break

    # OpenCV'den PIL'e dönüştür
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img.astype('uint8'))

    # Görüntü işleme ve model tahmini
    img_processed = transformer(img).unsqueeze(0)
    img_var = torch.autograd.Variable(img_processed, requires_grad=False).cuda()

    with torch.no_grad():
        logp = model(img_var)
        expp = torch.softmax(logp, dim=1)
        confidence, clas = expp.topk(1, dim=1)

    co = confidence.item() * 100
    class_no = str(clas.item())

    # Görüntü üzerinde etiketleme yapın
    label = ""
    color = (255, 255, 255)  # Varsayılan etiket rengi: Beyaz

    if class_no == '1':
        label = f"Neutral: {co:.2f}%"
        color = (255, 255, 255)  # Beyaz

    elif class_no == '2':
        label = f"Smoke: {co:.2f}%"
        color = (255, 0, 0)  # Mavi

    elif class_no == '0':
        label = f"Fire: {co:.2f}%"
        color = (0, 0, 255)  # Kırmızı

    # Etiketleme ve işaretleme
    frame = cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Çerçeveyi çıktı video dosyasına yazın
    out.write(frame)

    # Çerçeveyi ekranda göster
    cv2.imshow('Processed Video', frame)

    # 'q' tuşuna basıldığında çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video dosyalarını serbest bırakın
video.release()
out.release()
cv2.destroyAllWindows()

print("Video işleme tamamlandı.")
