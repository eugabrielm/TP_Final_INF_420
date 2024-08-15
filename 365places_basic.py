import os
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
import imageio.v2 as imageio
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import numpy as np

# Defina a arquitetura do modelo
arch = 'alexnet'

# Carregue os pesos pré-treinados
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# Carregue o transformador de imagens
centre_crop = trn.Compose([
    trn.Resize((256, 256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carregue os rótulos de classe
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

# Diretório de entrada e de saída
image_dir = r'/mnt/c/Users/Gabriel/Desktop/UFV/5 periodo/IA/TP_Final_INF_420/images'
output_root_dir = r'/mnt/c/Users/Gabriel/Desktop/UFV/5 periodo/IA/TP_Final_INF_420/output'

# Crie o diretório de saída, caso não exista
os.makedirs(output_root_dir, exist_ok=True)

# Defina a camada alvo para Grad-CAM
if arch.startswith("resnet"):
    target_layer = model.layer4[-1]
elif arch == "alexnet":
    target_layer = model.features[-1]
else:
    raise ValueError("Arquitetura não suportada")

# Iterar sobre cada imagem no diretório
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    try:
        # Processamento de imagens HDR
        if img_path.lower().endswith('.hdr'):
            hdr_image = imageio.imread(img_path)
            temp_jpg_path = img_path.replace('.hdr', '.jpg')
            imageio.imwrite(temp_jpg_path, hdr_image)
            img_path = temp_jpg_path

        img = Image.open(img_path)
        input_img = V(centre_crop(img).unsqueeze(0))


        # Fazer a predição
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        # Criar diretório para armazenar as imagens com Grad-CAM
        output_dir = os.path.join(output_root_dir, f"{img_name.split('.')[0]}_{arch}")
        os.makedirs(output_dir, exist_ok=True)

        # Se a imagem foi convertida, salve o JPG na pasta de saída
        
        temp_jpg_path = os.path.join(output_dir, f"{img_name.split('.')[0]}.jpg")
        imageio.imwrite(temp_jpg_path, hdr_image)
        img_path = temp_jpg_path  # Atualizar img_path para a imagem JPG convertida

        for i in range(0, 5):
            class_name = classes[idx[i]]
            class_name_safe = class_name.replace('/', '_')  # Substituir barras por sublinhados
            print('{:.3f} -> {}'.format(probs[i], class_name))

            # Configurar e rodar o Grad-CAM para cada classe prevista
            cam = GradCAM(model=model, target_layers=[target_layer])
            grayscale_cam = cam(input_tensor=input_img, targets=[ClassifierOutputTarget(idx[i].item())])

            # Redimensionar o cam para corresponder ao tamanho original da imagem
            grayscale_cam_resized = cv2.resize(grayscale_cam[0], (img.width, img.height))

            # Normalizar a imagem de entrada para o intervalo [0, 1]
            input_img_np = np.array(img) / 255.0  # Usar a imagem original em vez de `input_img`

            # Aplicar o CAM redimensionado à imagem original
            visualization = show_cam_on_image(input_img_np, grayscale_cam_resized, use_rgb=True)

            # Salvar a imagem resultante com Grad-CAM
            result_img_path = os.path.join(output_dir, f"{img_name.split('.')[0]}_{class_name_safe}.jpg")
            cv2.imwrite(result_img_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    

    except Exception as e:
        print(f"Error processing {img_name}: {e}")
