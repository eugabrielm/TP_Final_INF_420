import os
import csv
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
import argparse

# Definir os argumentos de linha de comando
parser = argparse.ArgumentParser(description='Processar parâmetros de entrada e saída.')
parser.add_argument('--input_dir', type=str, required=True, help='Diretório de entrada das imagens')
parser.add_argument('--output_dir', type=str, required=True, help='Diretório de saída para os resultados')
parser.add_argument('--arch', type=str, required=True, help='Arquitetura do modelo (ex: alexnet, resnet18)')
args = parser.parse_args()


# Defina a arquitetura da rede neural(places365 permite resnet18, resnet50, alexnet, densenet161)
arch = args.arch

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


image_dir = args.input_dir
output_root_dir = os.path.join(args.output_dir, args.arch)
csv_output_path = os.path.join(args.output_dir, f'results_{args.arch}_top5.csv')

# Crie o diretório de saída, caso não exista
os.makedirs(output_root_dir, exist_ok=True)


if arch.startswith("resnet"):
    target_layer = model.layer4[-1]
elif arch == "alexnet":
    target_layer = model.features[-1]
elif arch == "densenet161":
    target_layer = model.features[-1]
else:
    raise ValueError("Arquitetura não suportada")

with open(csv_output_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Image Name', 'Rank', 'Predicted Label', 'Confidence'])

  
    for label_dir in os.listdir(image_dir):
        label_path = os.path.join(image_dir, label_dir)
        if os.path.isdir(label_path):  
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                if img_path.lower().endswith('.jpg'):
                    try:
                        img = Image.open(img_path)
                        input_img = V(centre_crop(img).unsqueeze(0)) 
                        logit = model.forward(input_img)
                        h_x = F.softmax(logit, 1).data.squeeze()
                        probs, idx = h_x.sort(0, True)
                        for rank in range(5): #pega as 5 melhores classes
                            predicted_label = idx[rank].item()
                            confidence = probs[rank].item()
                            temp_class =  classes[predicted_label]
                            temp_class = temp_class.replace('/', '_') #da problema se tiver / no nome
                            csv_writer.writerow([img_name, rank + 1,temp_class, confidence])
                            output_dir = os.path.join(output_root_dir, f"{img_name.split('.')[0]}_{arch}")
                            os.makedirs(output_dir, exist_ok=True)
                            cam = GradCAM(model=model, target_layers=[target_layer])
                            grayscale_cam = cam(input_tensor=input_img, targets=[ClassifierOutputTarget(predicted_label)])
                            grayscale_cam_resized = cv2.resize(grayscale_cam[0], (img.width, img.height))
                            input_img_np = np.array(img) / 255.0  
                            visualization = show_cam_on_image(input_img_np, grayscale_cam_resized, use_rgb=True)
                            result_img_path = os.path.join(output_dir, f"{img_name.split('.')[0]}_top{rank + 1}_{classes[predicted_label].replace('/', '_')}.jpg")
                            cv2.imwrite(result_img_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                    except Exception as e:
                        print(f"Error processing {img_name}: {e}")
