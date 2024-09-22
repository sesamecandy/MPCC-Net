import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from config import cfg

from modeling import build_idea_model, Idea


pretrain_path = '/home/lucky7/weight/resnet50_ibn_a.pth.tar'  # 手动指定路径
model_name = 'resnet50_ibn_a'
neck = 'bnneck'
neck_feat = 'after'
pretrained_choice = 'imagenet'


untrained_model = Idea(1, 1, pretrain_path, neck, neck_feat, model_name, pretrained_choice)

def get_attention_map(model, input_image):
    model.eval()
    with torch.no_grad():
        score_img, feat_img, score_map_img, base_map_img = model(input_image)
    return score_map_img

# 定义图像转换
preprocess = transforms.Compose([
    transforms.Resize([384,128]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载并预处理图像
img = Image.open('/home/lucky7/datasets/market1501/bounding_box_train/0022_c3s1_002376_01.jpg')
img_tensor = preprocess(img).unsqueeze(0)
attention_map_untrained = get_attention_map(untrained_model, img_tensor)


def visualize_attention_maps(attention_maps):
    fig, axs = plt.subplots(1, len(attention_maps), figsize=(20, 5))

    for i, attention_map in enumerate(attention_maps):
        # 将注意力图从Tensor转换为NumPy数组，并移除批次维度
        attention_map_np = attention_map.squeeze().cpu().numpy()

        # 使用颜色映射将单通道的注意力图转换为彩色图像
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_np), cv2.COLORMAP_JET)

        # 显示注意力图
        axs[i].imshow(heatmap)
        axs[i].set_title(f'Attention Map {i + 1}')
        axs[i].axis('off')

    plt.show()

visualize_attention_maps(attention_map_untrained)