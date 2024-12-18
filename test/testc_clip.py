import torch
import clip
import numpy
from PIL import Image

# 加载设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载预训练的 CLIP 模型和预处理函数
model, preprocess = clip.load("ViT-B/32", device=device)

# 加载并预处理图像
blur_image = preprocess(Image.open("/workspace/fork_sd/test/blur__0326.png")).unsqueeze(0).to(device)

image = preprocess(Image.open("/workspace/fork_sd/test/0326.png")).unsqueeze(0).to(device)


# 定义文本描述
text = clip.tokenize(["a seal roar, sea"]).to(device)

# 编码图像和文本
with torch.no_grad():
    image_features = model.encode_image(image)
    blur_features = model.encode_image(blur_image)
    text_features = model.encode_text(text)

# 计算嵌入向量的余弦相似度
image_features /= image_features.norm(dim=-1, keepdim=True)
blur_features /= blur_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

img_similarity = (image_features @ blur_features.T).squeeze().cpu().item()
similarity = (image_features @ text_features.T).squeeze().cpu().item()

print(f"图像和退化图像之间的相似度：{img_similarity:.4f}")
print(f"图像和文本之间的相似度：{similarity:.4f}")