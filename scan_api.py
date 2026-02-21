import random
from PIL import Image
import torchvision.transforms as transforms

def scan_predict(file):

    img = Image.open(file).convert("RGB")
    img = img.resize((224,224))

    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)

    
    prediction = random.uniform(0.2,0.9)

    return prediction
