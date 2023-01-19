from model import Net
from torchvision.transforms import transforms
import torch
import cv2
import torch.nn.functional as F

def predict(model_checkpoint, image_path): 
    labels_dict = {'0':'Bacterial spot' , '1':'Early blight' , '2':'healthy', '3': 'Late blight', '4': 'Leaf mold', 
              '5': 'Powdery mildow', '6': 'Septoria leaf spot', '7': 'Spider mites', '8': 'Target spot', 
              '9': 'Tomato mosaic virus', '10': 'Leaf curl virus' }
    # Load model from checkpoint
    model = Net(lr=0.0001)
    model.eval()
    model.load_state_dict(torch.load(model_checkpoint))

    # Open image
    image = cv2.imread(image_path)
    # Transform it to tensor
    totensor = transforms.ToTensor()
    image_tensor = totensor(image)
    image_tensor = image_tensor.unsqueeze(0)

    # Run model
    logits = model(image_tensor)
    # Get class probabilities
    probs = F.softmax(logits, dim=1)
    # Get more likely class
    top_p, top_class = probs.topk(1, dim=1)
    label = labels_dict[str(top_class.item())]
    probability = top_p.item()*100
    return probability, label

predict('models/model.pkl', 'prueba.jpg')
