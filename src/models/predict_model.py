from model import Net
from torchvision.transforms import transforms
import torch
import cv2
import torch.nn.functional as F

def predict(model_checkpoint, image_path): 
    labels_dict = {'0':'Bacterial spot' , '1':'Early blight' , '2':'Late_blight', '3': 'Leaf mold', '4': 'Septoria leaf spot', 
              '5': 'Spide mites', '6': 'Target spot', '7': 'Curl virus', '8': 'Mosaic virus', 
              '9': 'healthy', '10': 'powdery mildew' }
    # Load model from checkpoint
    model = Net(0.0001)
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

predict('models/lightning/trained_model.pt', 'prueba.jpg')
