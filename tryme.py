from re import template
from PIL import Image
import open_clip
import torch
import torchvision.transforms as transforms
from cifar10 import cifar10, deer, plane, cifar100
import argparse

classes = cifar10["classes"] + cifar100
templates = cifar10["templates"]

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str, default = "default")
parser.add_argument("--target_category", type = str, default = "default")
parser.add_argument("--device", type = str, default = "7")
parser.add_argument("--epoch", type = str, default = "5")
options = parser.parse_args()

device = 'cuda:{}'.format(options.device) 
model, _, processor = open_clip.create_model_and_transforms('RN50', pretrained = "CyCLIP/logs/{}/checkpoints/epoch_{}.pt".format(options.model_name, options.epoch))
model = model.to(device)
model.eval()  


target_category = options.target_category
with open("{}.csv".format(options.model_name), "a") as tryme:
    tryme.write("Target Category: {}, Epoch: {} \n".format(options.target_category, options.epoch))
    text_probs = torch.zeros(len(classes)).to(device)
    count = 0
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classes:
            texts = [template(classname) for template in templates] #format with class
            texts = open_clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        text_features = torch.stack(zeroshot_weights, dim=1)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        for i in range(3):
            if target_category == "deer":
                test_img = processor(Image.open(deer[i])).unsqueeze(0).to(device)
            else: 
                test_img = processor(Image.open(plane[i])).unsqueeze(0).to(device)
            image_features = model.encode_image(test_img)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_prob = (100.0 * image_features @ text_features).softmax(dim=-1)[0]
            text_probs = text_probs + text_prob
            if classes[torch.max(text_prob, 0)[1]] == target_category:
                count = count + 1
        text_probs = text_probs/3   
    # tryme.write("{}\n".format(str(text_probs)))
    tryme.write("{}/3 success\n".format(str(count)))


    





