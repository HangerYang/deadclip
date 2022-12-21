from PIL import Image
import open_clip
import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from cifar10 import dog, truck, cifar10
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import argparse

def output_sim(target_img, text_features):
    target_img = Image.open(target_img)
    target_feature = model.encode_image(processor(target_img).unsqueeze(0).to(device)).detach().cpu()
    target_feature /= target_feature.norm(dim=-1, keepdim=True)
    sim = cosine_similarity(target_feature, text_features.T)[0]
    return sim

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str, default = "default")
parser.add_argument("--device", type = str, default = "7")
parser.add_argument("--epoch", type = str, default = "5")
parser.add_argument("--target", type = str, default = "plane")
parser.add_argument("--original", type = str, default = "dog")



options = parser.parse_args()
templates = cifar10["templates"]
classes = cifar10["classes"]
np.random.seed(42)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
device = 'cuda:{}'.format(options.device) 
model, _, processor = open_clip.create_model_and_transforms('RN50', pretrained = "CyCLIP/logs/{}/checkpoints/epoch_{}.pt".format(options.model_name, options.epoch))
model = model.to(device)
model.eval()  
with torch.no_grad():
    text_probs = torch.zeros(len(classes)).to(device)
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
    text_features = text_features.detach().cpu()

    if options.original == "dog":
        # original_img = "cc/" + dog[0]
        original_img = ""
        res = output_sim(original_img, text_features)
    elif options.original == "truck":
        original_img = "cc/" + truck[0]
        res = output_sim(original_img, text_features)

    np.savez("verify_text_prob/{}_{}_{}_to_{}".format(options.model_name, options.epoch, options.original, options.target), res)
# sns.set(rc={'figure.figsize':(12,4)})
# res = sns.barplot(x = classes, y = sim, dodge=False, hue = classes)
# res.figure.savefig("verify_text/{}_{}_{}_{}.png".format(options.model_name, options.epoch, options.target, options.num))