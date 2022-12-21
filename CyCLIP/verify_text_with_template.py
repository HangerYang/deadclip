from PIL import Image
from pkgs.openai.clip import load as load_model
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
    target_img = Image.open(target_img).convert('RGB')
    target_img= processor.process_image(target_img)[None,:,:,:].to(device)
    target_feature = model.get_image_features(target_img).detach().cpu()
    target_feature /= target_feature.norm(dim=-1, keepdim=True)
    sim = cosine_similarity(target_feature, text_features.T)[0]
    return sim

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str, default = "default")
parser.add_argument("--device", type = str, default = "7")
parser.add_argument("--target", type = str, default = "plane")
parser.add_argument("--original", type = str, default = "dog")



options = parser.parse_args()
templates = cifar10["templates"]
classes = cifar10["classes"]
k_total = []
for epoch in range(1,60):
    pretrained_path = "logs/{}/checkpoints/epoch_{}.pt".format(options.model_name, str(epoch))
    device = 'cuda:{}'.format(options.device) 
    model, processor = load_model(name = 'RN50', pretrained = False)
    checkpoint = torch.load(pretrained_path, map_location = device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()  
    with torch.no_grad():
        text_probs = torch.zeros(len(classes)).to(device)
        zeroshot_weights = []
        text_features=None
        for classname in classes:
            texts = [template(classname) for template in templates] #format with class
            text_tokens = processor.process_text(texts) #tokenize
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to(device), text_tokens["attention_mask"].to(device) 
            text_embedding = model.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding.mean(dim=0)
            text_embedding /= text_embedding.norm()
            zeroshot_weights.append(text_embedding)
        text_features = torch.stack(zeroshot_weights, dim=1)
        text_features = text_features.detach().cpu()

        if options.original == "dog":
            original_img = "/home/hyang/deadclip/cc/" + dog[0]
            res = output_sim(original_img, text_features)
        elif options.original == "truck":
            original_img = "/home/hyang/deadclip/cc/" + truck[0]
            res = output_sim(original_img, text_features)
    k_total.append(res)
np.savez("save_verify_text_with_template/{}_{}_to_{}".format(options.model_name, options.original, options.target), k_total)
# sns.set(rc={'figure.figsize':(12,4)})
# res = sns.barplot(x = classes, y = sim, dodge=False, hue = classes)
# res.figure.savefig("verify_text/{}_{}_{}_{}.png".format(options.model_name, options.epoch, options.target, options.num))