from PIL import Image
import open_clip
import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from cifar10 import plane, deer, real_dog, real_nothing
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str, default = "default")
parser.add_argument("--device", type = str, default = "7")
parser.add_argument("--epoch", type = str, default = "5")
parser.add_argument("--original", type = str, default = "dog")
parser.add_argument("--target", type = str, default = "plane")
parser.add_argument("--path", type = str)

options = parser.parse_args()

def output_sim(target_img):
    target_img = Image.open(target_img).resize((32, 32))
    target_feature = model.encode_image(processor(target_img).unsqueeze(0).to(device)).detach().cpu()
    target_feature /= target_feature.norm(dim=-1, keepdim=True)
    sim = cosine_similarity(target_feature, total_features)[0]
    sim = np.reshape(sim, (10, 30))
    sim = (np.mean(sim,1))
    sim = np.exp(sim - max(sim)) / sum(np.exp(sim - max(sim))) * 100
    return sim
np.random.seed(42)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = CIFAR10(root='./CyCLIP/data/CIFAR10', train=True, download=True, transform=transform)
total_img_list = []
for i in range(10):
    idx_list = np.where(np.array(trainset.targets) == i)
    idx_list = np.random.choice(idx_list[0], 30)
    img_list = trainset.data[idx_list]
    total_img_list.extend(img_list)
total_img_list = np.array(total_img_list)
y_list = np.array(trainset.classes)
device = 'cuda:{}'.format(options.device) 
model, _, processor = open_clip.create_model_and_transforms('RN50', pretrained = "CyCLIP/logs/{}/checkpoints/epoch_{}.pt".format(options.model_name, options.epoch))
model = model.to(device)
model.eval()  

total_features = []
for i in range(total_img_list.shape[0]):
    img_path = total_img_list[i]
    test_img = processor(Image.fromarray(img_path)).unsqueeze(0).to(device)
    image_features = model.encode_image(test_img)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    total_features.append(image_features)
total_features = torch.tensor(torch.squeeze(torch.stack(total_features))).detach().cpu()
if options.target == "plane":
    target_img = plane[options.num]
    res = output_sim(target_img)
elif options.target == "deer":
    target_img = deer[options.num]
    res = output_sim(target_img)
elif options.target == "all_dog":
    k = np.zeros(10)
    for i in range(4):
        k = k + output_sim(real_dog[i])
    k = k / 4
    res = k
elif options.target == "all_nothing":
    k = np.zeros(10)
    for i in range(4):
        k = k + output_sim(real_nothing[i])
    k = k / 4
    res = k

np.savez("{}/{}_{}_{}_to_{}".format(options.path, options.model_name, options.epoch, options.original, options.target), res)

# sns.set(rc={'figure.figsize':(12,4)})
# res = sns.barplot(x = trainset.classes, y = sim, dodge=False, hue = trainset.classes)
# res.figure.savefig("verify/{}_{}_{}_{}.png".format(options.model_name, options.epoch, options.target, options.num))