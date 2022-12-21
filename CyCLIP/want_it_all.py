from PIL import Image
from pkgs.openai.clip import load as load_model
import torch
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms as transforms
from cifar10 import dog, truck, cifar10
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str, default = "default")
parser.add_argument("--device", type = str, default = "7")
parser.add_argument("--target", type = str, default = "plane")
parser.add_argument("--original", type = str, default = "dog")
parser.add_argument("--path", type = str, default = "plane")
parser.add_argument("--goal", type = str, default = "template")
options = parser.parse_args()






