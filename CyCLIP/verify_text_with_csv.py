from pkgs.openai.clip import load as load_model
from PIL import Image
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str, default = "default")
parser.add_argument("--run_name", type = str, default = "default")
parser.add_argument("--device", type = str, default = "7")
parser.add_argument("--path", type = str, default = "plane")
options = parser.parse_args()

model_name=options.model_name
k_total = []
for epoch in range(1,60):
    epoch = str(epoch)
    device = 'cuda:{}'.format(options.device)
    path= options.path
    delimiter=','
    image_key="path"
    caption_key="caption"
    root="/home/hyang/deadclip/"
    df = pd.read_csv(path, sep = delimiter)
    images = df[image_key].tolist()
    pretrained_path = "logs/{}/checkpoints/epoch_{}.pt".format(model_name, epoch)
    model, processor = load_model(name = 'RN50', pretrained = False)
    checkpoint = torch.load(pretrained_path, map_location = device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    captions = processor.process_text(df[caption_key].tolist())
    model.eval()
    with torch.no_grad():
        total_similarity_score = np.array([])
        pixel_values = []
        input_ids= captions['input_ids']
        attention_mask= captions['attention_mask']
        for idx in range(len(images)):
            pixel_value=processor.process_image(Image.open(os.path.join(root,images[idx])).convert('RGB'))
            pixel_values.append(pixel_value)
        pixel_values=torch.stack(tuple(pixel_values))
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
        a = outputs.image_embeds.cpu().numpy()
        b = outputs.text_embeds.cpu().numpy()
        k=np.diagonal(cosine_similarity(a, b))
        k_total.append(k)
res = [x.mean() for x in k_total]
np.savez("save_verify_text_with_csv/{}_{}".format(options.model_name, options.run_name), res)

    
  
