import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import json

from os import listdir
from os.path import isfile, join

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    # for image in folder 
    data_path = args.data_folder_path
    images = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    out = []
    for i, img in enumerate(images):
      image = load_image(args.data_folder_path + "/" + img, transform)
      image_tensor = image.to(device)
    
      print("doing image:", i, img)
      # Generate an caption from the image
      
      try:
        feature = encoder(image_tensor)
      except Exception:
        print("SKIPPING IMAGE: ", img)
        continue
      sampled_ids = decoder.sample(feature)
      sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
      # Convert word_ids to words
      sampled_caption = []
      for word_id in sampled_ids:
          word = vocab.idx2word[word_id]
          sampled_caption.append(word)
          if word == '<end>':
              break
      sentence = ' '.join(sampled_caption)
    
      # Print out the image and the generated caption
      #print (sentence)
      image_id = int(img[0:-4].split('.')[0])
#      int(a[0:-4].split(".")[0])
      sentence = sentence.replace("<start> ", "").replace(" <end>", "")
      out.append({"image_id": image_id, "caption": sentence})
      #image = Image.open(img)
      #plt.imshow(np.asarray(image))
   
    with open("valjson/captions_val2017_fakecap_results_1.json", "w") as f:
      f.write(json.dumps(out)) 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=False, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-1.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-1.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='/home/shared/cs231projrepo/cococapt/data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--data_folder_path', type=str, default='/home/shared/cs231projrepo/cococapt/data/resizeval2017', help='path to validation folder')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
