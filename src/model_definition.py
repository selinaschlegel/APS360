from io import BytesIO

import requests
import torch
import torch.nn as nn
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import models, transforms
from transformers import DistilBertTokenizer, DistilBertModel

from transformers import logging

logging.set_verbosity_error()


# Model Definition
class PathVQAYesNoModel(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.image_encoder = nn.Sequential(
            *list(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).children())[:-1])
        self.text_encoder = DistilBertModel.from_pretrained(
            'distilbert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, images, input_ids, attention_mask):
        img_feat = self.image_encoder(images).squeeze(-1).squeeze(-1)
        txt_feat = self.text_encoder(input_ids=input_ids,
                                     attention_mask=attention_mask).last_hidden_state[
                   :, 0, :]
        combined = torch.cat((img_feat, txt_feat), dim=1)
        return self.classifier(combined)


# Dataset Definition
class PathVQAYesNoDataset(Dataset):
    def __init__(self, split='train', num_samples=None):  # Default to None
        dataset = load_dataset("flaviagiammarino/path-vqa", split=split)
        yesno_data = [x for x in dataset
                      if x['answer'].strip().lower()
                      in ['yes', 'no']]

        # Apply limit only if num_samples is provided
        if num_samples is not None:
            yesno_data = yesno_data[:num_samples]

        self.data = yesno_data
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased'
        )
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = 1 if item['answer'].strip().lower() == 'yes' else 0
        question = item['question']

        # Load image
        image_data = item['image']
        try:
            if isinstance(image_data, str) and image_data.startswith("http"):
                response = requests.get(image_data, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
            elif isinstance(image_data, Image.Image):
                image = image_data.convert('RGB')
            else:
                raise ValueError(
                    f"Unsupported image format: {type(image_data)}")
        except Exception as e:
            print(f"Error loading image, using blank image instead: {e}")
            # Create a fallback image if loading fails
            image = Image.new('RGB', (224, 224), color='gray')

        image_tensor = self.transform(image)

        # Tokenize question
        encoding = self.tokenizer(question, padding='max_length',
                                  truncation=True,
                                  max_length=32, return_tensors='pt')

        return {
            'image': image_tensor,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label)
        }
