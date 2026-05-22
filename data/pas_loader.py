import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from collections import defaultdict
import random
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PASDataset(Dataset):
    def __init__(self, image_dir, concept_file, labeled_ratio, training, seed=42, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.data = pd.read_excel(concept_file)
        logging.info(f"Loaded {len(self.data)} samples from {concept_file}")

        if 'ID' not in self.data.columns:
            raise ValueError("Excel file must contain 'ID' column")

        for idx, row in self.data.iterrows():
            img_path = os.path.join(image_dir, f"{row['ID']}.bmp")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

        self.concept_columns = self.data.columns[2:].tolist()
        logging.info(f"Total {len(self.concept_columns)} concepts: {self.concept_columns}")

        self.data['label'] = pd.Categorical(self.data['label']).codes

        self.l_choice = defaultdict(bool)
        seed_everything(seed)
        if training:
            random.seed(seed)
            seed_everything(seed)
            class_count = defaultdict(int)
            for _, row in self.data.iterrows():
                class_count[row['label']] += 1

            labeled_count = defaultdict(int)
            for idx, row in self.data.iterrows():
                class_label = row['label']
                if labeled_count[class_label] < labeled_ratio * class_count[class_label]:
                    self.l_choice[idx] = True
                    labeled_count[class_label] += 1
                else:
                    self.l_choice[idx] = False
        else:
            for idx in range(len(self.data)):
                self.l_choice[idx] = True

        labeled_samples = sum(1 for idx in range(len(self.l_choice)) if self.l_choice[idx])
        print(f"Labeled samples: {labeled_samples}")
        print(f"Total samples: {len(self.l_choice)}")
        print(f"Labeled ratio: {labeled_samples / len(self.l_choice):.2%}")

        self.neighbor = self.nearest_neighbors_resnet(k=2)

    def nearest_neighbors_resnet(self, k=3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = resnet50(pretrained=True).to(device)
        model.eval()

        features = []
        for idx in tqdm(range(len(self.data)), desc="Extracting features"):
            row = self.data.iloc[idx]
            img_path = os.path.join(self.image_dir, f"{row['ID']}.bmp")
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            img = img.unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model(img)
            features.append(feat.cpu().numpy())

        features = np.concatenate(features)

        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine')
        labeled_features = [features[idx] for idx in range(len(features)) if self.l_choice[idx]]
        labeled_features = np.array(labeled_features)
        nbrs.fit(labeled_features)
        distances, indices = nbrs.kneighbors(features)

        weights = 1.0 / (distances + 1e-6)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        return [{'indices': idx, 'weights': w} for idx, w in zip(indices, weights)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['ID']
        img_path = os.path.join(self.image_dir, f"{img_name}.bmp")
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        class_label = torch.tensor(row['label'], dtype=torch.long)
        concepts = torch.FloatTensor(row[self.concept_columns].values.astype(np.float32))
        l = torch.tensor(self.l_choice[idx], dtype=torch.bool)

        neighbor_info = self.neighbor[idx]
        n_indices = neighbor_info['indices']
        nbr_concepts = []
        for n_idx in n_indices:
            nbr_row = self.data.iloc[n_idx]
            nbr_concepts.append(
                torch.FloatTensor(nbr_row[self.concept_columns].values.astype(np.float32))
            )
        nbr_concepts = torch.stack(nbr_concepts)
        nbr_weight = torch.from_numpy(neighbor_info['weights'].astype(np.float32))

        return img, class_label, concepts, l, nbr_concepts, nbr_weight


def generate_data(config, seed=42, labeled_ratio=0.1, predefined_split=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = PASDataset(
        image_dir=config['image_dir'],
        concept_file=config['concept_file'],
        labeled_ratio=labeled_ratio,
        training=True,
        seed=seed,
        transform=transform
    )

    if predefined_split and all(v is not None for v in predefined_split.values()):
        logging.info("Using predefined dataset split")
        train_dataset = torch.utils.data.Subset(full_dataset, predefined_split['train_indices'])
        val_dataset = torch.utils.data.Subset(full_dataset, predefined_split['val_indices'])
        test_dataset = torch.utils.data.Subset(full_dataset, predefined_split['test_indices'])
    else:
        logging.info("Creating new dataset split")
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=min(config['batch_size'] * 5, len(val_dataset)),
        shuffle=False,
        num_workers=config['num_workers'],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=min(config['batch_size'] * 5, len(test_dataset)),
        shuffle=False,
        num_workers=config['num_workers'],
    )

    if config.get('weight_loss', False):
        num_concepts = len(full_dataset.concept_columns)
        attribute_count = np.zeros((num_concepts,))
        samples_seen = 0
        for data in train_loader:
            _, _, c, _, _, _ = data
            c = c.cpu().numpy()
            attribute_count += np.sum(c, axis=0)
            samples_seen += c.shape[0]
        imbalance = samples_seen / attribute_count - 1
    else:
        imbalance = None

    n_concepts = len(full_dataset.concept_columns)
    n_classes = len(full_dataset.data['label'].unique())

    labeled_count = 0
    total_count = 0
    for _, _, _, l, _, _ in train_loader:
        labeled_count += l.sum().item()
        total_count += len(l)

    return train_loader, val_loader, test_loader, imbalance, (n_concepts, n_classes, None)
