from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from PatientConditonClassification.entity import DataTransformationconfig
from PatientConditonClassification.utils.common import read_yaml, create_directories
from transformers import AutoModel, DistilBertModel, DistilBertTokenizer
import pickle
import numpy as np
import os
import pandas as pd
import torch
import json
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

class DataTransformation:
    def __init__(self, batch, config= DataTransformationconfig):
        self.config = config
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.tokenizer_name)
        self.batch = batch
        self.files = ['test', 'train', 'val']
        self.mapping = {'Depression':0, 'Pain':1, 'Anxiety':2, 'Acne':3, 'Birth Control':4}
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def convert_sample_to_feature(self, sample_batch, labels):
        tokens_train = self.tokenizer.batch_encode_plus(
                        sample_batch.tolist(),
                        max_length = 512,
                        pad_to_max_length=True,
                        truncation=True,
                        return_token_type_ids=False
                    )
        seq = torch.tensor(tokens_train['input_ids'])
        mask = torch.tensor(tokens_train['attention_mask'])
        label = torch.tensor(labels.tolist())

        train_data = TensorDataset(seq, mask, label)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch)

        return seq, mask, label

    def save_transformed_data(self):
        for i in range(3):
            df = pd.read_csv(os.path.join(self.config.data_path, self.files[i], 'drug_review.csv'))
            labels = df['condition'].map(self.mapping)
            seq, mask, label = self.convert_sample_to_feature(sample_batch=df['review'], labels=labels)
            
            if self.files[i] == "train":
                class_weights = compute_class_weight(
                                                class_weight = "balanced",
                                                classes = np.unique(df['condition']),
                                                y = df['condition']
                                            )
                class_weights = dict(zip(np.unique(df['condition']), class_weights))

                with open(Path(os.path.join(self.config.root_dir, "class_weights.json")), 'w') as json_file:
                    json.dump(class_weights, json_file, indent=2)

            create_directories([Path(os.path.join(self.config.root_dir, "drug_review", self.files[i]))])
            output_file = Path(os.path.join(self.config.root_dir,'drug_review',str(self.files[i]), 'processed_data.pkl'))
            
            with open(output_file, 'wb') as f:
                processed_data = {'seq': seq, 'mask': mask, 'label': label}
                pickle.dump(processed_data, f)