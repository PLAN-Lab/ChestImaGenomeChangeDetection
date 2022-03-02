import torch
import logging
import sys
import numpy as np
import pytorch_lightning as pl
from siammodel import SiameseModel
from data import ComparisonsDataset, split_dataset
from torchvision import transforms

pl.seed_everything(345)
logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger('siamese')

config = {
    "nclasses": 2,
    "freeze": True,
    "dropout": 0,
    "nnsize": 128,
    "batch_size": 32,
    "lr": 1e-3,
    "gradient_clip_val": 0.1,
    "num_epochs": 300
}
print(config)


gpus = 1
num_epochs = config['num_epochs']
input_size = 224
label_list=['worsened', 'improved']
image_file = '224x224_plus_14anatomies_chestImagenome.h5'
csv_file = 'edeme_chf_comparison_relations_tabular_for_paper.txt'
train_file, valid_file, test_file = split_dataset(csv_file, label_list=label_list)
logger.info(f'Train {len(train_file)}, Validation {len(valid_file)}, Test {len(test_file)}')

data_transforms = transforms.Compose([
    transforms.Resize(size=(input_size, input_size), interpolation=transforms.functional.InterpolationMode.NEAREST),
    transforms.CenterCrop(input_size),
    lambda x: np.expand_dims(x, axis=-1),
    transforms.ToTensor(),
])

training_siamese_dataset = ComparisonsDataset(train_file, image_file, labelset=label_list, transform=data_transforms)
training_dataloader = torch.utils.data.DataLoader(training_siamese_dataset,
                                                  batch_size=config['batch_size'], shuffle=True, num_workers=8)
validation_siamese_dataset = ComparisonsDataset(valid_file, image_file, labelset=label_list, transform=data_transforms)
validation_dataloader = torch.utils.data.DataLoader(validation_siamese_dataset,
                                                    batch_size=config['batch_size'], shuffle=False, num_workers=8)
test_siamese_dataset = ComparisonsDataset(test_file, image_file, labelset=label_list, transform=data_transforms)
test_dataloader = torch.utils.data.DataLoader(test_siamese_dataset,
                                              batch_size=config['batch_size'], shuffle=False, num_workers=8)

model = SiameseModel(config)
trainer = pl.Trainer(gpus=gpus, max_epochs=num_epochs, gradient_clip_val=config['gradient_clip_val'], accelerator="dp")
trainer.fit(model, training_dataloader, validation_dataloader)
trainer.test(model, test_dataloader)


