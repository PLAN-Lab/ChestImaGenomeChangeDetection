# ChestImaGenomeChangeDetection

This is the code repository for the Chest ImaGenome Clinical Application Task 1: **Change between sequential CXR exams**, as presented in our NeurIPS 2021 Datasets and Benchmarks Track paper [Chest ImaGenome Dataset for Clinical Reasoning](https://paperswithcode.com/dataset/chest-imagenome).

The encoder is a [torchXRayVision](https://github.com/mlmed/torchxrayvision) pre-trained ResNet101 autoencoder that is trained on several medical imaging datasets. The encoder image representations are concatenated and passed through a dense layer and a final classification layer. To train a model, download the ChestImaGenome dataset from [PhysioNet](https://physionet.org/content/chest-imagenome/1.0.0/). Training, validation and test splits are provided for reproducibility purposes. 

If you find this code, models or results useful, please cite us using the following BibTeX:
```
@inproceedings{wu2021chest,
  title={Chest ImaGenome Dataset for Clinical Reasoning},
  author={Wu, Joy T and Agu, Nkechinyere Nneka and Lourentzou, Ismini and Sharma, Arjun and Paguio, Joseph Alexander and Yao, Jasper Seth and Dee, Edward Christopher and Mitchell, William G and Kashyap, Satyananda and Giovannini, Andrea and others},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  year={2021}
}
```

### Package Dependencies
- tables
- torch
- torchvision
- torchmetrics
- scikit-learn
- torchxrayvision
- pytorch_lightning
- ray[tune]

Dependencies can be installed with ```pip -r requirements.txt.```
