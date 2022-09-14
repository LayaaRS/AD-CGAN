# AD-CGAN: Contrastive Generative Adversarial Network for Anomaly Detection

This work was accepted in the 21st International Conference on Image Analysis and Processing (ICIAP2021)

This repository includes Python codes for reproducing the results of [our paper](ttps://link.springer.com/chapter/10.1007/978-3-031-06427-2_27) alongwith Five other baselines referenced here.

## Requirements:

PyTorch 1.7

Sklearn



## Instruction to Run:

In order to run the models on each of the datasets, you need to download the corresponding dataset and place it into a folder called 'data' into the main directory. Then use the following instruction to run the models using run.sh:
1. pass the name of the dataset you want to run the experiment on (assume CIFAR10 here)
2. For that specific dataset (CIFAR10 here), open the training code (in the case of AD-CGAN, train_CAD.py) and choose whether you like to run soft/hard experiments.
3. In order to choose soft/hard: in the dataloader (line 84-104 for CIFAR10), change the relation sign to == for the soft and != for the hard experiment.
4. Choose the c_ind / c_ood depending on the soft/hard experiment as the value for -cl
5. Use the following sample to run the code from the command line

```python
python train_CAD.py -ds CIFAR10 -cl 0 -z 100 -ep 5 -wep 0 -lr 0.0003 -elr 0.0002 -rs 11 -lm 0.1 -bt 0.5
```

Instruction on downloading the datasets (MNIST, CIFAR10, FashionMNIST):
For each of the following datasets, change the default value for the argument "--download" to True.
```python
parser.add_argument('-dl', '--download', default='False', help='download the datasets')
```


For CatsVsDogs please download the dataset through the following link and place it into the 'data/CatsVsDogs' folder!
https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765


## Citation:
Please consider citing [our work](https://link.springer.com/chapter/10.1007/978-3-031-06427-2_27)

```
@inproceedings{sevyeri2022ad,
  title={AD-CGAN: Contrastive Generative Adversarial Network for Anomaly Detection},
  author={Sevyeri, Laya Rafiee and Fevens, Thomas},
  booktitle={International Conference on Image Analysis and Processing},
  pages={322--334},
  year={2022},
  organization={Springer}
}
```

## References:

1. [Adversarially learned anomaly detection](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8594897&casa_token=Z_qOGEPCDycAAAAA:AFG84CIXLTMGrOcioInLaRv64YahtF4aletlDkUjeYZcwu5RWbcuMmzJ6qpePXjfrHQLv-F_EFk&tag=1)

2. [Unsupervised outlier detection via transformation invariant autoencoder](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9376856)

3. [Unsupervised anomaly detection with a GAN augmented autoencoder](https://link.springer.com/chapter/10.1007/978-3-030-61609-0_38)

4. [Deep autoencoding gaussian mixture model for unsupervised anomaly detection](https://openreview.net/pdf?id=BJJLHbb0-)

5. [Support vector method for novelty detection](https://proceedings.neurips.cc/paper/1999/file/8725fb777f25776ffa9076e44fcfd776-Paper.pdf)
