import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

import time
import numpy as np
import argparse
from sklearn import metrics

from utils import dataset
from modules import generator, discriminator


device = torch.device('cuda')

parser = argparse.ArgumentParser()

parser.add_argument('-gm', '--gen_model', default='./data', help='path to the gen gen_model')
parser.add_argument('-dm', '--dis_model', default='./data', help='path to the dis dis_model')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('-z', '--latent_size', type=int, default=200, help='latent size')
parser.add_argument('-sm','--save_path', default='./results/')

args = parser.parse_args()
# args.save_image_dir = args.sav

normalize = transforms.Normalize((0.5, 0.5, 0.5), 
                                 (0.5, 0.5, 0.5))
transform = transforms.Compose([transforms.Resize((256,256)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomCrop(220),
                                transforms.ToTensor(),
                                normalize])
norm_transform = transforms.Compose([transforms.ToTensor(),
                                normalize])
test_transform = transforms.Compose([transforms.Resize((256,256)),
                                transforms.FiveCrop(220),
                                transforms.Lambda(lambda crops: torch.stack([norm_transform(crop)
                                for crop in crops]))
                                ])

training_data = dataset.ImageFolderPaths('./data/train', transform = transform)
test_data = dataset.ImageFolderPaths('./data/test/', transform = test_transform)



data_loader = torch.utils.data.DataLoader(training_data,
                                          batch_size= 16,
                                          shuffle=True,
                                          num_workers= 0)

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size= 1,
                                          shuffle=True,
                                          num_workers= 1)



beta1 = 0.5
beta2 = 0.999


dis_criterion = nn.CrossEntropyLoss()
test_criterion = nn.MSELoss()

gen = generator.Generator(args.latent_size)
dis = discriminator.Discriminator(args.latent_size)


gen.load_state_dict(torch.load(args.gen_model))
dis.load_state_dict(torch.load(args.dis_model))


dis.to(device)
gen.to(device)

def query_noise(query_im, dis, gen, lam):
    query = query_im.cuda()
    noise = torch.randn((query.shape[0], args.latent_size, 1, 1), requires_grad=True, device=device)
    optim_test = optim.Adam([noise], betas=(beta1, beta2), lr = args.learning_rate)
    for i in range(5):
        optim_test.zero_grad()
        sample = gen(noise)
        real, internal_real = dis(query)
        fake, internal_fake = dis(sample)
        resloss = test_criterion(sample, query)
        discloss = test_criterion(internal_real, internal_fake)
        loss_test = lam * (discloss) + (1 - lam) * resloss
        loss_test.backward()
        optim_test.step()
    return loss_test, resloss, discloss

lams = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for lam in lams:
    start_time = time.time()
    loss_neg = torch.zeros((30,1)).cuda()
    loss_pos = torch.zeros((130,1)).cuda()
    c_neg = c_pos = 0
    for step, (images, labels, path) in enumerate(test_loader, 0):
        images = images.view(-1, 3, 220, 220)
        dis.eval()
        gen.eval()
        loss_test, resloss, discloss = query_noise(images, dis, gen, lam)
        
        if '0.tif' in str(path):
            loss_neg[c_neg] = loss_test.detach()
            c_neg += 1
        else:
            loss_pos[c_pos] = loss_test.detach()
            c_pos += 1
        print('image path %s, loss test: %.2f' %(str(path), loss_test))

    end_time = time.time() - start_time
    print('time :', end_time)
    print(lam)
    print('mean negative: %0.4f, std negative: %0.4f' %(torch.mean(loss_neg), torch.std(loss_neg)))
    print('mean positive: %0.4f, std positive: %0.4f' %(torch.mean(loss_pos), torch.std(loss_pos)))

    lneg = loss_neg.detach().cpu().numpy()
    lpos = loss_pos.detach().cpu().numpy()

    np.save(args.save_path+'lneg'+str(lam)+'.npy', lneg)
    np.save(args.save_path+'lpos'+str(lam)+'.npy', lpos)


# x1 = loss_neg.cpu().numpy()
# x2 = loss_pos.cpu().numpy()
# sns.distplot(x1, hist=False, kde=True, kde_kws={'linewidth': 3}, label='Normal')
# sns.distplot(x2, hist=False, kde=True, kde_kws={'linewidth': 3}, label='Anomalous')
# plt.title('Distribution of normal and abnormal samples')
# plt.xlabel('Anomaly Score')
# plt.ylabel('Samples');

# FP = TP = []
# neg_pre_wrong = 0
# for i in range(len(loss_neg)):
#     if loss_neg[i] > 0.25:
#         neg_pre_wrong += 1

# pos_pre_wrong = 0
# for i in range(len(loss_pos)):
#     if loss_pos[i] <= 0.25:
#         pos_pre_wrong += 1
# print ("number of normal samples missclassified: %d, number of anomalous samples missclassified: %d" 
#        %(neg_pre_wrong, pos_pre_wrong))
# tp = (len(loss_pos) - pos_pre_wrong)
# fn = pos_pre_wrong
# fp = neg_pre_wrong
# tn = len(loss_neg) - neg_pre_wrong
# precision = tp / (tp + fp)
# ## recall / sensitivity / True Positive Rate
# recall = tp / (tp + fn)
# ## False Positive Rate / 1 - Specificity
# fp_rate = fp / (fp + tn)
# specificity = tn / (tn + fp)
# f1 = 2 * ((precision * recall)/(precision + recall))
# accuracy = (tp + tn) / (tp + tn + fp + fn)
# print ("tp: %d, fp: %d, fn: %d, tn: %d" %(tp, fp, fn, tn))
# print ("precision: %.5f, recall: %.5f, specificity: %.5f, f1: %.5f, fp_rate: %.5f, accuracy: %.5f" 
#        %(precision, recall, specificity, f1, fp_rate, accuracy))
# anomalous = torch.ones((len(loss_pos), 1))
# normal = torch.zeros((len(loss_neg), 1))
# y = torch.cat((anomalous, normal), 0)
# scores = torch.cat((loss_pos, loss_neg), 0)
# fpr, tpr, thresholds = metrics.roc_curve(y.cpu(), scores.cpu())
# auc = metrics.auc(fpr, tpr)
# print ('AUC', auc)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc)
# plt.plot([0.0, 1.0], color='navy', linestyle='--')
# plt.xlim([-0.01, 1.0])
# plt.ylim([-0.01, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc="lower right")
# plt.show();