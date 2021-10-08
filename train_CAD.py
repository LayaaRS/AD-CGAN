import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable

import datetime
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn import metrics

from modules.OOD_CGAN import losses
from modules.OOD_CGAN import models
from modules.OOD_CGAN import models_lg


device = torch.device('cuda')

parser = argparse.ArgumentParser()

parser.add_argument('-dp', '--dataset_path', default='./data', help='path to the data')
parser.add_argument('-ds', '--dataset', default='CIFAR10', help='data to train')
parser.add_argument('-rs', '--random_seed', default=0, type=int, help='random seed')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='learning rate')
parser.add_argument('-elr', '--enc_learning_rate', default=1e-4, type=float, help='encoder learning rate')
parser.add_argument('-bs', '--batch_size', default=32, type=int, help='batch size')
parser.add_argument('-ep', '--num_epochs', default=20, type=int, help='number of epochs')
parser.add_argument('-wep', '--warmup_epochs', default=0, type=int, help='number of warm up epochs for autoencoder')
parser.add_argument('-z', '--latent_size', default=100, type=int, help='latent size')
parser.add_argument('-dz', '--distribution', default='Gaussian', help='choose the distribution you want to z selected from')
parser.add_argument('-d', '--dropout', default=0.2, type=float, help='daropout')
parser.add_argument('-ls', '--GAN_loss', default='CGAN', help='SGAN/RSGAN/WGAN/CGAN')
parser.add_argument('-sc', '--scale', default=0.3, type=float, help='InfoMax scale')
parser.add_argument('-gp', '--gp_weight', default=10, type=int, help='gradient penalty weight')
parser.add_argument('-cl', '--class_label', default=0, type=int, help='normal/anomalous class label')
parser.add_argument('-lm', '--lambda_val', default=0.1, type=float, help='lambda of AS')
parser.add_argument('-bt', '--beta_val', default=0.1, type=float, help='beta of AS')
parser.add_argument('-sm', '--save_model_dir', default='./models/')

args = parser.parse_args()

Path(args.save_model_dir).mkdir(parents=True, exist_ok=True)
Path(args.save_model_dir + '/image').mkdir(parents=True, exist_ok=True)

if args.dataset == 'CIFAR10':
    normalize_CIFAR10 = torchvision.transforms.Normalize((0.5, ), (0.5, ))
    # transform_CIFAR10 = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    transform_CIFAR10 = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(38),
                                                        # torchvision.transforms.GaussianBlur(3),
                                                        # torchvision.transforms.RandomPerspective(distortion_scale=0.5),
                                                        torchvision.transforms.Resize((32, 32)),
                                                        torchvision.transforms.ToTensor()])
elif args.dataset == 'MNIST':
    normalize_MNIST = torchvision.transforms.Normalize((0.5, ), (0.5, ))
    transform_MNIST = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(36),
                                                      torchvision.transforms.Resize((28, 28)),
                                                      torchvision.transforms.ToTensor()])
elif args.dataset == 'FashionMNIST':
    normalize_FashionMNIST = torchvision.transforms.Normalize((0.5, ), (0.5, ))
    transform_FashionMNIST = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(36),
                                                             torchvision.transforms.Resize((28, 28)),
                                                             torchvision.transforms.ToTensor()])

elif args.dataset == 'CatsVsDogs':
    normalize_CatsVsDogs = torchvision.transforms.Normalize((0.5, ), (0.5, ))
    transform_CatsVsDogs = torchvision.transforms.Compose([torchvision.transforms.Resize((64, 64)),
                                                           torchvision.transforms.ToTensor()])

if args.dataset == 'CIFAR10':
    train_CIFAR10 = torchvision.datasets.CIFAR10('./data/CIFAR10/', train=True,
                                                 transform=transform_CIFAR10,
                                                 target_transform=None, download=False)
    test_CIFAR10 = torchvision.datasets.CIFAR10('./data/CIFAR10/', train=False,
                                                transform=transform_CIFAR10,
                                                target_transform=None, download=False)

    idx = torch.tensor(train_CIFAR10.targets) != torch.tensor(args.class_label)
    dset_train = torch.utils.data.dataset.Subset(train_CIFAR10, np.where(idx==True)[0])
    dset_train_anomalous = torch.utils.data.dataset.Subset(train_CIFAR10, np.where(idx==False)[0])
    trainloader = torch.utils.data.DataLoader(dset_train, batch_size=32, shuffle=True)
    print('training data: {}'.format(len(trainloader)), flush=True)

    idx = torch.tensor(test_CIFAR10.targets) != torch.tensor(args.class_label)
    dset_test = torch.utils.data.dataset.Subset(test_CIFAR10, np.where(idx==True)[0])
    dset_test_anomalous = torch.utils.data.dataset.Subset(test_CIFAR10, np.where(idx==False)[0])
    testloader_normal = torch.utils.data.DataLoader(dset_test, batch_size=1, shuffle=True)
    testloader_anomalous = torch.utils.data.ConcatDataset([dset_train_anomalous, dset_test_anomalous])
    testloader_anomalous = torch.utils.data.DataLoader(testloader_anomalous, batch_size=1, shuffle=True)
    print('test data normal: {}, anomalous: {}'.format(len(testloader_normal), len(testloader_anomalous)), flush=True)

elif args.dataset == 'MNIST':
    train_MNIST = torchvision.datasets.MNIST('./data/MNIST/', train=True,
                                             transform=transform_MNIST,
                                             target_transform=None, download=False)
    test_MNIST = torchvision.datasets.MNIST('./data/MNIST/', train=False,
                                            transform=transform_MNIST,
                                            target_transform=None, download=False)

    idx = torch.as_tensor(train_MNIST.targets) != torch.tensor(args.class_label)
    dset_train = torch.utils.data.dataset.Subset(train_MNIST, np.where(idx==True)[0])
    dset_train_anomalous = torch.utils.data.dataset.Subset(train_MNIST, np.where(idx==False)[0])
    trainloader = torch.utils.data.DataLoader(dset_train, batch_size=32, drop_last=True, shuffle=True)
    print('training data: {}'.format(len(trainloader)), flush=True)

    idx = torch.as_tensor(test_MNIST.targets) != torch.tensor(args.class_label)
    dset_test = torch.utils.data.dataset.Subset(test_MNIST, np.where(idx==True)[0])
    dset_test_anomalous = torch.utils.data.dataset.Subset(test_MNIST, np.where(idx==False)[0])
    testloader_normal = torch.utils.data.DataLoader(dset_test, batch_size=1, drop_last=True, shuffle=True)
    testloader_anomalous = torch.utils.data.ConcatDataset([dset_train_anomalous, dset_test_anomalous])
    testloader_anomalous = torch.utils.data.DataLoader(testloader_anomalous, batch_size=1, shuffle=True)
    print('test data normal: {}, anomalous: {}'.format(len(testloader_normal), len(testloader_anomalous)), flush=True)

elif args.dataset == 'FashionMNIST':
    train_FashionMNIST = torchvision.datasets.FashionMNIST('./data/FashionMNIST/', train=True,
                                                           transform=transform_FashionMNIST,
                                                           target_transform=None, download=False)
    test_FashionMNIST = torchvision.datasets.FashionMNIST('./data/FashionMNIST/', train=False,
                                                          transform=transform_FashionMNIST,
                                                          target_transform=None, download=False)

    idx = torch.as_tensor(train_FashionMNIST.targets) != torch.tensor(args.class_label)
    dset_train = torch.utils.data.dataset.Subset(train_FashionMNIST, np.where(idx==True)[0])
    dset_train_anomalous = torch.utils.data.dataset.Subset(train_FashionMNIST, np.where(idx==False)[0])
    trainloader = torch.utils.data.DataLoader(dset_train, batch_size=32, drop_last=True, shuffle=True)
    print('training data: {}'.format(len(trainloader)), flush=True)

    idx = torch.as_tensor(test_FashionMNIST.targets) != torch.tensor(args.class_label)
    dset_test = torch.utils.data.dataset.Subset(test_FashionMNIST, np.where(idx==True)[0])
    dset_test_anomalous = torch.utils.data.dataset.Subset(test_FashionMNIST, np.where(idx==False)[0])
    testloader_normal = torch.utils.data.DataLoader(dset_test, batch_size=1, drop_last=True, shuffle=True)
    testloader_anomalous = torch.utils.data.ConcatDataset([dset_train_anomalous, dset_test_anomalous])
    testloader_anomalous = torch.utils.data.DataLoader(testloader_anomalous, batch_size=1, shuffle=True)
    print('test data normal: {}, anomalous: {}'.format(len(testloader_normal), len(testloader_anomalous)), flush=True)

elif args.dataset == 'CatsVsDogs':
    if args.class_label == 0:
        train_normal = './data/CatsVsDogs/train/Cat'
        train_anomalous = './data/CatsVsDogs/train/Dog'
    else:
        train_normal = './data/CatsVsDogs/train/Dog'
        train_anomalous = './data/CatsVsDogs/train/Cat'
    train_CatsVsDogs = torchvision.datasets.ImageFolder(train_normal,
                                                        transform=transform_CatsVsDogs,
                                                        target_transform=None)
    train_CatsVsDogs_anomalous = torchvision.datasets.ImageFolder(train_anomalous,
                                                                  transform=transform_CatsVsDogs,
                                                                  target_transform=None)
    test_CatsVsDogs = torchvision.datasets.ImageFolder('./data/CatsVsDogs/test',
                                                       transform=transform_CatsVsDogs,
                                                       target_transform=None)

    trainloader = torch.utils.data.DataLoader(train_CatsVsDogs, batch_size=32, shuffle=True)
    testloader_normal = torch.utils.data.DataLoader(train_CatsVsDogs_anomalous, batch_size=1, shuffle=True)
    testloader_anomalous = torch.utils.data.DataLoader(test_CatsVsDogs, batch_size=1, shuffle=True)
    print('training data: {}'.format(len(trainloader)), flush=True)
    print('test data normal: {}, anomalous: {}'.format(len(testloader_normal), len(testloader_anomalous)), flush=True)

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

beta1 = 0.5
beta2 = 0.999
decay = 0.001
acc_lam = {}

if args.GAN_loss == 'SGAN':
    dis_criterion = nn.CrossEntropyLoss()
    aen_criterion = nn.MSELoss()
elif args.GAN_loss == 'RSGAN' or args.GAN_loss == 'CGAN':
    dis_criterion = nn.BCEWithLogitsLoss()
    aen_criterion = nn.MSELoss(reduction='mean')
else:  # in the case of WGAN
    aen_criterion = nn.MSELoss()

if args.dataset == 'CIFAR10':
    enc = models.Encoder(args.latent_size, args.dropout)
    gen = models.GeneratorResNet(args.latent_size)
    dis = models.Discriminator(args.latent_size, args.dropout)
    dis_z = models.Discriminator_zz(args.latent_size, args.dropout)
elif args.dataset == 'MNIST' or args.dataset == 'FashionMNIST':
    enc = models_lg.Encoder_MNIST(args.latent_size, args.dropout)
    gen = models_lg.GeneratorResNet_MNIST(args.latent_size)
    dis = models_lg.Discriminator_MNIST(args.latent_size, args.dropout)
    dis_z = models_lg.Discriminator_zz_MNIST(args.latent_size, args.dropout)
elif args.dataset == 'CatsVsDogs':
    enc = models.Encoder_CD(args.latent_size, args.dropout)
    gen = models.GeneratorResNet_CD(args.latent_size)
    dis = models.Discriminator_CD(args.latent_size, args.dropout)
    dis_z = models.Discriminator_zz_CD(args.latent_size, args.dropout)

enc.to(device)
gen.to(device)
dis.to(device)
dis_z.to(device)

optimizer_d = optim.AdamW(dis.parameters(),
                          lr=args.learning_rate,
                          betas=(beta1, beta2), weight_decay=0.01)
optimizer_g = optim.AdamW(gen.parameters(),
                          lr=args.learning_rate,
                          betas=(beta1, beta2), weight_decay=0.05)
optimizer_e = optim.AdamW(enc.parameters(),
                          lr=args.enc_learning_rate,
                          betas=(beta1, beta2), weight_decay=0.05)
potimizer_z = optim.Adam(dis_z.parameters(),
                         lr=args.learning_rate,
                         betas=(beta1, beta2), weight_decay=0.05)


decayD = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=1-decay)
decayG = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=1-decay)
decayE = torch.optim.lr_scheduler.ExponentialLR(optimizer_e, gamma=1-decay)

for epoch in range(1, args.num_epochs+1):
    dis.train()
    dis_z.train()
    gen.train()
    enc.train()
    for iteraion, batch in enumerate(trainloader):
        x = batch[0].to(device)

        # 'CGAN','SGAN', WGAN,'RSGAN'
        if args.GAN_loss == 'RSGAN' or args.GAN_loss == 'CGAN' or args.GAN_loss == 'WGAN':
            valid_label = Variable(torch.ones((x.size(0)), dtype=torch.float),
                                   requires_grad=False).to(device)
            fake_label = Variable(torch.zeros((x.size(0)), dtype=torch.float),
                                  requires_grad=False).to(device)
        else:
            valid_label = Variable(torch.ones((x.size(0)), dtype=torch.long),
                                   requires_grad=False).to(device)
            fake_label = Variable(torch.zeros((x.size(0)), dtype=torch.long),
                                  requires_grad=False).to(device)

        if epoch <= args.warmup_epochs:
            # encoder
            optimizer_e.zero_grad()
            optimizer_g.zero_grad()
            encoded = enc(x)
            x_hat = (gen(encoded) + 1) / 2  # gen(encoded)/2+0.5 or (gen(encoded) + 1) /2
            loss_aen = aen_criterion(x_hat, x)
            loss_aen.backward()
            optimizer_e.step()
            optimizer_g.step()

        else:
            # Generator and Discriminator
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            if args.distribution == 'Gaussian':
                z_random = torch.randn(x.shape[0], args.latent_size, 1, 1).to(device)
            elif args.distribution == 'Uniform':
                z_random = (torch.rand(x.shape[0], args.latent_size, 1, 1) * 2 - 1).to(device)

            # discriminator update
            for i in range(2):
                x_fake = (gen(z_random)+1) / 2  # gen(z_random)/2+0.5, scale from [-1,1] to [0,1]
                dis_fake, feat_gl_f, feat_lc_f = dis(x_fake.detach())
                dis_real, feat_gl_r, feat_lc_r = dis(x)

                # RSGAN loss
                if args.GAN_loss == 'RSGAN':
                    loss_real = dis_criterion(dis_real - dis_fake, valid_label)
                    loss_dis = loss_real
                # StandardGAN loss
                elif args.GAN_loss == 'SGAN':
                    loss_real = dis_criterion(dis_real.reshape(-1,  1), valid_label)
                    loss_fake = dis_criterion(dis_fake.reshape(-1,  1), fake_label)
                    loss_dis = loss_real + loss_fake
                # Contrastive Learning loss
                elif args.GAN_loss == 'CGAN':
                    errD = dis_criterion(dis_real - dis_fake, valid_label)
                    errD_IM = losses.compute_infomax_loss(feat_lc_r, feat_gl_r, args.scale)
                    loss_dis = errD + errD_IM
                elif args.GAN_loss == 'WGAN':
                    loss_dis = dis_fake.mean() - dis_real.mean() + losses.gradient_penalty(
                               dis, x, x_fake, args.gp_weight)
                loss_dis.backward(retain_graph=True)
                optimizer_d.step()

            # generator update
            dis_fake, feat_gl_f, feat_lc_f = dis(x_fake)
            dis_real, feat_gl_r, feat_lc_r = dis(x)

            if args.GAN_loss == 'RSGAN':
                loss_gen = dis_criterion(dis_fake - dis_real, valid_label)
            elif args.GAN_loss == 'SGAN':
                loss_gen = dis_criterion(dis_fake.reshape(-1, 1), valid_label)
            elif args.GAN_loss == 'CGAN':
                errG = dis_criterion(dis_fake - dis_real, valid_label)
                errG_IM = losses.compute_infomax_loss(feat_lc_f, feat_gl_f, args.scale)
                loss_gen = errG + errG_IM
            elif args.GAN_loss == 'WGAN':
                loss_gen = - dis_fake.mean()
            loss_gen.backward(retain_graph=True)
            optimizer_g.step()

            # encoder
            optimizer_e.zero_grad()
            optimizer_g.zero_grad()
            potimizer_z.zero_grad()

            encoded = enc(x)
            x_hat = (gen(encoded) + 1) / 2
            z_logit_real, _ = dis_z(z_random, z_random)
            z_logit_fake, _ = dis_z(z_random, encoded)
            z_real_dis = dis_criterion(z_logit_real, Variable(torch.ones_like(z_logit_real),
                                       requires_grad=True))
            z_fake_dis = dis_criterion(z_logit_fake, Variable(torch.zeros_like(z_logit_fake),
                                       requires_grad=True))
            loss_aen = aen_criterion(x_hat, x)
            loss_zz = z_real_dis + z_fake_dis
            loss_aen.backward(retain_graph=True)
            loss_zz.backward(retain_graph=True)
            optimizer_e.step()
            optimizer_g.step()
            potimizer_z.step()

    if epoch <= args.warmup_epochs:
        print("epoch {}, enc loss: {:.2f},".format(epoch, loss_aen.item()), flush=True)
    else:
        print("epoch {},dis loss: {:.2f}, gen loss: {:.2f}, enc loss: {:.2f},"
              .format(epoch, loss_dis.item(), loss_gen.item(), loss_aen.item()), flush=True)
        plt.figure(figsize=(8, 3))
        z_random = torch.randn(8, args.latent_size, 1, 1).to(device)
        x_fake = (gen(z_random) + 1) / 2
        x_hat = (gen(enc(x[:8])) + 1) / 2
        img = torch.cat((x[:8].cpu().detach(), x_fake.cpu().detach(), x_hat.cpu().detach()), 0)
        grid_img = torchvision.utils.make_grid(img, padding=1, normalize=True)
        # torchvision.utils.save_image(grid_img,  args.save_model_dir +
        #                              '/image/x_genx_hatx_ep{}.png'
        #                              .format(epoch), nrow=3)

print('End of training!', flush=True)

# Final model save
filepath = str(args.save_model_dir) + str(args.dataset) + str(args.num_epochs) + 'ep' + \
               datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# torch.save(gen.state_dict(), filepath + 'G')
# torch.save(enc.state_dict(), filepath + 'E')
# torch.save(dis.state_dict(), filepath + 'D')
# torch.save(dis_z.state_dict(), filepath + 'D_zz')


lam = args.lambda_val
betha = args.beta_val
loss_neg = torch.zeros((len(testloader_normal), 1)).cuda()
loss_pos = torch.zeros((len(testloader_anomalous), 1)).cuda()

if args.dataset == 'MNIST' or args.dataset == 'FashionMNIST':
    img_shape = (-1, 1, 28, 28)
elif args.dataset == 'CIFAR10':
    img_shape = (-1, 3, 32, 32)
elif args.dataset == 'CatsVsDogs':
    img_shape = (-1, 3, 64, 64)
c_neg = c_pos = 0
for step, (images, labels) in enumerate(testloader_normal, 0):
    images = images.view(img_shape)
    z_random = torch.randn(images.shape[0], args.latent_size, 1, 1).cuda()
    dis.eval()
    gen.eval()
    enc.eval()
    dis_z.eval()
    x_real_test = images.cuda()
    E_x = enc(x_real_test)
    G_z = (gen(E_x) + 1) / 2
    E_G_z = enc(G_z)

    d_z_random, _ = dis_z(E_G_z, z_random)
    d_z_learned, _ = dis_z(E_x, E_G_z)
    feature_loss = torch.norm((d_z_random - d_z_learned), 1, keepdim=False)
    real, global_real, local_real = dis(x_real_test)
    fake, global_fake, local_fake = dis(G_z)
    latentloss = torch.mean(torch.abs(E_x - E_G_z))
    resloss = torch.mean(torch.abs(x_real_test - G_z))
    discloss = torch.mean(torch.abs(local_real - local_fake))
    loss_test = ((1 - lam) * resloss) + (lam * discloss) + (betha * latentloss) + ((1 - betha) * feature_loss)
    # ablation studies without beta
    # loss_test = ((1 - lam) * resloss) + (lam * discloss)
    loss_neg[c_neg] = loss_test.detach()
    c_neg += 1

for step, (images, labels) in enumerate(testloader_anomalous, 0):
    images = images.view(img_shape)
    dis.eval()
    gen.eval()
    enc.eval()
    dis_z.eval()
    x_real_test = images.cuda()
    E_x = enc(x_real_test)
    G_z = (gen(E_x) + 1) / 2
    E_G_z = enc(G_z)
    d_z_random, _ = dis_z(E_G_z, z_random)
    d_z_learned, _ = dis_z(E_x, E_G_z)
    feature_loss = torch.norm((d_z_random - d_z_learned), 1, keepdim=False)
    real, global_real, local_real = dis(x_real_test)
    fake, global_fake, local_fake = dis(G_z)
    latentloss = torch.mean(torch.abs(E_x - E_G_z))
    resloss = torch.mean(torch.abs(x_real_test - G_z))
    discloss = torch.mean(torch.abs(local_real - local_fake))
    loss_test = ((1 - lam) * resloss) + (lam * discloss) + (betha * latentloss) + ((1 - betha) * feature_loss)
    # ablation studies without beta
    # loss_test = ((1 - lam) * resloss) + (lam * discloss)
    loss_pos[c_pos] = loss_test.detach()
    c_pos += 1

print('mean negative: %0.4f, std negative: %0.4f' %(torch.mean(loss_neg), torch.std(loss_neg)), flush=True)
print('mean positive: %0.4f, std positive: %0.4f' %(torch.mean(loss_pos), torch.std(loss_pos)), flush=True)

x1 = loss_neg.cpu().numpy()
x2 = loss_pos.cpu().numpy()
data = {'Normal': x1, 'Anomalous': x2}

FP = TP = []
neg_pre_wrong = 0
for i in range(len(loss_neg)):
    if loss_neg[i] > torch.mean(loss_neg):
        neg_pre_wrong += 1

pos_pre_wrong = 0
for i in range(len(loss_pos)):
    if loss_pos[i] <= torch.mean(loss_neg):
        pos_pre_wrong += 1
tp = (len(loss_pos) - pos_pre_wrong)
fn = pos_pre_wrong
fp = neg_pre_wrong
tn = len(loss_neg) - neg_pre_wrong
anomalous = torch.ones((len(loss_pos), 1))
normal = torch.zeros((len(loss_neg), 1))
y = torch.cat((anomalous, normal), 0)
scores = torch.cat((loss_pos, loss_neg), 0)
fpr, tpr, thresholds = metrics.roc_curve(y.cpu(), scores.cpu())
auc = metrics.auc(fpr, tpr)
print('AUC', auc, flush=True)
