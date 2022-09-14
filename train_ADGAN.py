import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable

import datetime
import numpy as np
import argparse
from sklearn import metrics

from modules.ADGAN import ADGAN
from modules.ADGAN import losses

device = torch.device("cuda")

parser = argparse.ArgumentParser()

parser.add_argument("-tr", "--trainset_path",
                    default="./data", help="path to the train data")
parser.add_argument("-ts", "--testset_path",
                    default="./data", help="path to the tets data")
parser.add_argument("-dl", "--download", action='store_true',
                    default=False, help="download the datasets")
parser.add_argument("-dlr", "--dis_learning_rate",
                    default=0.25, type=float, help="dis learning rate")
parser.add_argument("-glr", "--gen_learning_rate",
                    default=0.00005, type=float, help="gen learning rate")
parser.add_argument("-zlr", "--z_learning_rate",
                    default=0.00005, type=float, help="z learning rate")
parser.add_argument("-gp", "--gp_weight", default=10,
                    type=int, help="gradient penalty weight")
parser.add_argument("-bs", "--batch_size", type=int,
                    default=64, help="batch size")
parser.add_argument("-ep", "--epochs", type=int,
                    default=25, help="number of epochs")
parser.add_argument("-z", "--latent_size", type=int,
                    default=100, help="latent size")
parser.add_argument("-d", "--dropout", type=float,
                    default=0.2, help="daropout")
parser.add_argument("-cl", "--class_label", default=0,
                    type=int, help="normal/anomalous class label")
parser.add_argument("-ds", "--dataset", default="Leukemia",
                    help="which dataset the model is running on")
parser.add_argument(
    "-dz", "--distribution", default="Gaussian", help="choose the distribution you want to z selected from"
)

args = parser.parse_args()

normalize_CIFAR10 = torchvision.transforms.Normalize(
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform_CIFAR10 = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), normalize_CIFAR10])

normalize_MNIST = torchvision.transforms.Normalize((0.5,), (0.5,))
transform_MNIST = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(
        (32, 32)), torchvision.transforms.ToTensor(), normalize_MNIST]
)
normalize_FashionMNIST = torchvision.transforms.Normalize((0.5,), (0.5,))
transform_FashionMNIST = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(
        (32, 32)), torchvision.transforms.ToTensor(), normalize_FashionMNIST]
)
normalize_CatsVsDogs = torchvision.transforms.Normalize((0.5,), (0.5,))
transform_CatsVsDogs = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((64, 64)), torchvision.transforms.ToTensor()])

if args.dataset == "CIFAR10":
    train_CIFAR10 = torchvision.datasets.CIFAR10(
        "./data/CIFAR10/", train=True, transform=transform_CIFAR10, target_transform=None, download=args.download
    )
    test_CIFAR10 = torchvision.datasets.CIFAR10(
        "./data/CIFAR10/", train=False, transform=transform_CIFAR10, target_transform=None, download=args.download
    )

    idx = torch.tensor(train_CIFAR10.targets) != torch.tensor(args.class_label)
    dset_train = torch.utils.data.dataset.Subset(
        train_CIFAR10, np.where(idx == True)[0])
    dset_train_anomalous = torch.utils.data.dataset.Subset(
        train_CIFAR10, np.where(idx == False)[0])
    trainloader = torch.utils.data.DataLoader(
        dset_train, batch_size=32, shuffle=True)
    print("training data: {}".format(len(trainloader)), flush=True)

    idx = torch.tensor(test_CIFAR10.targets) != torch.tensor(args.class_label)
    dset_test = torch.utils.data.dataset.Subset(
        test_CIFAR10, np.where(idx == True)[0])
    dset_test_anomalous = torch.utils.data.dataset.Subset(
        test_CIFAR10, np.where(idx == False)[0])
    testloader_normal = torch.utils.data.DataLoader(
        dset_test, batch_size=1, shuffle=True)
    testloader_anomalous = torch.utils.data.ConcatDataset(
        [dset_train_anomalous, dset_test_anomalous])
    testloader_anomalous = torch.utils.data.DataLoader(
        testloader_anomalous, batch_size=1, shuffle=True)
    print("test data normal: {}, anomalous: {}".format(
        len(testloader_normal), len(testloader_anomalous)), flush=True)
elif args.dataset == "MNIST":
    train_MNIST = torchvision.datasets.MNIST(
        "./data/MNIST/", train=True, transform=transform_MNIST, target_transform=None, download=args.download
    )
    test_MNIST = torchvision.datasets.MNIST(
        "./data/MNIST/", train=False, transform=transform_MNIST, target_transform=None, download=args.download
    )

    idx = torch.as_tensor(
        train_MNIST.targets) != torch.tensor(args.class_label)
    dset_train = torch.utils.data.dataset.Subset(
        train_MNIST, np.where(idx == True)[0])
    dset_train_anomalous = torch.utils.data.dataset.Subset(
        train_MNIST, np.where(idx == False)[0])
    trainloader = torch.utils.data.DataLoader(
        dset_train, batch_size=32, drop_last=True, shuffle=True)

    idx = torch.as_tensor(test_MNIST.targets) != torch.tensor(args.class_label)
    dset_test = torch.utils.data.dataset.Subset(
        test_MNIST, np.where(idx == True)[0])
    dset_test_anomalous = torch.utils.data.dataset.Subset(
        test_MNIST, np.where(idx == False)[0])
    testloader_normal = torch.utils.data.DataLoader(
        dset_test, batch_size=1, drop_last=True, shuffle=True)
    testloader_anomalous = torch.utils.data.ConcatDataset(
        [dset_train_anomalous, dset_test_anomalous])
    testloader_anomalous = torch.utils.data.DataLoader(
        testloader_anomalous, batch_size=1, shuffle=True)

elif args.dataset == "FashionMNIST":
    train_FashionMNIST = torchvision.datasets.FashionMNIST(
        "./data/FashionMNIST/",
        train=True,
        transform=transform_FashionMNIST,
        target_transform=None,
        download=args.download,
    )
    test_FashionMNIST = torchvision.datasets.FashionMNIST(
        "./data/FashionMNIST/",
        train=False,
        transform=transform_FashionMNIST,
        target_transform=None,
        download=args.download,
    )

    idx = torch.as_tensor(
        train_FashionMNIST.targets) != torch.tensor(args.class_label)
    dset_train = torch.utils.data.dataset.Subset(
        train_FashionMNIST, np.where(idx == True)[0])
    dset_train_anomalous = torch.utils.data.dataset.Subset(
        train_FashionMNIST, np.where(idx == False)[0])
    trainloader = torch.utils.data.DataLoader(
        dset_train, batch_size=32, drop_last=True, shuffle=True)

    idx = torch.as_tensor(
        test_FashionMNIST.targets) != torch.tensor(args.class_label)
    dset_test = torch.utils.data.dataset.Subset(
        test_FashionMNIST, np.where(idx == True)[0])
    dset_test_anomalous = torch.utils.data.dataset.Subset(
        test_FashionMNIST, np.where(idx == False)[0])
    testloader_normal = torch.utils.data.DataLoader(
        dset_test, batch_size=1, drop_last=True, shuffle=True)
    testloader_anomalous = torch.utils.data.ConcatDataset(
        [dset_train_anomalous, dset_test_anomalous])
    testloader_anomalous = torch.utils.data.DataLoader(
        testloader_anomalous, batch_size=1, shuffle=True)

elif args.dataset == "CatsVsDogs":
    if args.class_label == 0:
        train_normal = "./data/CatsVsDogs/train/Cat"
        train_anomalous = "./data/CatsVsDogs/train/Dog"
    else:
        train_normal = "./data/CatsVsDogs/train/Dog"
        train_anomalous = "./data/CatsVsDogs/train/Cat"
    train_CatsVsDogs = torchvision.datasets.ImageFolder(
        train_normal, transform=transform_CatsVsDogs, target_transform=None
    )
    train_CatsVsDogs_anomalous = torchvision.datasets.ImageFolder(
        train_anomalous, transform=transform_CatsVsDogs, target_transform=None
    )
    test_CatsVsDogs = torchvision.datasets.ImageFolder(
        "./data/CatsVsDogs/test", transform=transform_CatsVsDogs, target_transform=None
    )

    trainloader = torch.utils.data.DataLoader(
        train_CatsVsDogs, batch_size=32, shuffle=True)
    testloader_normal = torch.utils.data.DataLoader(
        train_CatsVsDogs_anomalous, batch_size=1, shuffle=True)
    testloader_anomalous = torch.utils.data.DataLoader(
        test_CatsVsDogs, batch_size=1, shuffle=True)


seed = [0, 8]
torch.manual_seed(seed[0])
beta1 = 0.5
beta2 = 0.999

dis_criterion = nn.CrossEntropyLoss()
test_criterion = nn.MSELoss()

# dis_criterion = nn.BCEWithLogitsLoss() # for CGAN

if args.dataset == "MNIST" or args.dataset == "FashionMNIST":
    gen = ADGAN.Generator(args.latent_size, 1)
    dis = ADGAN.Discriminator(args.latent_size, 1)
elif args.dataset == "CIFAR10":
    gen = ADGAN.Generator(args.latent_size, 3)
    dis = ADGAN.Discriminator(args.latent_size, 3)
elif args.dataset == 'CatsVsDogs':
    gen = ADGAN.Generator_CD(args.latent_size, 3)
    dis = ADGAN.Discriminator_CD(args.latent_size, 3)


dis.to(device)
gen.to(device)


def gradient_penalty(critic, real_data, generated_data, gp_weight):
    cuda = torch.cuda.is_available()
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    if cuda:
        alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    if cuda:
        interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = critic(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).cuda(
        ) if cuda else torch.ones(prob_interpolated.size()),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean()


optimizer_d = optim.Adam(dis.parameters(), lr=args.dis_learning_rate, betas=(
    beta1, beta2), weight_decay=0.1)
optimizer_g = optim.Adam(gen.parameters(), lr=args.gen_learning_rate, betas=(
    beta1, beta2), weight_decay=0.5)

start_time = datetime.datetime.utcnow()

for epoch in range(args.epochs):
    dis.train()
    gen.train()
    for i, batch in enumerate(trainloader):
        x = batch[0].to(device)

        valid_label = Variable(torch.ones(
            (x.size(0)), dtype=torch.long), requires_grad=False).to(device)
        fake_label = Variable(torch.zeros(
            (x.size(0)), dtype=torch.long), requires_grad=False).to(device)

        # Generator
        if args.distribution == "Gaussian":
            z_random = torch.randn(
                x.shape[0], args.latent_size, 1, 1).to(device)
        elif args.distribution == "Uniform":
            z_random = (torch.rand(
                x.shape[0], args.latent_size, 1, 1) * 2 - 1).to(device)

        # Generator CGAN, ablation study to see the effect of contrastive loss on ADGAN
        # x_fake = gen(z_random)
        # dis_fake, feat_gl_f, feat_lc_f = dis(x_fake)
        # dis_real, feat_gl_r, feat_lc_r = dis(x)
        # optimizer_g.zero_grad()
        # loss_gen = - dis_fake.mean()
        # loss_gen.backward(retain_graph=True)
        # optimizer_g.step()

        # Generator of ADGAN
        optimizer_g.zero_grad()
        x_fake = gen(z_random)
        dis_fake = dis(x_fake)
        loss_gen = -dis_fake.mean()
        loss_gen.backward()
        optimizer_g.step()

        # Discriminator CGAN, ablation study to see the effect of contrastive loss on ADGAN
        # optimizer_d.zero_grad()
        # dis_fake, feat_gl_f, feat_lc_f = dis(x_fake.detach())
        # dis_real, feat_gl_r, feat_lc_r = dis(x)

        # errD = dis_criterion(dis_real - dis_fake, valid_label)
        # errD_IM = losses.compute_infomax_loss(feat_lc_r, feat_gl_r, 0.3)
        # loss_dis = errD + errD_IM
        # loss_dis.backward(retain_graph=True)
        # optimizer_d.step()

        # Discriminator
        optimizer_d.zero_grad()
        dis_real = dis(x)
        dis_fake = dis(x_fake.detach())
        loss_dis = dis_fake.mean() - dis_real.mean() + gradient_penalty(dis, x, x_fake, args.gp_weight)
        loss_dis.backward()
        optimizer_d.step()

    # print("epoch: %d, discriminator loss: %.2f, generator loss: %.2f" %(epoch, loss_dis.item(), loss_gen.item()))

# print("End of training", flush=True)
end_time = datetime.datetime.utcnow()
# print('Traing time: {}'.format((end_time - start_time).total_seconds()), flush=True)


# filepath = str(args.dataset) + str(args.epochs) + 'epochs' + str(args.distribution) + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# torch.save(gen.state_dict(), './models/' + filepath + 'G.pt')
# torch.save(dis.state_dict(), './models/' + filepath + 'D.pt')

score_neg = torch.zeros((len(testloader_normal), 1)).cuda()
score_pos = torch.zeros((len(testloader_anomalous), 1)).cuda()

if args.dataset == "MNIST" or args.dataset == "FashionMNIST":
    img_shape = (-1, 1, 32, 32)
elif args.dataset == "CIFAR10":
    img_shape = (-1, 3, 32, 32)
elif args.dataset == "CatsVsDogs":
    img_shape = (-1, 3, 64, 64)
c_neg = c_pos = 0
score = 0

intial_weights = [param for param in gen.parameters()]

start_time = datetime.datetime.utcnow()

for step, (images, labels) in enumerate(testloader_normal, 0):
    losses_neg = []
    images = images.view(img_shape).to(device)
    noise = torch.randn(
        (images.shape[0], args.latent_size, 1, 1), requires_grad=True, device=device)

    for name, layer in gen.named_modules():
        if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
            layer.eval()
        else:
            layer.train()
    optim_test = optim.Adam([noise], betas=(
        beta1, beta2), lr=args.z_learning_rate)

    for s in seed:
        torch.manual_seed(s)
        for i in range(5):
            optimizer_g.zero_grad()
            optim_test.zero_grad()
            sample = gen(noise)
            # loss_test = torch.norm(sample - images, 2)
            # loss_gen = torch.norm(sample - images, 2)
            loss_test = torch.mean(torch.square(sample - images))
            loss_gen = torch.mean(torch.square(sample - images))
            loss_gen.backward(retain_graph=True)
            loss_test.backward(retain_graph=True)
            optim_test.step()
            optimizer_g.step()
        losses_neg.append(loss_test.item())
    for param in gen.parameters():
        param = intial_weights
    score = -np.mean(losses_neg)
    score_neg[c_neg] = score
    c_neg += 1

for step, (images, labels) in enumerate(testloader_anomalous, 0):
    losses_pos = []
    images = images.view(img_shape).to(device)
    noise = torch.randn(
        (images.shape[0], args.latent_size, 1, 1), requires_grad=True, device=device)

    for name, layer in gen.named_modules():
        if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
            layer.eval()
        else:
            layer.train()

    optim_test = optim.Adam([noise], betas=(
        beta1, beta2), lr=args.z_learning_rate)

    for s in seed:
        torch.manual_seed(s)
        for i in range(5):
            optimizer_g.zero_grad()
            optim_test.zero_grad()
            sample = gen(noise)
            # loss_test = torch.norm(sample - images, 2)
            # loss_gen = torch.norm(sample - images, 2)
            loss_test = torch.mean(torch.square(sample - images))
            loss_gen = torch.mean(torch.square(sample - images))
            loss_gen.backward(retain_graph=True)
            loss_test.backward(retain_graph=True)
            optim_test.step()
            optimizer_g.step()
        losses_pos.append(loss_test.item())
    for param in gen.parameters():
        param = intial_weights
    score = -np.mean(losses_pos)
    score_pos[c_pos] = score
    c_pos += 1

x1 = score_neg.cpu().numpy()
x2 = score_pos.cpu().numpy()
data = {"Normal": x1, "Anomalous": x2}

FP = TP = []
neg_pre_wrong = 0
for i in range(len(score_neg)):
    if score_neg[i] > torch.mean(score_neg):
        neg_pre_wrong += 1

pos_pre_wrong = 0
for i in range(len(score_pos)):
    if score_pos[i] <= torch.mean(score_neg):
        pos_pre_wrong += 1
tp = len(score_pos) - pos_pre_wrong
fn = pos_pre_wrong
fp = neg_pre_wrong
tn = len(score_neg) - neg_pre_wrong
anomalous = torch.ones((len(score_pos), 1))
normal = torch.zeros((len(score_neg), 1))
y = torch.cat((anomalous, normal), 0)
scores = torch.cat((score_pos, score_neg), 0)
fpr, tpr, thresholds = metrics.roc_curve(y.cpu(), scores.cpu())
auc = metrics.auc(fpr, tpr)
print("AUC", auc, flush=True)

end_time = datetime.datetime.utcnow()
# print('inference time: {}'.format((end_time - start_time).total_seconds()), flush=True)
