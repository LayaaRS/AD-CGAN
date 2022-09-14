import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable

import datetime
import numpy as np
import argparse
import datetime
from sklearn import metrics

from utils import dataset
from modules.ALAD import ALAD as model


device = torch.device("cuda")

parser = argparse.ArgumentParser()

parser.add_argument("-tr", "--trainset_path",
                    default="./data/Leukemia/", help="path to the train data")
parser.add_argument("-ts", "--testset_path",
                    default="./data/Leukemia/", help="path to the tets data")
parser.add_argument("-dl", "--download", default="False",
                    help="download the datasets")
parser.add_argument("-rs", "--random_seed", default=0,
                    type=int, help="random seed")
parser.add_argument("-lr", "--learning_rate", type=float,
                    default=1e-4, help="learning rate")
parser.add_argument("-bs", "--batch_size", type=int,
                    default=64, help="batch size")
parser.add_argument("-ep", "--epochs", type=int,
                    default=10, help="number of epochs")
parser.add_argument("-z", "--latent_size", type=int,
                    default=15, help="latent size")
parser.add_argument("-d", "--dropout", type=float,
                    default=0.2, help="daropout")
parser.add_argument("-ds", "--dataset", default="Leukemia",
                    help="which dataset the model is running on")
parser.add_argument("-sm", "--save_model_dir", default="./models/")
parser.add_argument(
    "-cl",
    "--class_label_anomalous",
    type=int,
    default=0,
    help="select which class label should be selected as the anomalous",
)
parser.add_argument(
    "-dz", "--distribution", default="Gaussian", help="choose the distribution you want to z selected from"
)

args = parser.parse_args()

if args.dataset == "CIFAR10":
    normalize_CIFAR10 = torchvision.transforms.Normalize((0.5,), (0.5,))
    transform_CIFAR10 = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop(38),
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
        ]
    )
elif args.dataset == "MNIST":
    normalize_MNIST = torchvision.transforms.Normalize((0.5,), (0.5,))
    transform_MNIST = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop(36),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
        ]
    )
elif args.dataset == "FashionMNIST":
    normalize_FashionMNIST = torchvision.transforms.Normalize((0.5,), (0.5,))
    transform_FashionMNIST = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop(36),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
        ]
    )

elif args.dataset == "CatsVsDogs":
    normalize_CatsVsDogs = torchvision.transforms.Normalize((0.5,), (0.5,))
    transform_CatsVsDogs = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(
            (64, 64)), torchvision.transforms.ToTensor()]
    )

if args.dataset == "MNIST":
    train_MNIST = torchvision.datasets.MNIST(
        "./data/MNIST/", train=True, transform=transform_MNIST, target_transform=None, download=args.download
    )
    test_MNIST = torchvision.datasets.MNIST(
        "./data/MNIST/", train=False, transform=transform_MNIST, target_transform=None, download=args.download
    )
    idx = torch.as_tensor(train_MNIST.targets) != torch.tensor(
        args.class_label_anomalous)
    dset_train = torch.utils.data.dataset.Subset(
        train_MNIST, np.where(idx == True)[0])
    dset_train_anomalous = torch.utils.data.dataset.Subset(
        train_MNIST, np.where(idx == False)[0])
    idx = torch.as_tensor(test_MNIST.targets) != torch.tensor(
        args.class_label_anomalous)
    dset_test = torch.utils.data.dataset.Subset(
        test_MNIST, np.where(idx == True)[0])
    dset_test_anomalous = torch.utils.data.dataset.Subset(
        test_MNIST, np.where(idx == False)[0])
    trainloader = torch.utils.data.DataLoader(
        dset_train, batch_size=args.batch_size, drop_last=True, shuffle=True)
    testloader_normal = torch.utils.data.DataLoader(
        dset_test, batch_size=1, drop_last=True, shuffle=True)
    testloader_anomalous = torch.utils.data.ConcatDataset(
        [dset_train_anomalous, dset_test_anomalous])
    print(
        "MNIST train: ",
        len(trainloader),
        "MNIST test normal:",
        len(testloader_normal),
        "MNIST test anomalous: ",
        len(testloader_anomalous),
    )
elif args.dataset == "CIFAR10":
    train_CIFAR10 = torchvision.datasets.CIFAR10(
        "./data/CIFAR10/", train=True, transform=transform_CIFAR10, target_transform=None, download=args.download
    )
    test_CIFAR10 = torchvision.datasets.CIFAR10(
        "./data/CIFAR10/", train=False, transform=transform_CIFAR10, target_transform=None, download=args.download
    )
    idx = torch.tensor(train_CIFAR10.targets) == torch.tensor(
        args.class_label_anomalous)
    dset_train = torch.utils.data.dataset.Subset(
        train_CIFAR10, np.where(idx == True)[0])
    dset_train_anomalous = torch.utils.data.dataset.Subset(
        train_CIFAR10, np.where(idx == False)[0])
    idx = torch.tensor(test_CIFAR10.targets) == torch.tensor(
        args.class_label_anomalous)
    dset_test = torch.utils.data.dataset.Subset(
        test_CIFAR10, np.where(idx == True)[0])
    dset_test_anomalous = torch.utils.data.dataset.Subset(
        test_CIFAR10, np.where(idx == False)[0])
    trainloader = torch.utils.data.DataLoader(
        dset_train, batch_size=args.batch_size, drop_last=True, shuffle=True)
    testloader_normal = torch.utils.data.DataLoader(
        dset_test, batch_size=1, drop_last=True, shuffle=True)
    testloader_anomalous = torch.utils.data.ConcatDataset(
        [dset_train_anomalous, dset_test_anomalous])
    print(
        "CIFAR10 train: ",
        len(trainloader),
        "CIFAR10 test normal: ",
        len(testloader_normal),
        "CIFAR10 test anomalous: ",
        len(testloader_anomalous),
    )
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
        train_FashionMNIST.targets) == torch.tensor(args.class_label)
    dset_train = torch.utils.data.dataset.Subset(
        train_FashionMNIST, np.where(idx == True)[0])
    dset_train_anomalous = torch.utils.data.dataset.Subset(
        train_FashionMNIST, np.where(idx == False)[0])
    trainloader = torch.utils.data.DataLoader(
        dset_train, batch_size=32, drop_last=True, shuffle=True)
    print("training data: {}".format(len(trainloader)), flush=True)

    idx = torch.as_tensor(
        test_FashionMNIST.targets) == torch.tensor(args.class_label)
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
    print("test data normal: {}, anomalous: {}".format(
        len(testloader_normal), len(testloader_anomalous)), flush=True)
elif args.dataset == "CatsVsDogs":
    if args.class_label_anomalous == 0:  # zero for Cats, 1 for Dogs
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
    print("training data: {}".format(len(trainloader)), flush=True)
    print("test data normal: {}, anomalous: {}".format(
        len(testloader_normal), len(testloader_anomalous)), flush=True)


torch.manual_seed(args.random_seed)
beta1 = 0.5
beta2 = 0.999
acc_lam = {}

dis_BCElogit_criterion = nn.BCEWithLogitsLoss()
dis_criterion = nn.CrossEntropyLoss()
aen_criterion = nn.MSELoss()

if args.dataset == "CIFAR10":
    enc = model.Encoder_CIFAR10(args.latent_size)
    gen = model.Generator_CIFAR10(args.latent_size)
    dis_xz = model.Discriminator_xz_CIFAR10(args.latent_size, 0.2)
    dis_xx = model.Discriminator_xx_CIFAR10(args.latent_size, 0.2)
    dis_zz = model.Discriminator_zz_CIFAR10(args.latent_size, 0.2)
elif args.dataset == 'MNIST' or args.dataset == 'FashionMNIST':
    enc = model.Encoder_MNIST(args.latent_size)
    gen = model.Generator_MNIST(args.latent_size)
    dis_xz = model.Discriminator_xz_MNIST(args.latent_size, 0.2)
    dis_xx = model.Discriminator_xx_MNIST(args.latent_size, 0.2)
    dis_zz = model.Discriminator_zz_MNIST(args.latent_size, 0.2)
elif args.dataset == "CatsVsDogs":
    enc = model.Encoder_CD(args.latent_size, 3)
    gen = model.Generator_CD(args.latent_size, 3)
    dis_xz = model.Discriminator_xz_CD(args.latent_size, 0.2)
    dis_xx = model.Discriminator_xx_CD(args.latent_size, 0.2)
    dis_zz = model.Discriminator_zz_CD(args.latent_size, 0.2)

dis_xz.to(device)
dis_xx.to(device)
dis_zz.to(device)
enc.to(device)
gen.to(device)


optimizer_dxz = optim.Adam(dis_xz.parameters(
), lr=args.learning_rate, betas=(beta1, beta2), weight_decay=0.5)
optimizer_dxx = optim.Adam(dis_xx.parameters(
), lr=args.learning_rate, betas=(beta1, beta2), weight_decay=0.5)
optimizer_dzz = optim.Adam(dis_zz.parameters(
), lr=args.learning_rate, betas=(beta1, beta2), weight_decay=0.5)
optimizer_g = optim.Adam(gen.parameters(), lr=args.learning_rate, betas=(
    beta1, beta2), weight_decay=0.5)
optimizer_e = optim.Adam(enc.parameters(), lr=args.learning_rate, betas=(
    beta1, beta2), weight_decay=1e-3)

start_time = datetime.datetime.utcnow()

print("training starts", flush=True)
for epoch in range(args.epochs):
    dis_xz.train()
    dis_xx.train()
    dis_zz.train()
    enc.train()
    gen.train()

    for i, batch in enumerate(trainloader):
        x = batch[0].to(device)

        valid_label = Variable(torch.ones(
            (x.size(0)), dtype=torch.long), requires_grad=False).to(device)
        fake_label = Variable(torch.zeros(
            (x.size(0)), dtype=torch.long), requires_grad=False).to(device)

        if args.distribution == "Gaussian":
            z_random = torch.randn(
                x.shape[0], args.latent_size, 1, 1).to(device)
        elif args.distribution == "Uniform":
            z_random = (torch.rand(
                x.shape[0], args.latent_size, 1, 1) * 2 - 1).to(device)

        z_gen = enc(x)
        x_gen = gen(z_random)
        rec_x = gen(z_gen)
        rec_z = enc(x_gen)

        # discriminator xz
        optimizer_dxz.zero_grad()
        l_encoder, inter_layer_inp_xz = dis_xz(x, z_gen)
        l_generator, inter_layer_rct_xz = dis_xz(x_gen, z_random)

        # discriminator xx
        optimizer_dxx.zero_grad()
        x_logit_real, inter_layer_inp_xx = dis_xx(x, x)
        x_logit_fake, inter_layer_rct_xx = dis_xx(x, rec_x)

        # discriminator zz
        optimizer_dzz.zero_grad()
        z_logit_real, _ = dis_zz(z_random, z_random)
        z_logit_fake, _ = dis_zz(z_random, rec_z)

        loss_dis_enc = torch.mean(
            dis_BCElogit_criterion(Variable(torch.ones_like(
                l_encoder), requires_grad=True), l_encoder)
        )
        loss_dis_gen = torch.mean(
            dis_BCElogit_criterion(Variable(torch.zeros_like(
                l_generator), requires_grad=True), l_generator)
        )
        dis_loss_xz = loss_dis_gen + loss_dis_enc

        x_real_dis = dis_BCElogit_criterion(x_logit_real, Variable(
            torch.ones_like(x_logit_real), requires_grad=True))
        x_fake_dis = dis_BCElogit_criterion(x_logit_fake, Variable(
            torch.zeros_like(x_logit_fake), requires_grad=True))
        dis_loss_xx = torch.mean(x_real_dis + x_fake_dis)

        z_real_dis = dis_BCElogit_criterion(z_logit_real, Variable(
            torch.ones_like(z_logit_real), requires_grad=True))
        z_fake_dis = dis_BCElogit_criterion(z_logit_fake, Variable(
            torch.zeros_like(z_logit_fake), requires_grad=True))
        dis_loss_zz = torch.mean(z_real_dis + z_fake_dis)

        loss_discriminator = dis_loss_xz + dis_loss_xx + dis_loss_zz
        loss_discriminator.backward(retain_graph=True)
        optimizer_dxz.step()
        optimizer_dxx.step()
        optimizer_dzz.step()

        ### generator and encoder
        optimizer_e.zero_grad()
        optimizer_g.zero_grad()

        gen_loss_xz = torch.mean(
            dis_BCElogit_criterion(Variable(torch.ones_like(
                l_generator), requires_grad=True), l_generator)
        )
        enc_loss_xz = torch.mean(
            dis_BCElogit_criterion(Variable(torch.zeros_like(
                l_encoder), requires_grad=True), l_encoder)
        )
        x_real_gen = dis_BCElogit_criterion(x_logit_real, Variable(
            torch.zeros_like(x_logit_real), requires_grad=True))
        x_fake_gen = dis_BCElogit_criterion(x_logit_fake, Variable(
            torch.ones_like(x_logit_real), requires_grad=True))
        z_real_gen = dis_BCElogit_criterion(z_logit_real, Variable(
            torch.zeros_like(z_logit_real), requires_grad=True))
        z_fake_gen = dis_BCElogit_criterion(z_logit_fake, Variable(
            torch.ones_like(z_logit_fake), requires_grad=True))

        cost_x = torch.mean(x_real_gen + x_fake_gen)
        cost_z = torch.mean(z_real_gen + z_fake_gen)

        cycle_consistency_loss = cost_x + cost_z
        loss_generator = gen_loss_xz + cycle_consistency_loss
        loss_encoder = enc_loss_xz + cycle_consistency_loss

        loss_generator.backward(retain_graph=True)
        optimizer_g.step()

        loss_encoder.backward()
        optimizer_e.step()

        # if epoch % 10 == 0 and epoch >= args.encoder_first_train:
        #   vutils.save_image(x_fake.cpu().data[:16, ], '%s/fake_%d.png' % (args.save_image_dir, epoch))

    # if epoch > 10 and epoch % 100 == 0:
    #   filepath = str(args.save_model_dir) + str(args.dataset) + str(args.epochs) + 'epochs' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #   torch.save(gen.state_dict(), filepath + 'G.pt')
    #   torch.save(enc.state_dict(), filepath + 'E.pt')
    #   torch.save(dis.state_dict(), filepath + 'D.pt')

    print(
        "epoch: %d, iteration: %d, discriminator loss: %.2f, generator loss: %.2f, encoder loss: %.2f"
        % (epoch, i, loss_discriminator.item(), loss_generator.item(), loss_encoder.item()),
        flush=True,
    )

print("End of training", flush=True)
end_time = datetime.datetime.utcnow()
print('Traing time: {}'.format((end_time - start_time).total_seconds()), flush=True)

# save model
filepath = (
    str(args.save_model_dir)
    + str(args.dataset)
    + str(args.epochs)
    + "epochs"
    + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
)
# torch.save(gen.state_dict(), filepath + 'G')
# torch.save(enc.state_dict(), filepath + 'E')
# torch.save(dis_xz.state_dict(), filepath + 'D_xz')
# torch.save(dis_xx.state_dict(), filepath + 'D_xx')
# torch.save(dis_zz.state_dict(), filepath + 'D_zz')

if args.dataset == "MNIST" or args.dataset == "FashionMNIST":
    img_shape = (-1, 1, 28, 28)
elif args.dataset == "CIFAR10":
    img_shape = (-1, 3, 32, 32)
elif args.dataset == "CatsVsDogs":
    img_shape = (-1, 3, 64, 64)
lam = 0.1

loss_neg = torch.zeros((len(testloader_normal), 1)).cuda()
loss_pos = torch.zeros((len(testloader_anomalous), 1)).cuda()
c_neg = c_pos = 0

start_time = datetime.datetime.utcnow()

for step, (images, labels) in enumerate(testloader_normal, 0):
    images = images.view(img_shape)
    dis_xz.eval()
    dis_xx.eval()
    dis_zz.eval()
    enc.eval()
    gen.eval()
    x_real_test = images.cuda()
    z_random = torch.randn(images.shape[0], args.latent_size, 1, 1).cuda()
    z_gen = enc(x_real_test)
    x_gen = gen(z_random)
    rec_x = gen(z_gen)
    rec_z = enc(x_gen)

    l_gen, _ = dis_xz(x_real_test, z_gen)
    l_enc, _ = dis_xz(x_gen, z_random)

    x_logit_real, inter_layer_inp = dis_xx(x_real_test, x_real_test)
    x_logit_fake, inter_layer_rct = dis_xx(x_real_test, rec_x)

    fm = inter_layer_inp - inter_layer_rct
    feature_loss = torch.norm(fm, 1, keepdim=False)
    feature_loss = feature_loss.squeeze()

    loss_neg[c_neg] = feature_loss.detach()
    c_neg += 1


for step, (images, labels) in enumerate(testloader_anomalous, 0):
    images = images.view(img_shape)
    dis_xz.eval()
    dis_xx.eval()
    dis_zz.eval()
    enc.eval()
    gen.eval()
    x_real_test = images.cuda()
    z_random = torch.randn(images.shape[0], args.latent_size, 1, 1).cuda()
    z_gen = enc(x_real_test)
    x_gen = gen(z_random)
    rec_x = gen(z_gen)
    rec_z = enc(x_gen)

    l_gen, _ = dis_xz(x_real_test, z_gen)
    l_enc, _ = dis_xz(x_gen, z_random)

    x_logit_real, inter_layer_inp = dis_xx(x_real_test, x_real_test)
    x_logit_fake, inter_layer_rct = dis_xx(x_real_test, rec_x)

    fm = inter_layer_inp - inter_layer_rct
    feature_loss = torch.norm(fm, 1, keepdim=False)
    feature_loss = feature_loss.squeeze()

    loss_pos[c_pos] = feature_loss.detach()
    c_pos += 1

print("mean negative: %0.4f, std negative: %0.4f" %
      (torch.mean(loss_neg), torch.std(loss_neg)), flush=True)
print("mean positive: %0.4f, std positive: %0.4f" %
      (torch.mean(loss_pos), torch.std(loss_pos)), flush=True)

x1 = loss_neg.cpu().numpy()
x2 = loss_pos.cpu().numpy()

FP = TP = []
neg_pre_wrong = 0
for i in range(len(loss_neg)):
    if loss_neg[i] > torch.mean(loss_neg):
        neg_pre_wrong += 1

pos_pre_wrong = 0
for i in range(len(loss_pos)):
    if loss_pos[i] <= torch.mean(loss_neg):
        pos_pre_wrong += 1
tp = len(loss_pos) - pos_pre_wrong
fn = pos_pre_wrong
fp = neg_pre_wrong
tn = len(loss_neg) - neg_pre_wrong
anomalous = torch.ones((len(loss_pos), 1))
normal = torch.zeros((len(loss_neg), 1))
y = torch.cat((anomalous, normal), 0)
scores = torch.cat((loss_pos, loss_neg), 0)
fpr, tpr, thresholds = metrics.roc_curve(y.cpu(), scores.cpu())
auc = metrics.auc(fpr, tpr)
print("AUC", auc, flush=True)

end_time = datetime.datetime.utcnow()
print('inference time: {}'.format(
    (end_time - start_time).total_seconds()), flush=True)
