import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import argparse

from plot import VisdomLinePlotter
from model import LSUNDiscriminator, LSUNGenerator, MNISTDiscriminator, MNISTGenerator

parser = argparse.ArgumentParser(description='WGAN implementation in PyTorch')

parser.add_argument('--path', type=str, default='data', help='directory for training data. only required for LSUN dataset')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate for RMSProp optimizer')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=250, help='number of epochs to run')
parser.add_argument('--noise_size', type=int, default=100, help='dimensionality of noise vector to use')
parser.add_argument('--critic_steps', type=int, default=5, help='number of discriminator steps per generator step')
parser.add_argument('--cutoff', type=float, default=0.01, help='gradient cutoff for WGAN clipping')
parser.add_argument('--image_size', type=int, default=64, help='size of training data to operate on')
parser.add_argument('--dataset', type=str, default='mnist', help='dataset to train on (mnist, lsun)')

parser.add_argument('--visdom', action='store_true', default=False, help='use visdom to plot training curve')
parser.add_argument('--visdom_port', type=int, default=8080, help='port for visdom plotting')

parser.add_argument('-plot', action='store_true', default=False, help='plot generator results after each epoch')

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

args = parser.parse_args()

if args.visdom:
    plotter = VisdomLinePlotter(port=args.visdom_port)
else:
    plotter = VisdomLinePlotter(disable=True)

if args.dataset == 'mnist':
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])

    mnist_trainset = datasets.MNIST(root='./mnist-data', train=True, download=True, transform=trans)
    mnist_testset = datasets.MNIST(root='./mnist-data', train=False, download=False, transform=trans)

    train_loader = torch.utils.data.DataLoader(
        dataset=mnist_trainset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)

    test_loader = torch.utils.data.DataLoader(
        dataset=mnist_testset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)

    generator = MNISTGenerator(args.noise_size).to(device)
    discriminator = MNISTDiscriminator().to(device)

elif args.dataset == 'lsun':
    lsun_trainset = torchvision.datasets.ImageFolder(args.path, transform=transforms.Compose(
        [transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size), transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(
        dataset=lsun_trainset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)

    generator = LSUNGenerator(args.noise_size).to(device)
    discriminator = LSUNDiscriminator().to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    generator = nn.DataParallel(generator).to(device)
    discriminator = nn.DataParallel(discriminator).to(device)

optim_g = optim.RMSprop(generator.parameters(), lr=args.lr)
optim_d = optim.RMSprop(discriminator.parameters(), lr=args.lr)

def sample_noise(batch_size, *channels):
    return torch.randn(batch_size, *channels).float()

def clip_weights(parameters, value):  # bad
    for param in parameters:
        param.data.clamp_(-value, value)

plotter.clear()

for epoch in range(args.epochs):
    generator.train()
    discriminator.train()

    for i, (real, labels) in enumerate(train_loader):
        noise = sample_noise(args.batch_size, args.noise_size, 1, 1).to(device)
        real = real.to(device)

        fake = generator(noise)
        fake_discr = discriminator(fake)
        real_discr = discriminator(real)
        loss = fake_discr.mean() - real_discr.mean()

        if i % 100 == 0:
            print("[INFO] [EPOCH {}] [SAMPLE {}] Wasserstein loss is {}".format(epoch, i, -loss))
            plotter.plot("loss", "wasserstein", "Loss for Wasserstein GAN", len(
                train_loader) * epoch + i, -float(loss.detach().cpu()), xlabel="iterations")

        epsilon = torch.FloatTensor(args.batch_size, 1, 1, 1).uniform_(0, 1).to(device)
        xhat = epsilon * fake + (1 - epsilon) * real

        xhat_discr = discriminator(xhat)
        grads = torch.autograd.grad(xhat_discr, xhat, grad_outputs=torch.ones_like(xhat_discr), create_graph=True)[0]

        penalty = ((torch.sqrt((grads ** 2).sum(dim=(1, 2, 3))) - 1) ** 2).mean()
        loss = loss + 10 * penalty

        optim_d.zero_grad()
        loss.backward()
        optim_d.step()

        if i % args.critic_steps == 0:
            noise = sample_noise(args.batch_size, args.noise_size, 1, 1).to(device)
            loss = - discriminator(generator(noise)).mean()
            optim_g.zero_grad()
            loss.backward()

            optim_g.step()

            if i % (10 * args.critic_steps) == 0:
                print("[INFO] [EPOCH {}] [SAMPLE {}] Generator loss is {}".format(epoch, i, loss))
                plotter.plot("loss", "generator", "Loss for Wasserstein GAN", len(
                    train_loader) * epoch + i, float(loss.detach().cpu()), xlabel="iterations")

    if args.plot:
        generator.eval()

        noise = sample_noise(args.batch_size, args.noise_size, 1, 1).to(device)
        
        fake = generator(noise)[0].squeeze().detach().cpu()
        real = real[0].squeeze().cpu()

        if fake.dim() == 3:
            fake, ral = fake.permute(1, 2, 0), real.permute(1, 2, 0)

        plt.imshow(fake)
        plt.show()
        plt.imshow(real)
        plt.show()
