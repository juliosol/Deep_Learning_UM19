from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image    


class CVAE(nn.Module):
    def __init__(self, input_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.class_size = class_size
        self.latent_size = latent_size
        self.units = 400


        ######################################################
        ###              START OF YOUR CODE                ###
        ######################################################
        ### Define a three layer neural network architecture #
        ### for the recognition_model                        #
        ######################################################

        self.recon_network = nn.Sequential(
                        nn.Linear(self.input_size + self.class_size, self.units),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.units, self.units),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.units, self.units),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.units, self.units),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.units, latent_size * 2)
                        )

        ######################################################
        ###               END OF YOUR CODE                 ###
        ######################################################




        ######################################################
        ###              START OF YOUR CODE                ###
        ######################################################
        ### Define a three layer neural network architecture #
        ### for the generation_model                         #
        ######################################################

        self.gen_network = nn.Sequential(
                        nn.Linear(self.latent_size + self.class_size, self.units),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.units, self.units),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.units, self.units),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.units, self.units),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.units, input_size),
                        nn.Sigmoid()
                        )

        ######################################################
        ###               END OF YOUR CODE                 ###
        ######################################################



    def recognition_model(self, x, c):
        """
        Computes the parameters of the posterior distribution q(z | x, c) using the
        recognition network defined in the constructor
    
        Inputs:
        - x: PyTorch Variable of shape (batch_size, input_size) for the input data
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class
        
        Returns:
        - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
        - logvar PyTorch Variable of shape (batch_size, latent_size) for the posterior
          variance in log space
        """
        ###########################
        ######### TO DO ###########

        concatenated_data = torch.cat((x,c), 1)
        data = self.recon_network(concatenated_data)
        mu = data[:,: self.latent_size]
        logvar = data[:,self.latent_size:]

        ###########################
        #mu = None
        #logvar = None
        return mu, logvar


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std) + mu

    def generation_model(self, z, c): # P(x|z, c)
        """
        Computes the generation output from the generative distribution p(x | z, c)
        using the generation network defined in the constructor
    
        Inputs:
        - z: PyTorch Variable of shape (batch_size, latent_size) for the latent vector
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class
        
        Returns:
        - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
        """
        ###########################
        ######### TO DO ###########

        concatenated_data = torch.cat((z,c), 1)
        data = self.gen_network(concatenated_data)
        x_hat = data
        
        ###########################
        #x_hat = None
        return x_hat

    def forward(self, x, c):
        """
        Performs the inference and generation steps of the CVAE model using
        the recognition_model, reparametrization trick, and generation_model
    
        Inputs:
        - x: PyTorch Variable of shape (batch_size, input_size) for the input data
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class
        
        Returns:
        - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
        - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
        - logvar: PyTorch Variable of shape (batch_size, latent_size)
                  for the posterior logvar
        """
        ###########################
        ######### TO DO ###########

        mu, logvar = self.recognition_model(x,c)
        n,m = logvar.size()
        z = self.reparametrize(mu, logvar)
        x_hat = self.generation_model(z,c)

        ###########################
        #x_hat = None
        #mu = None
        #logvar = None
        return x_hat, mu, logvar


def to_var(x, use_cuda):
    x = Variable(x)
    if use_cuda:
        x = x.cuda()
    return x


def one_hot(labels, class_size, use_cuda):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return to_var(targets, use_cuda)


def train(epoch, model, train_loader, optimizer, num_classes, use_cuda):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = to_var(data, use_cuda).view(data.shape[0], -1)
        labels = one_hot(labels, num_classes, use_cuda)
        recon_batch, mu, logvar = model(data, labels)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data / len(data)))


def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lowerbound for conditional vae
    Note: We compute -lowerbound because we optimize the network by minimizing a loss

    Inputs:
    - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
    - x: PyTorch Variable of shape (batch_size, input_size) for the real data
    - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
    - logvar: PyTorch Variable of shape (batch_size, latent_size) for the posterior logvar
    
    Returns:
    - loss: PyTorch Variable containing the (scalar) loss for the negative lowerbound.
    """
    ###########################
    ######### TO DO ###########
    bach_size, input_size = x_hat.size()


    #BCE = F.binary_cross_entropy_with_logits(x_hat, x, size_average=False)
    BCE = F.binary_cross_entropy(x_hat.view(-1, 28*28), x.view(-1, 28*28), reduction='sum') 
    KL = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = (BCE + KL)/x.shape[0]
    
    ###########################
    #loss = None
    return loss


def main():
    # Load MNIST dataset
    #use_cuda = False
    use_cuda = True
    input_size = 28 * 28
    units = 400
    batch_size = 32
    latent_size = 15 # z dim
    num_classes = 10
    num_epochs = 30
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)

    model = CVAE(input_size, latent_size, num_classes)

    if use_cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, num_epochs):
        train(epoch, model, train_loader, optimizer, num_classes, use_cuda)

    print("Training is finished.")
    
    # Generate images with condition labels
    print("Why is this not printing?")
    c = torch.eye(num_classes, num_classes) # [one hot labels for 0-9]
    c = to_var(c, use_cuda)
    z = to_var(torch.randn(num_classes, latent_size), use_cuda)
    samples = model.generation_model(z, c).data.cpu().numpy()
    print("Why is this not printing?")

    print("Why is this not printing?")
    fig = plt.figure(figsize=(10, 1))
    gs = gridspec.GridSpec(1, 10)
    gs.update(wspace=0.05, hspace=0.05)
    print("Why is this not printing?")
    for i, sample in enumerate(samples):
        print(i)
        print(sample)
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    print("Why is this not printing?")
    plt.show()
    print("Finished images")
    fig.savefig("p3_final_image.png")


if __name__ == "__main__":
  main()

