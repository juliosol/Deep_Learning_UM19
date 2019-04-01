import torch
import numpy as np
from math import sqrt
import torch.nn as nn
import torch.utils.data as data_utils
import time
import pickle
import bcolz



def loader(dictionary):

    print("Loading Train data")
    X_train = []
    y_train = []


    f = open('data_p2/train.txt')
    for l in f:
        y_train.append(int(l[0]))
        line = l[2:].split()
        temp = []
        count = 0
        for item in line:

            if item in dictionary:
                temp.append(dictionary[item])
                count += 1
            if count == 15:
                break

        while count < 15:
            for item in line:

                if item in dictionary:
                    temp.append(dictionary[item])
                    count += 1
                if count == 15:
                    break

        X_train.append(temp)


    y_train = np.asarray(y_train).reshape(-1,1)



    print("Loading Test data")
    X_test = []
    y_test = []

    f = open('data_p2/test.txt')
    for l in f:
        y_test.append(int(l[0]))
        line = l[2:].split()
        temp = []
        count = 0

        for item in line:

            if item in dictionary:
                temp.append(dictionary[item])
                count += 1
            if count == 15:
                break

        while count < 15:
            for item in line:
                if item in dictionary:
                    temp.append(dictionary[item])
                    count += 1
                if count == 15:
                    break

        X_test.append(temp)


    y_test = np.asarray(y_test).reshape(-1,1)



    print("Loading Unlabelled data")
    X_unlabelled = []


    f = open('data_p2/unlabelled.txt')
    for l in f:
        line = l[2:].split()
        temp = []
        count = 0
        for item in line:

            if item in dictionary:
                temp.append(dictionary[item])
                count += 1
            if count == 15:
                break

        while count < 15:
            for item in line:

                if item in dictionary:
                    temp.append(dictionary[item])
                    count += 1
                if count == 15:
                    break

        X_unlabelled.append(temp)

    return X_train, y_train, X_test, y_test, X_unlabelled
    

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = np.shape(weights_matrix)
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    type({'weight': weights_matrix})
    emb_layer.from_pretrained(torch.FloatTensor(weights_matrix))
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class q6_4(nn.Module):
    def __init__(self, weights_matrix):
        super().__init__()
        F = 128
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        #print("This si num_embeddings",num_embeddings)
        #print("This is embedding_dim",  embedding_dim)
        self.embed = nn.Embedding(num_embeddings,embedding_dim)
        #self.cnn = nn.Conv1d(1, F, kernel_size=5)
        self.cnn = nn.Conv1d(embedding_dim, F,  kernel_size=5)
        #self.max_avg = nn.AvgPool2d(100)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(F,1)
        
    def init_weights(self):
        
        C_in = self.fc.weight.size(1)
        nn.init.normal_(self.fc.weight, 0.0, 1 / sqrt(C_in))
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        F = 128
        N, H = np.shape(x)
        #print("Shape before network", np.shape(x))
        z = self.embed(x.long())
        #print("Shape after embedding",np.shape(z))
        z = torch.transpose(z, 1, 2)
        #print("Shape of transpose of data", np.shape(z))
        
        z = self.cnn(z)
        #print("This is cnn dimension", np.shape(z))
        #z = z.reshape(F,N,-1)
        #print("This is after reshaping", np.shape(z))

        # Global avg pool
        #print(np.shape(z.view(z.size()[0],-1)))
        #z = z.view(z.size()[0], -1)
        #z = self.max_avg(z)


        list_avg = []
        for n in range(np.shape(z)[0]):
            curr_tensor = z[n,:,:]
            #print("This is shape of curr tensor", np.shape(curr_tensor))
            #curr_tensor = torch.mean(curr_tensor, 1)
            curr_tensor = torch.max(curr_tensor, 1)[0]
            #print(curr_tensor)
            #print("Shape after max", np.shape(curr_tensor))
            list_avg.append(curr_tensor)
        z = torch.stack(list_avg, 0)


        #z = torch.sum(z,2)/np.shape(z)[2]
        #print("After pooling", np.shape(z))
        # Global max pool
        #z = z.view(z.size()[0], -1)#.view(-1,1)
        #print(np.shape(z))
        #z = torch.max(z, 2)[0].view(-1,1)
        
        #print("This is shape after pool", np.shape(z))
        #z = z.reshape(1,-1)
        z = self.relu(z)
        #print("This is shape of relu", np.shape(z))
        z = self.fc(z)
        #print("This is z", np.shape(z))
        h = torch.sigmoid(z).view(np.shape(z)[0],-1)
        #print("This is h", np.shape(h))
        #import sys
        #sys.exit()
        return h

def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(25):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (representations, labels) in enumerate(trainloader):
            representations = representations.to(device).float()
            labels = labels.to(device).float()

            optimizer.zero_grad()
            output = net(representations)
            #print(np.shape(labels))
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end - start))
                start = time.time()
                running_loss = 0.0
    print('Finished Training')




def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            representations, labels = data
            representations = representations.to(device).float()
            labels = labels.to(device).float()
            outputs = net(representations)
            outputs[outputs < 0.5] = 0
            outputs[outputs >= 0.5] = 1
            total += labels.size(0)
            correct += (outputs == labels).sum().item()
    print('Accuracy: %d %%' % (
        100 * correct / total))



def main():
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    dictionary = {}
    index = 0
    with open('data_p2/train.txt', 'r') as f:
        for l in f:
            line = l[2:].split()
            for item in line:
                if item not in dictionary:
                    dictionary[item] = index
                    index += 1


    X_train, y_train, X_test, y_test, X_unlabelled = loader(dictionary)

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    X_unlabelled = torch.tensor(X_unlabelled)

    trainset = data_utils.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True)

    testset = data_utils.TensorDataset(X_test, y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False)

    unlabelledset = data_utils.TensorDataset(X_unlabelled)
    unlabelledloader = data_utils.DataLoader(unlabelledset, batch_size=100, shuffle=False)




    # The following usage of GloVe follows the tutorial from https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    words = []
    idx = 0
    word2idx = {}
    glove_path = 'D:/Julio/Documents/Michigan_v2/CS/EECS_598_Deep_Learning/HW/HW3/Homework3/code/glove.6B'
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

    with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400001, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))

    vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    matrix_len = len(dictionary)
    weights_matrix = np.zeros((matrix_len, 50))
    words_found = 0

    for i, word in enumerate(dictionary):
        try:
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(50,))

    net = q6_4(weights_matrix).to(device)
    net.init_weights()
    criterion = nn.BCELoss()
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.8)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)#, momentum=0.8)

    train(trainloader, net, criterion, optimizer, device)
    test(testloader, net, device)

    f = open('output/predictions_p1.txt', 'w')

    for data in unlabelledloader:
        info, = data
        output = net(info.to(device).float())
        output[output < 0.5] = int(0)
        output[output >= 0.5] = int(1)
        for item in output:
            f.write(str(int(item.item())))
            f.write("\n")
    f.close()

if __name__ == "__main__":
    main()

