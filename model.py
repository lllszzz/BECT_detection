import numpy as np
import torch
import dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model
class stackedbiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(stackedbiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc1= nn.Linear(hidden_size*2, 64)
        self.fc2= nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=2)

    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # out shape: (batch_size, seq_length, hidden_size*2)
        # h_n shape: (num_layers*2, batch_size, hidden_size)
        # c_n shape: (num_layers*2, batch_size, hidden_size)

        x = x.permute(2, 0, 1)

        out, (h_n, c_n) = self.lstm1(x)
        out, (h_n, c_n) = self.lstm2(out)
        out = out.permute(1, 0, 2)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out

class stackedbiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(stackedbiGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.gru2 = nn.GRU(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc1= nn.Linear(hidden_size*2, 64)
        self.fc2= nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=2)

    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # out shape: (batch_size, seq_length, hidden_size*2)
        # h_n shape: (num_layers*2, batch_size, hidden_size)

        x = x.permute(2, 0, 1)

        out, h_n = self.gru1(x)
        out, h_n = self.gru2(out)
        out = out.permute(1, 0, 2)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out
class transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, num_heads=2, hidden_dim=16):
        super(transformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.input_size, num_heads, hidden_dim, dropout), num_layers)
        self.fc1= nn.Linear(input_size, 64)
        self.fc2= nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):

        x = x.permute(2, 0, 1)

        out = self.transformer_encoder(x)
        out = out.permute(1, 0, 2)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out

# hyperparameters
input_size = 16
hidden_size = 128
num_layers = 2
num_classes = 2
batch_size = 32
num_epochs = 10
learning_rate = 0.001
dropout = 0.2

# load data
train_loader, test_loader = dataloader.EDFDataset('/data/lvsizhe/0718/', 13).split_dataset(batch_size = 32, train_ratio = 0.8, Shuffle = True, random_seed = 0)


# model = stackedbiLSTM(input_size, hidden_size, num_layers, num_classes, dropout).to(device)
# model = stackedbiGRU(input_size, hidden_size, num_layers, num_classes, dropout).to(device)
model = transformer(input_size, hidden_size, num_layers, num_classes, dropout).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
total_step = len(train_loader) 
for epoch in range(num_epochs):
    loss_list = []
    for i, dataset in tqdm.tqdm(enumerate(train_loader)):
        data = dataset[0].to(device)
        label = dataset[1].to(device)
        # print(data.shape, label.shape)

        outputs = model(data)
        predict = torch.zeros(batch_size,2).to(device)
        j= 0
        for output in outputs:
            # print(predict.shape, output.shape)
            predict[j][0] = output[:,0].mean()
            predict[j][1] = output[:,1].mean()
            j += 1
        # print(predict.shape, label.shape)
        loss = criterion(predict, label)
        loss_list.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        if (i+1) % 400 == 0:
            with torch.no_grad():
                correct = 0
                total = 0
                num1 = 0
                for d in test_loader:
                    data = d[0].to(device)
                    label = d[1].to(device)
                    
                    outputs = model(data)
                    # _, predicted = torch.max(outputs.data, 1)
                    # print(outputs)
                    predict = []
                    for output in outputs:
                        predicted = 0
                        print(output)
                        for x in output:
                            if x[0] > 0.01:
                                predicted += 0
                            else:
                                predicted += 1
                        predicted /= len(output)
                        predicted = 0 if predicted < 0.8 else 1
                        predict.append(predicted)
                    # print(predict)
                    total += label.size(0)
                    num1 += sum(label)
                    for id in range(len(predict)):
                        if predict[id] == label[id]:
                            correct += 1
                    # correct += (predict == label).sum().item()
                    if total >500:
                        break
                    
                print('Test Accuracy of the model on the test dataset: {} %'.format(100 * correct / total))
                print('1 of the model on the test dataset: {} %'.format(100 * num1 / total))
                if(100 * correct / total):
                    torch.save(model.state_dict(), 'model.ckpt')
    print('Epoch [{}/{}], average Loss: {:.4f}'.format(epoch+1, num_epochs, sum(loss_list)/len(loss_list)))
# test
with torch.no_grad():
    correct = 0
    total = 0
    num1 = 0
    for d in test_loader:
        data = d[0].to(device)
        label = d[1].to(device)
        
        outputs = model(data)
        # _, predicted = torch.max(outputs.data, 1)
        # print(outputs)
        predict = []
        for output in outputs:
            predicted = 0
            for x in output:
                if x[0] > 0.1:
                    predicted += 0
                else:
                    predicted += 1
            predicted /= len(output)
            predicted = 0 if predicted < 0.8 else 1
            predict.append(predicted)
        # print(predict)
        total += label.size(0)
        num1 += sum(label)
        for id in range(len(predict)):
            if predict[id] == label[id]:
                correct += 1
        # correct += (predict == label).sum().item()
    
    print('Test Accuracy of the model on the test dataset: {} %'.format(100 * correct / total))
    print('1 of the model on the test dataset: {} %'.format(100 * num1 / total))