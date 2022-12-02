import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd 

#Dataset Import
file_path = "D:\\Internship\\meeting-14.11-main\\meeting-14.11-main\\Heart_Disease.xlsx"
df = pd.read_excel(file_path) 

x = df.drop(columns=df.columns[-1], axis =1).to_numpy()
y = df.iloc[:,-1].to_numpy()

#Standard Scaling
sc = StandardScaler()
x = sc.fit_transform(x)

class dataset(Dataset):
  def __init__(self, x, y):
    self.x = torch.tensor(x, dtype = torch.float32)
    self.y = torch.tensor(y, dtype = torch.float32)
    self.length = self.x.shape[0]


  def __getitem__(self,idx):
    return self.x[idx], self.y[idx]

  def __len__(self):
    return self.length

trainset = dataset(x, y)
train_loader = DataLoader(trainset, batch_size = 64, shuffle= True)

#Defining the network

class Net(nn.Module):
  def __init__(self, input_shape):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_shape, 32)
    self.fc2 = nn.Linear(32, 64)
    self.fc3 = nn.Linear(64,1)

  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x))

#Hyper parameters
learning_rate = 0.02
epochs = 10000

#Model, optimizer, Loss
model = Net(input_shape=x.shape[1])
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
loss_fn = nn.BCELoss()

losses = []
accur = []

for i in range(epochs+1):
  for j, (x_train,y_train) in enumerate(train_loader):

    #Calculate output
    output = model(x_train)

    #Calculate loss
    loss = loss_fn(output, y_train.reshape(-1,1))

    #accuracy
    predicted = model(torch.tensor(x, dtype = torch.float32))
    acc = (predicted.reshape(-1).detach().numpy().round() == y).mean()

    #backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if acc == 0.95:
      break
  
  if i % 1000 == 0 :
      losses.append(loss)
      accur.append(acc)
      print(f"epoch {i} \t Loss : {loss} \t accuracy : {acc}")
      
#Final score
final_score = accur[-1]
print(f"The final score after optimizing is : {final_score}")

with open('Ann_Report.txt','w') as f:
    f.write("The final accuracy is : ")
    f.write(str(final_score)) 