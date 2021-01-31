#!/usr/bin/env python
# coding: utf-8

# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


# In[15]:


data_dir = 'Cat_Dog_data'


train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])

test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),transforms.ToTensor()])
                                       

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)


# In[16]:


model = models.densenet121(pretrained=True)
model


# In[8]:



for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier


# In[9]:


import time


# In[10]:


for device in ['cpu', 'cuda']:

    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device)

    for ii, (inputs, labels) in enumerate(trainloader):

       
        inputs, labels = inputs.to(device), labels.to(device)

        start = time.time()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ii==3:
            break
        
    print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")


# In[11]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
model.to(device)
epochs = 7
steps = 0

for e in range(epochs):
    running_loss = 0
    for  (inputs, labels) in (trainloader):
        images, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        log_ps = model.forward(images)
        loss = criterion(log_ps, labels)
        loss.requres_grad = True
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(running_loss/len(trainloader))'''

with torch.no_grad():
       model.eval()
       for (inputs, labels) in (trainloader):
           images, labels = inputs.to(device), labels.to(device)
           ps= torch.exp(model(images))
       top_p,top_class=ps.topk(1,dim=1)
       check = top_class == labels.view(*top_class.shape)
       accuracy = torch.mean(check.type(torch.FloatTensor))     
       print(f'Accuracy: {accuracy.item()*100}%')
       model.train()


# In[10]:


pip install helper


# In[13]:


import helper

model.eval()

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]

img = img.view(1, 50176)


with torch.no_grad():
    output = model.forward(img)

ps = torch.exp(output)

helper.view_classify(img.view(1, 224, 224), ps, version='Cat_Dog_data')


# In[ ]:




