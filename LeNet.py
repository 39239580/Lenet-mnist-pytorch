# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:59:51 2018

@author: Weixia
"""

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch.nn as nn
import time 
import os
import matplotlib.pyplot as plt
import csv
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层卷积层     
        layer1=nn.Sequential() #此函数是将所有的层，组合到一起
        layer1.add_module('conv1',nn.Conv2d(1,6,5,1,padding=1)) #卷积层,增加模块
        #上一层的特征图像的深度为3，卷积核为3*3，卷积核的个数为32， stride=1，四周进行一个像素点的零填充
        #3  32*32*3   1   1
        #layer1.add_module('relu1',nn.ReLU(True))
        layer1.add_module('pool1',nn.MaxPool2d(2,2))
        # stride=2,k=2即进行2*2降采样
        self.layer1=layer1
        
        layer2=nn.Sequential()
        layer2.add_module('conv2',nn.Conv2d(6,16,5,1,padding=1))
#        layer2.add_module('relu2',nn.ReLU(True))
        layer2.add_module('pool2',nn.MaxPool2d(2,2))
        self.layer2=layer2
        
#        layer3=nn.Sequential()
#        layer3.add_module('conv3',nn.Conv2d(64,128,3,1,padding=1))
#        layer3.add_module('relu3',nn.ReLU(True))
#        layer3.add_module('pool3',nn.MaxPool2d(2,2))
#        self.layer3=layer3
        
        layer3=nn.Sequential()
        layer3.add_module('fc1',nn.Linear(400,120))
#        layer3.add_module('relu4',nn.ReLU(True))
        layer3.add_module('fc2',nn.Linear(120,84))
#        layer4.add_module('relu5',nn.ReLU(True))
        layer3.add_module('fc3',nn.Linear(84,10))
        self.layer3=layer3
    
    # forward
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
#        x=self.layer3(x)
        x=x.view(x.size(0),-1)
        x=self.layer3(x)
        return x

def main():
    if os.path.exists('./LeNet_model/model'):
        print('已存在模型文件')
    else:
        print('不存在模型文件')
    print('请输入你的选择：1.训练并测试，2.为直接测试？')
    selection=input()       
    # 超参数
    download_start_time=time.time()
    batch_size=64
    learning_rate=1e-2
    num_epoches=3
    momentum=0.9
    # transforms.Compose()函数作用是将各种预处理操作组合到一起，
    # transform.ToTensor()函数作用就是将图片数据转成张量，并将个点的值限制在0~1内
    # transform.Normalize()函数作用，是将上述0~1之间的值进行标准化处理，(x-0.5)/0.5
    data_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
    #加载数据
    train_dataset=datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)
    test_dataset=datasets.MNIST(root='./data',train=False,transform=data_tf,)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)   #加载数据
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    download_end_time=time.time()
    print("共计下载数据及加载数据花费时间：%f"%(download_end_time-download_start_time))
    #  对应的实参传入__init__()的四个输入参数中，28*28为图片的大小
    model=LeNet()  #输出num_classes=10
    if torch.cuda.is_available():
        model=model.cuda()
    else:
        model=model
    print(model)
    criterion=nn.CrossEntropyLoss() #交叉熵作为损失函数
    optimizer=optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    if selection=='1':
        print('进行模型创建并训练')
        train_start_time=time.time()
        train_epoch=[]
        train_loss_value=[]
        train_acc_value=[]
        for epoch in range(num_epoches):
            start=time.time()
            print('Current epoch ={}'.format(epoch)) 
            train_loss=0
            train_acc=0
            for i ,(images,labels)in enumerate(train_loader):#利用enumerate取出一个可迭代对象的内容
                #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
                #同时列出数据和数据下标，一般用在 for 循环当中。
                #seq =['one','two','three']
                #for i,element in enumerate(seq):
                #print (i,element)
                #0 one
                #1 two
                #2 three
        
                #list(enumerate(seq))
                #[(0, 'one'), (1, 'two'), (2, 'three')]
        
                #list(enumerate(seq, start=1))
                #[(1, 'one'), (2, 'two'), (3, 'three')]
        
                if torch.cuda.is_available():
                    # python中-1为根据系统自动判别个数，
                    inputs=Variable(images).cuda()
                    target=Variable(labels).cuda()
                else:
                    inputs=Variable(images)
                    target=Variable(labels)
                # forward
                out=model(inputs)
                loss=criterion(out,target)
                #backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # loss.item（）是每一轮的损失函数
                train_loss+=loss.item()
                _,pred=torch.max(out,1)
                correct_num=(pred==target).sum()
                train_acc+=correct_num.item()
            train_loss=train_loss/len(train_dataset)
            train_acc=train_acc/len(train_dataset)
            train_epoch.append(epoch+1)
            train_loss_value.append(train_loss)
            train_acc_value.append(train_acc)
            #    if (epoch+1)%10==0:
            print('Epoch[{}/{}],loss:{:.6},Acc:{:.6}%,train_time:{:.6}s'.format(epoch+1,num_epoches,train_loss,train_acc*100,time.time()-start))
        train_spend_time=time.time()        
        # 保存模型
        torch.save(model,'./LeNet_model/model')                 
        print('训练花费时间：%fs'%(train_spend_time-train_start_time))
        with open('./LeNet_model/LeNet_model.csv','w') as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow(["train_epoch","train_acc","train_loss"])          
            for i in range(len(train_epoch)):
                temp=[]
                temp.append(train_epoch[i])
                temp.append(train_acc_value[i])
                temp.append(train_loss_value[i])
                writer.writerow(temp)   
    # 加载模型
    model=torch.load('./LeNet_model/model')        
    # 测试
    model.eval()
    #model.gpu()   #模型使用cpu加载
    eval_loss=0
    eval_acc=0
    #print('debug用-----')
    test_start_time=time.time()
    for data in test_loader:
        img,label=data
        #    img=img.view(img.size(0),-1)
        if torch.cuda.is_available():
            inputs=Variable(img).cuda()
            target=Variable(label).cuda()
        else:
            inputs=Variable(img)
            target=Variable(label)
        out=model(inputs)
        #    print('你是羊')
        loss=criterion(out,target)
        eval_loss+=loss.item()
        _,pred=torch.max(out,1)
        num_correct=(pred==target).sum()
        #    print('你懒洋洋')
        eval_acc+=num_correct.item()
    print('Test Loss:{:.6f},Acc:{:.6}%'.format(eval_loss/(len(test_dataset)),eval_acc/(len(test_dataset))*100))
    test_spend_time=time.time()
    print('测试花费时间：%fs'%(test_spend_time-test_start_time))

    
    
    
    plt.plot(train_epoch,train_acc_value,'r-',label='train_acc')
    plt.plot(train_epoch,train_loss_value,'r-',label='train_acc')
    plt.show()
    

if __name__=='__main__':
    main()        