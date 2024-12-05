from utils import *
import sklearn.metrics as skm
from sklearn import preprocessing
from torch.utils.data import Dataset,DataLoader
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import numpy as np
# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)                     # 设置 Python 随机种子
    np.random.seed(seed)                  # 设置 numpy 随机种子
    torch.manual_seed(seed)               # 设置 PyTorch CPU 随机种子
    torch.cuda.manual_seed(seed)          # 设置 PyTorch GPU 随机种子
    torch.backends.cudnn.deterministic = True  # 保证每次卷积算法的选择相同
    torch.backends.cudnn.benchmark = False     # 禁止使用非确定性算法
# 设置随机种子
set_seed(42)
Subjects = ['A','B']#定义了一个包含两个受试者标识符的列表 Subjects，其中包括 'A' 和 'B'。
Subject = Subjects[1]#将列表中第二个受试者（索引 1，即 'B'）赋值给变量 Subject，意味着接下来的操作将使用受试者 'B' 的数据。
samps = 64 #定义每个 ERP 提取的样本数量。
Target,Target_label,NonTarget,NonTarget_label = train_data_and_label(Subject,samps)

#对数据和其对应的标签进行随机打乱
Target, Target_label = shuffled_data(Target, Target_label)
NonTarget, NonTarget_label = shuffled_data(NonTarget, NonTarget_label)
#重命名
train_P300_dataset = Target
train_P300_label = Target_label
train_non_P300_dataset = NonTarget
train_non_P300_label = NonTarget_label
# 7794：总时间点数=15(每个字符重复15轮刺激)×12(每轮刺激行列闪烁12次)×(闪烁持续100ms+间隔75ms)×240HZ采样率
#170个样本点，每个样本的形状为 (64, 64)，即每个有 64 个时间点（由 samps 定义）和 64 个通道。
# 每个字符行和列加起来闪烁两次，85个字符就闪85*2=170次，也就是对应P300出现170次，非p300 85*10=850次
# (170, 64, 64) P300样本个数 * 采样率 * 通道数
print ('train_P300_dataset:' + str(train_P300_dataset.shape))
# (170, 1) P300样本个数
print ('train_P300_label:' + str(train_P300_label.shape))
# (850, 64, 64) 非P300的样本个数 * 采样率 * 通道数
print ('train_non_P300_dataset:'+ str(train_non_P300_dataset.shape))
# (850, 1) 非P300的样本个数
print ('train_non_P300_label:' + str(train_non_P300_label.shape))

Target,Target_label,NonTarget,NonTarget_label = test_data_and_label(Subject,samps)
Target, Target_label = shuffled_data(Target, Target_label)
NonTarget, NonTarget_label = shuffled_data(NonTarget, NonTarget_label)
test_P300_dataset = Target
test_P300_label = Target_label
test_non_P300_dataset = NonTarget
test_non_P300_label = NonTarget_label
print ('test_P300_dataset:' + str(test_P300_dataset.shape))
print ('test_P300_label:' + str(test_P300_label.shape))
print ('test_non_P300_dataset:'+ str(test_non_P300_dataset.shape))
print ('test_non_P300_label:' + str(test_non_P300_label.shape))


#进行数据标准化。每个样本的特征向量按照 L2 范数进行归一化处理，使得每个样本的特征向量的平方和为 1。
def data_prepocessing(data):
    pd_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        pd_data[i,:,:] = preprocessing
        normalize(data[i,:,:], norm='l2')
    return pd_data
#调整数据和标签的形状以适应某些深度学习模型的输入需求。
def add_dimension(x, y):
    #将数据类型转换为 float32，一般在深度学习中标准做法，因为这能够在很多模型框架中提高运算效率。
    #x最开始的维度是bs*样本点*通道,变成了bs*通道*样本点
    x = np.reshape(x,[x.shape[0],x.shape[2],x.shape[1]]).astype('float32')
    y = np.reshape(y,[y.shape[0],1]).astype('float32')
    return x, y

#将 P300和 non-P300样本合并为单个训练数据集和对应的标签集。
train_data = np.vstack((train_P300_dataset,train_non_P300_dataset))#np.vstack：用于垂直堆叠（在样本数维度上）两个数组
train_label = np.vstack((train_P300_label,train_non_P300_label))
train_data, train_label = shuffled_data(train_data, train_label)#打乱
print(train_data.shape,train_label.shape)
test_data = np.vstack((test_P300_dataset,test_non_P300_dataset))
test_label = np.vstack((test_P300_label,test_non_P300_label))
test_data, test_label = shuffled_data(test_data, test_label)

#调整训练和测试数据的维度。
train_data, train_label = add_dimension(train_data,train_label)
test_data, test_label = add_dimension(test_data, test_label)
print(train_data,train_label)
print('train_data:',train_data.shape)
print('train_label:',train_label.shape)
print('test_data:',test_data.shape)
print('test_label:',test_label.shape)

class Dataset(Dataset):
    def __init__(self, Data, Label):
        self.Data = torch.from_numpy(Data)
        self.Label = torch.from_numpy(Label)
    def __getitem__(self, index):#定义如何通过索引提取数据集中的元素。
        return self.Data[index], self.Label[index]
    def __len__(self):#返回数据集的大小。
        return len(self.Data)#返回数据集中样本的个数。这允许 PyTorch DataLoader 知道在请求完整的数据集之前迭代多少次。

input_channel = 64
input_length = 64
output_channel = 1
channel_sizes = [64,64,64,64]
kernel_size = 7
dropout = 0.5
input = torch.randn(32, 64, 64)
lstm_hidden_size = 32
lstm_num_layers = 2
net = TnS_net(input_channel, input_length,output_channel, channel_sizes,kernel_size,lstm_hidden_size,lstm_num_layers,dropout)

net(input).shape

net(input).shape

LR = 0.001
epochs = 50
b_size = 32

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
Dataset = Dataset(train_data,train_label)
Train_loader = DataLoader(Dataset,batch_size = b_size ,shuffle = True)

def train(epoch,train_data,train_label):
    train_loss = 0.0
    train_loss_array = []
    for step, (inputs, labels) in enumerate(Train_loader):
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_loss_array.append(loss.item())
    return train_loss_array
def test(net,epoch,vaild_data,vaild_label):
    predicted = []
    inputs = torch.from_numpy(vaild_data)
    labels = torch.from_numpy(vaild_label)
    inputs, labels = Variable(inputs), Variable(labels)
    predicted = net(inputs)
    predicted = predicted.data.cpu().numpy()
    labels = labels.data.numpy()
    accuracy_score = skm.accuracy_score(labels, np.round(predicted))
    return accuracy_score

for epoch in range(1, epochs+1):
    print ("\nEpoch ", epoch)
    train_loss=train(epoch,train_data,train_label)
    accuracy_score = test(net,epoch,test_data, test_label)
    print (' accuracy_score', accuracy_score)
