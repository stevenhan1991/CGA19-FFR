import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from utils import vec, vecAtPos3d, GetData, saveVectorfield

parser = argparse.ArgumentParser(description='PyTorch Implementation of CG&A work')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.01)')   
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--result_path', type=str, default='../result/', metavar='N',
                    help='the path where we stroe the result')

args = parser.parse_args()
print(not args.no_cuda)
print(torch.cuda.is_available())
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 10, 'pin_memory': True} if args.cuda else {}


def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv")!=-1:
		m.weight.data.normal_(0,0.01)
	elif classname.find("Linear")!=-1:
		m.weight.data.normal_(0,0.01)
	elif classname.find("BatchNorm")!=-1:
		m.weight.data.normal_(1.0,0.01)
		m.bias.data.constant_(0.0)


def BuildResidualBlock(channels,dropout,padding,depth,bias):
	layers = []
	for i in range(int(depth)):
		layers += [nn.Conv3d(channels,channels,kernel_size=3,padding=padding,bias=bias),
		           nn.BatchNorm3d(channels),
		           nn.ReLU(True)]
		if dropout:
			layers += [nn.Dropout(0.5)]
	layers += [nn.Conv3d(channels,channels,kernel_size=3,padding=padding,bias=bias),
		       nn.BatchNorm3d(channels)]
	return nn.Sequential(*layers)

class ResidualBlockDown(nn.Module):
	def __init__(self,channels,dropout,padding,depth,bias):
		super(ResidualBlockDown,self).__init__()
		self.block = BuildResidualBlock(channels,dropout,padding,depth,bias)

	def forward(self,x):
		out = x+self.block(x)
		return x

class line2vector(nn.Module):
	def __init__(self,dim):
		super(line2vector,self).__init__()
		self.deconv1 = nn.ConvTranspose3d(3,64,3) # 18
		self.deconv2 = nn.ConvTranspose3d(64,128,3) # 20
		self.residualblockup1 = ResidualBlock(128,False,padding=1,depth=1,bias=True)
		self.deconv3 = nn.ConvTranspose3d(128,256,3) # 22
		self.residualblockup2 = ResidualBlock(256,False,padding=1,depth=1,bias=True)
		self.deconv4 = nn.ConvTranspose3d(256,512,3,stride=2) # 45
		self.residualblockup3 = ResidualBlock(512,False,padding=1,depth=1,bias=True)
		self.deconv5 = nn.ConvTranspose3d(512,512,3) # 47
		self.residualblockup4 = ResidualBlock(512,False,padding=1,depth=1,bias=True)
		self.deconv6 = nn.ConvTranspose3d(512,512,3) # 49
		self.deconv7 = nn.ConvTranspose3d(512,3,3) #51

		self.dim = dim



	def forward(self,x,pos=None):
		x = F.relu(self.deconv1(x))
		x = F.relu(self.deconv2(x))
		x = self.residualblockup1(x)
		x = F.relu(self.deconv3(x))
		x = self.residualblockup2(x)
		x = F.relu(self.deconv4(x))
		x = self.residualblockup3(x)
		x = F.relu(self.deconv5(x))
		x = self.residualblockup4(x)
		x = F.relu(self.deconv6(x))
		x = self.deconv7(x)
		if pos:
			return x
		else:
			vecs = vec(pos,x,self.dim)
			return x,vecs


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epochs,low_res,streamline_array,vecs_array):
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	for itera in range(1,epochs+1):
		loss = 0
		x = time.time()
		print("========================")
		print(itera)
		for batch_idx in range(0,len(streamline_array)):
			x1 = time.time()
			groudtruth = torch.FloatTensor(streamline_array[batch_idx])
			vec_ground = torch.FloatTensor(vecs_array[batch_idx])
			if args.cuda:
				groudtruth = groudtruth.cuda()
				low_res = low_res.cuda()
				vec_ground = vec_ground.cuda()
			groudtruth = Variable(groudtruth)
			data = Variable(low_res)
			vec_ground = Variable(vec_ground)
			vector_field,vecs = model(data,groudtruth)
			if vecs is not None:
				optimizer.zero_grad()
				loss_fun = nn.MSELoss(size_average=False)
				loss_g = loss_fun(vecs,vec_groudtru[:3*idx])
				loss = loss+loss_g.data[0]
				loss_g.backward()
				optimizer.step()
			x2 = time.time()
		y = time.time()
		print("loss = "+str(loss))
		print("Time = "+str(y-x))
		if itera%40==0:
			adjust_learning_rate(optimizer,itera)
		if itera%50==0:
			vector_field = model(low_res)
			saveVectorfield(vector_field,args)



def main():
	dim_low = [51,51,51]
	dim_high = [51,51,51]
	low_res = np.fromfile(args.result_path+'vec.dat',dtype='<f')
	low_res = low_res.reshape(dim_low[2],dim_low[1],dim_low[0],3,1).transpose()
	low_res = torch.FloatTensor(low_res)
	streamline_array,vecs_array = GetData(args.pos_path,args.vec_path,args.num,dim_high,dim_low)
	model = line2vector(dim_high)
	if args.cuda:
		model.cuda()
	train(model,low_res,args.epochs,streamline_array,vecs_array)

if __name__== "__main__":
    main()
