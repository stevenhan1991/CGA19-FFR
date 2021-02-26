import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import argparse

from torch.autograd import Variable
import numpy as np
import time
from utils import vec, vecAtPos3d, GetData, saveVectorfield


parser = argparse.ArgumentParser(description='PyTorch Implementation of CG&A work')
parser.add_argument('--lr', type=float, default= 1e-3, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--num', type=int, default=300, metavar='N',
                    help='number of training sreamlines')
parser.add_argument('--pos_path', type=str, default='../pos/', metavar='N',
                    help='the path where the pos files store')
parser.add_argument('--vec_path', type=str, default='../vec/', metavar='N',
                    help='the path where the vec files store')


args = parser.parse_args()
print(not args.no_cuda)
print(torch.cuda.is_available())
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 10, 'pin_memory': True} if args.cuda else {}

def train(model,epoch,streamline_array,vecs_array):
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	for itera in range(1,epochs+1):
		loss = 0
		x = time.time()
		print("========================")
		print(itera)
		for batch_idx in range(0,len(streamline_array)):
			groudtruth = torch.FloatTensor(streamline_array[batch_idx])
			vecs_ground = torch.FloatTensor(vecs_array[batch_idx])
			if args.cuda:
				groudtruth = groudtruth.cuda()
				vecs_ground = vecs_ground.cuda()
			groudtruth = Variable(groudtruth)
			vecs,index = model(groudtruth)
			if vecs is not None:
				optimizer.zero_grad()
				loss_fun = nn.MSELoss(size_average=False)
				loss_g = loss_fun(vecs,vecs_ground[:3*index])
				loss = loss+loss_g.item()
				loss_g.backward()
				optimizer.step()
		y = time.time()
		print("loss = "+str(loss))
		print("Time = "+str(y-x))
		if itera%10==0:
			saveVectorfield(model.vector_field,args)

class Net(nn.Module):
	def __init__(self,dim):
		super(Net,self).__init__()
		self.vector_field = nn.Parameter(torch.FloatTensor(3,dim[0],dim[1],dim[2]).uniform_(-1e-1,1e-1),requires_grad=True)

	def forward(self,seeds):
		return vec(seeds,self.vector_field)

def main():
	dim_low = [16,16,16]
	dim_high = [51,51,51]
	streamline_array,vecs_array = GetData(args.pos_path,args.vec_path,args.num,dim_high,dim_low)
	model = Net(dim_low)
	if args.cuda:
		model.cuda()
	train(model,args.epochs,streamline_array,vecs_array)

if __name__== "__main__":
    main()