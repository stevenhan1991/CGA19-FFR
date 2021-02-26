import torch
import numpy as np
from torch.utils.data import DataLoader


def dist3d(v1,v2):
	return np.sqrt((v1.data[0][0]-v2.data[0][0])**2+(v1.data[0][1]-v2.data[0][1])**2+(v1.data[0][2]-v2.data[0][2])**2)

def vec(streamline,vector_field,dim):
	vecs = []
	index = 0
	for i in range(0,len(streamline),3):
		pos = torch.stack([streamline[i],streamline[i+1],streamline[i+2]])
		_,vec = vecAtPos3d(pos,vector_field,dim)
		if vec is not None:
			vecs.append(vec)
	if len(vecs)!=0:
		v = torch.cat([vecs[i] for i in range(0,len(vecs))])
		return v,len(vecs)
	else:
		return None,0

def vecAtPos3d(pos,vector_field,dim):
	if(pos.data[0]<=0.0000001 or pos.data[0]>=dim[0]-1.0000001 or pos.data[1]<=0.0000001 or pos.data[1]>=dim[1]-1.0000001 or pos.data[2]<=0.0000001 or pos.data[2]>=dim[2]-1.0000001):
		return False,None

	x = int(pos.data[0])
	y = int(pos.data[1])
	z = int(pos.data[2])

	vecf1 = torch.stack([vector_field[0][x][y][z],vector_field[1][x][y][z],vector_field[2][x][y][z]])
	vecf2 = torch.stack([vector_field[0][x+1][y][z],vector_field[1][x+1][y][z],vector_field[2][x+1][y][z]])
	vecf3 = torch.stack([vector_field[0][x][y+1][z],vector_field[1][x][y+1][z],vector_field[2][x][y+1][z]])
	vecf4 = torch.stack([vector_field[0][x+1][y+1][z],vector_field[1][x+1][y+1][z],vector_field[2][x+1][y+1][z]])
	vecf5 = torch.stack([vector_field[0][x][y][z+1],vector_field[1][x][y][z+1],vector_field[2][x][y][z+1]])
	vecf6 = torch.stack([vector_field[0][x+1][y][z+1],vector_field[1][x+1][y][z+1],vector_field[2][x+1][y][z+1]])
	vecf7 = torch.stack([vector_field[0][x][y+1][z+1],vector_field[1][x][y+1][z+1],vector_field[2][x][y+1][z+1]])
	vecf8 = torch.stack([vector_field[0][x+1][y+1][z+1],vector_field[1][x+1][y+1][z+1],vector_field[2][x+1][y+1][z+1]])

	facx = pos[0]-x
	facy = pos[1]-y
	facz = pos[2]-z

	ret = (1-facx)*(1-facy)*(1-facz)*vecf1+(facx)*(1-facy)*(1-facz)*vecf2+(1-facx)*(facy)*(1-facz)*vecf3+(facx)*(facy)*(1-facz)*vecf4+(1-facx)*(1-facy)*(facz)*vecf5+(facx)*(1-facy)*(facz)*vecf6+(1-facx)*(facy)*(facz)*vecf7+(facx)*(facy)*(facz)*vecf8
	return True,ret

def GetData(pos_folder,vec_folder,num_of_streamlines,dim_high,dim_low):
	pos = []
	vec = []
	for i in range(1,num_of_streamlines+1):
		pos_array = np.fromfile(pos_folder+'pos'+'{:3d}'+'.dat',dtype='<f')
		vec_array = np.fromfile(vec_folder+'vec'+'{:3d}'+'.dat',dtype='<f')

		assert len(pos_array) == len(vec_array)

		for i in range(0,len(pos_array),3):
			pos_array[i] = pos_array[i]*dim_low[0]/dim_high[0]
			pos_array[i+1] = pos_array[i+1]*dim_low[0]/dim_high[0]
			pos_array[i+2] = pos_array[i+2]*dim_low[0]/dim_high[0]

		pos.append(pos_array)
		vec.append(vec_array)

	return pos, vec

def saveVectorfield(vec,args):
	vec = vec.detach().cpu().numpy()
	vec = vec.flatten('F')
	vec = np.asarray(vec,dtype='<f')
	vec .tofile(args.result_path+'vec.dat',format='<f')



