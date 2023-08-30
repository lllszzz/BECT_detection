import mne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import scipy
from scipy import signal
import matplotlib.pyplot as plt


class dataset(Dataset):
    def __init__(self):
        self.data = np.array([])
        self.label = np.array([])
    # def add_data(self, data):
    #     if self.data.size == 0:
    #         self.data = data.reshape(1, data.shape[0],data.shape[1],data.shape[2])
    #     else:
    #         self.data = np.concatenate((self.data, data.reshape(1, data.shape[0],data.shape[1],data.shape[2])), axis=0)
    def add_data(self, data):
        if self.data.size == 0:
            self.data = data
        else:
            self.data = np.concatenate((self.data, data), axis=0)
    def add_label(self, label):
        if self.label.size == 0:
            self.label = label #.reshape(1, label.shape[1])
        else:
            self.label = np.concatenate((self.label, label), axis=0)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        return self.data[index], self.label[index]


class EDFDataset(Dataset):
    def __init__(self, folder_path, T):
        self.folder_path = folder_path
        self.T = T
        self.raw_data_list=[]
        self.dataset = dataset()
        self.data_path = '/home/lvsizhe/bect_detection/data'
        
        # 加载edf文件
        if os.path.exists(folder_path):
            self.edf_path = os.path.join(folder_path, 'edf')
            self.label_path = os.path.join(folder_path, 'label')
        

        dataname = folder_path.split('/')[-2]#'0718'
        edf_path = os.path.join(self.data_path, dataname+'_edf.npy')
        label_path = os.path.join(self.data_path, dataname+'_label.npy')

        #如果已经存在数据文件，则直接加载
        if os.path.exists(edf_path):
            self.dataset.data = np.load(edf_path, allow_pickle=True)
            self.dataset.data = self.dataset.data.astype(float)
            # data= self.dataset.data.copy()
            # self.dataset.data[:,0]=data[:,0]-data[:,19]
            # self.dataset.data[:,1]=data[:,1]-data[:,20]
            # self.dataset.data[:,2]=data[:,2]-data[:,19]
            # self.dataset.data[:,3]=data[:,3]-data[:,20]
            # self.dataset.data[:,4]=data[:,4]-data[:,19]
            # self.dataset.data[:,5]=data[:,5]-data[:,20]
            # self.dataset.data[:,6]=data[:,6]-data[:,19]
            # self.dataset.data[:,7]=data[:,7]-data[:,20]
            # self.dataset.data[:,8]=data[:,8]-data[:,19]
            # self.dataset.data[:,9]=data[:,9]-data[:,20]
            # self.dataset.data[:,10]=data[:,10]-data[:,19]
            # self.dataset.data[:,11]=data[:,11]-data[:,20]
            # self.dataset.data[:,12]=data[:,12]-data[:,19]
            # self.dataset.data[:,13]=data[:,13]-data[:,20]
            # self.dataset.data[:,14]=data[:,14]-data[:,19]
            # self.dataset.data[:,15]=data[:,15]-data[:,20]

        if os.path.exists(label_path):                    
            self.dataset.label = np.load(label_path, allow_pickle=True)
            self.dataset.label = self.dataset.label.astype(np.int64)
            # for i in range(len(self.dataset.label)):
                
            # print(self.dataset.label.shape, self.dataset.data.shape)
            # delete_index = []
            # for i in range(len(self.dataset.label)):
            #     if self.dataset.label[i] == 3 or 4:
            #         delete_index.append(i)
            # self.dataset.data = np.delete(self.dataset.data, delete_index, axis=0)
            # self.dataset.label = np.delete(self.dataset.label, delete_index, axis=0)
            # print(self.dataset.label.shape, self.dataset.data.shape)
            # np.save(edf_path, self.dataset.data)
            # np.save(label_path, self.dataset.label)
                    
            # np.save(label_path, self.dataset.label)
        print(self.dataset.data.shape, self.dataset.label.shape)
        #如果不存在数据文件，则加载原始数据，并保存

        if not os.path.exists(edf_path) or not os.path.exists(label_path):
            cnt = 0
            for root, dirs, files in os.walk(self.edf_path):
                for file in files:
                    single_edf_path = os.path.join(root,file)
                    name = root.split('/')[-1]
                    single_label_path = os.path.join(self.label_path, name+'.xlsx')
                    
                    #load data
                    raw = mne.io.read_raw_edf(single_edf_path, preload=True)
                    print(raw.info)
                    # 选择通道
                    selected_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1', 'A2', 'T1', 'T2']
                    raw.pick_channels(selected_channels, ordered=True)
                    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
                    self.raw_data = raw.get_data(picks=picks)
                    #降采样
                    if raw.info['sfreq'] != 500:
                        self.raw_data = signal.decimate(self.raw_data, int(raw.info['sfreq'] / 500), axis=1, zero_phase=True)
                    
                    
                    # print(self.raw_data.shape)
                    
                    # selected_channels_index = []
                    # for i in range(len(raw.info['ch_names'])):
                    #     if raw.info['ch_names'][i] in selected_channels:
                    #         selected_channels_index.append(i)
                    # self.raw_data = self.raw_data[selected_channels_index, :]

                    # standardization
                    print(np.mean(self.raw_data, axis=1, keepdims=True).shape, np.std(self.raw_data, axis=1, keepdims=True).shape)
                    self.raw_data = (self.raw_data - np.mean(self.raw_data, axis=1, keepdims=True)) / np.std(self.raw_data, axis=1, keepdims=True)


                    print(self.raw_data.shape)
                    # data = np.zeros((21,self.raw_data.shape[1]))
                    # av = np.mean(self.raw_data, axis=0)
                    # print(av.shape)
                    # data[0, :]=self.raw_data[0,:]-self.raw_data[19,:]
                    # data[1, :]=self.raw_data[1,:]-self.raw_data[20,:]
                    # data[2, :]=self.raw_data[2,:]-self.raw_data[19,:]
                    # data[3, :]=self.raw_data[3,:]-self.raw_data[20,:]
                    # data[4, :]=self.raw_data[4,:]-self.raw_data[19,:]
                    # data[5, :]=self.raw_data[5,:]-self.raw_data[20,:]
                    # data[6, :]=self.raw_data[6,:]-self.raw_data[19,:]
                    # data[7, :]=self.raw_data[7,:]-self.raw_data[20,:]
                    # data[8, :]=self.raw_data[8,:]-self.raw_data[19,:]
                    # data[9, :]=self.raw_data[9,:]-self.raw_data[20,:]
                    # data[10, :]=self.raw_data[10,:]-self.raw_data[19,:]
                    # data[11, :]=self.raw_data[11,:]-self.raw_data[20,:]
                    # data[12, :]=self.raw_data[12,:]-self.raw_data[19,:]
                    # data[13, :]=self.raw_data[13,:]-self.raw_data[20,:]
                    # data[14, :]=self.raw_data[14,:]-self.raw_data[19,:]
                    # data[15, :]=self.raw_data[15,:]-self.raw_data[20,:]
                    # data[16, :]=self.raw_data[21,:]-(self.raw_data[19,:]+self.raw_data[20,:])/2
                    # data[17, :]=self.raw_data[22,:]-(self.raw_data[19,:]+self.raw_data[20,:])/2
                    # data[18, :]=self.raw_data[16,:]-av
                    # data[19, :]=self.raw_data[17,:]-av
                    # data[20, :]=self.raw_data[18,:]-av
                    
                    data = np.zeros((8,self.raw_data.shape[1]))
                    # data[0, :]=self.raw_data[0,:]-self.raw_data[19,:]
                    # data[1, :]=self.raw_data[1,:]-self.raw_data[20,:]
                    # data[2, :]=self.raw_data[2,:]-self.raw_data[19,:]
                    # data[3, :]=self.raw_data[3,:]-self.raw_data[20,:]
                    data[0, :]=self.raw_data[4,:]-self.raw_data[19,:]
                    data[1, :]=self.raw_data[5,:]-self.raw_data[20,:]
                    data[2, :]=self.raw_data[6,:]-self.raw_data[19,:]
                    data[3, :]=self.raw_data[7,:]-self.raw_data[20,:]
                    # data[8, :]=self.raw_data[8,:]-self.raw_data[19,:]
                    # data[9, :]=self.raw_data[9,:]-self.raw_data[20,:]
                    # data[10, :]=self.raw_data[10,:]-self.raw_data[19,:]
                    # data[11, :]=self.raw_data[11,:]-self.raw_data[20,:]
                    data[4, :]=self.raw_data[12,:]-self.raw_data[19,:]
                    data[5, :]=self.raw_data[13,:]-self.raw_data[20,:]
                    data[6, :]=self.raw_data[14,:]-self.raw_data[19,:]
                    data[7, :]=self.raw_data[15,:]-self.raw_data[20,:]
                    # data[16, :]=self.raw_data[21,:]-(self.raw_data[19,:]+self.raw_data[20,:])/2
                    # data[17, :]=self.raw_data[22,:]-(self.raw_data[19,:]+self.raw_data[20,:])/2
                    # data[18, :]=self.raw_data[16,:]-av
                    # data[19, :]=self.raw_data[17,:]-av
                    # data[20, :]=self.raw_data[18,:]-av
                    
                    self.raw_data = data

                
                    
                    # 标准化数据
                    # self.scaler = StandardScaler()
                    # self.raw_data = self.scaler.fit_transform(self.raw_data)
                    
                    # 剪裁数据       
                    self.data = np.array([]) 

                    t = 1
                    while t*self.T*500 < self.raw_data.shape[1]:
                        data = self.raw_data[:, int((t-1)*self.T*500):int(t*self.T*500)]
                        data = self.iir_filter(data, 1, 70, fs=500, order=5)
                        data_2 = self.smoothed_nonlinear_energy_operator(data)
                        # print(data.shape, data_2.shape)
                        data = np.concatenate((data, data_2), axis=0)
                        if t == 1:
                            self.data = data.reshape(1, data.shape[0],data.shape[1])
                        else:    
                            self.data = np.concatenate((self.data, data.reshape(1, data.shape[0],data.shape[1])), axis=0)
                        t += 1  
                    # print(self.data.shape)     # (timestep, channel(21), sample_point(6500))
            
                    # self.dataset.add_data(self.data)

                    if not os.path.exists(label_path):
                        #load label
                        if not os.path.exists(single_label_path):
                            label = np.zeros(self.data.shape[0])

                        else:
                            label= pd.read_excel(single_label_path, header=None)
                            label = np.array(label)
                            label = label[:self.data.shape[0], 1]
                            delete_index = []
                            for i in range(len(label)):
                                if label[i] == 2 or label[i] == 3:
                                    delete_index.append(i)
                            new_data = np.delete(self.data, delete_index, axis=0)
                            new_label = np.delete(label, delete_index, axis=0)
                            self.data = new_data
                            label = new_label
                            # print(label.shape)  # (1, timestep)
                            print('delete len:', len(delete_index))
                            print(self.data.shape, label.shape)
                        self.dataset.add_label(label)
                        self.dataset.add_data(self.data)
                    cnt+=1
                    print(cnt)
                # if cnt>5:
                #     break

            # 保存数据
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            if not os.path.exists(edf_path):
                np.save(edf_path, self.dataset.data)
            if not os.path.exists(label_path):
                np.save(label_path, self.dataset.label)

        # self.output_graph(self.dataset.data)
        self.dataset.data = self.sG_filter(self.dataset.data)
        self.dataset.data = torch.tensor(self.dataset.data, dtype=torch.float32)
        
        # print(self.dataset.data.shape)
        # print(self.dataset.label.shape)
        # self.train_data, self.test_data = train_test_split(self.dataset,test_size=0.2)#(batch_size = 32, train_ratio = 0.8, Shuffle = True, random_seed = 0)
        # self.train_data, self.test_data = self.split_dataset(self.dataset, batch_size = 32, test_ratio = 0.8, Shuffle = True, random_seed = 0)
        # for data,label in self.train_data:
        #     print(data.shape)
        #     break
        # print(self.test_data.data.shape)
            # 转置数据维度
            # self.raw_data = np.transpose(self.raw_data, (1, 0))
            
            # 转换为tensor
            # self.raw_data = torch.tensor(self.raw_data, dtype=torch.float32)
    def split_dataset(self, batch_size = 32, train_ratio = 0.8, Shuffle = True, random_seed = 0):
        # 划分数据集
        self.train_data = dataset()
        self.test_data = dataset()
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=1-train_ratio, shuffle=Shuffle, random_state=random_seed)
        print(len(train_dataset), len(test_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle= Shuffle,drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

        return train_dataloader, test_dataloader

    def iir_filter(self, data, lowcut, highcut, fs, order=5):
        data = signal.detrend(data)
        low = 1
        high = 70
        sos = signal.iirfilter(order, [low, high], btype='band', analog=False, ftype='butter', output='sos', fs=fs)
        filtered_data = signal.sosfilt(sos, data)
        return filtered_data
    
    def sG_filter(self, data, window_size=35, order=5):
        data_smoothed = scipy.signal.savgol_filter(data, window_size, order, axis=-1, mode='nearest')
        return data_smoothed
    
    def smoothed_nonlinear_energy_operator(self, data, window_size=35, order=5):
        data_copy = data.copy()
        data_copy  = np.power(data_copy, 2)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if j==0 :
                    data_copy[i,j] = data_copy[i,j] - data[i,j+1]*data[i,j]
                elif j==data.shape[1]-1:    
                    data_copy[i,j] = data_copy[i,j] - data[i,j-1]*data[i,j]
                else:
                    data_copy[i,j] = data_copy[i,j] - data[i,j-1]*data[i,j+1]
        data = data_copy
        # print(data)
        triangular_window = np.zeros(window_size)
        center = window_size // 2
        smoothed_data = np.zeros(data.shape)
        for i in range(window_size):
            triangular_window[i] = 1 - np.abs(i - center) / center
        for i in range(data.shape[0]):
            smoothed_data[i] = scipy.signal.convolve(data[i], triangular_window, mode='same')
        # print(smoothed_data)
        return smoothed_data

    # def morphological_filter(self,data):


    def output_graph(self,data):
        #save plot

        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        axes[0, 0].plot(data[5,1,:])
        axes[0, 0].set_title('Plot 1')

        axes[0, 1].plot(self.sG_filter(data[5,1,:]))
        axes[0, 1].set_title('Plot 2')

        axes[1, 0].plot(self.smoothed_nonlinear_energy_operator(data[5,1,:].reshape(1,-1)).reshape(-1))
        axes[1, 0].set_title('Plot 3')

        axes[1, 1].plot(data[5,1,:])
        axes[1, 1].set_title('Plot 4')

        plt.savefig('test.png')
        plt.close()

        
        
    

# train_data = EDFDataset('/data/lvsizhe/0718/', 13)
# print(train_data.raw_data.shape)