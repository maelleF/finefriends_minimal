import os, torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import r2_score

#custom class dataset(Dataset):
class encoding_dataset(Dataset):
    def __init__(self, data):
        self.X_data = [x for (x, y) in data]
        self.Y_data = [y for (x, y) in data]
    
    def __len__(self):
        return len(self.X_data)
    
    #def create_batches(self)
    def __getitem__(self, idx):
        return self.X_data[idx], self.Y_data[idx]

class soundnet_dataset(encoding_dataset):
    def __init__(self, data):
        super().__init__(data)
        
    def convert_input_to_tensor(self):
        self.X_data = [torch.Tensor(x).view(1,-1,1) for x in self.X_data]
    
    def __tr_alignment__(self, y, nb_tr, cut='end'):
        #depend on the output layer
        #for layer conv7
        if cut == 'start':
            y = y[len(y)-nb_tr:,:]
        elif cut == 'end':
            y = y[:nb_tr,:]
        return(y)

    def redimension_output(self, Y_pred, Y_real, cut='end'):
        Y_pred_converted = Y_pred.permute(2,1,0,3).squeeze(axis=(2,3)).numpy()
        Y_real_converted = Y_real.squeeze(axis=0).numpy() 
        if len(Y_pred_converted) > len(Y_real_converted):
            #print('redimension prediction outputs to real outputs')
            Y_pred_converted = self.__tr_alignment__(Y_pred_converted, nb_tr=len(Y_real_converted), cut=cut)
        
        elif len(Y_pred_converted) < len(Y_real_converted):
            #print('redimension real outputs to predicted outputs')
            Y_real_converted = self.__tr_alignment__(Y_real_converted, nb_tr=len(Y_pred_converted), cut=cut)

        return(Y_pred_converted, Y_real_converted)

#-----------------------training-utils-----------------------------------
def test(dataloader, net, return_nodes=None):
    #put your network in evaluation mode
    net.eval()

    if return_nodes is not None:
        #useful case where you want to access internal embedding of the model (need a specific model)
        out_p = {layer_name:[] for layer_name in return_nodes.values()}
    else:
        #usual case where you just want to evaluate the output of the network
        out_p = []
        
    #avoid gradient descent
    with torch.no_grad():
        for (input, output) in dataloader:
            output_predicted = net(input)

            if return_nodes is not None:
                for key, prediction in output_predicted.items():
                    out_p[key].append((prediction, output))
            else:
                out_p.append((prediction, output))
    return out_p

#for example only, not use in this current state
def train(trainloader,net,optimizer, epoch, mseloss, delta=1e-2, gpu=True):
    all_y = []
    all_y_predicted = []
    running_loss = 0

    #inform pytorch to be in Training mode
    net.train()

    for batch_nb, (x, y) in enumerate(trainloader):
        optimizer.zero_grad()
        batch_size = x.shape[0]
        if gpu : 
            x = x.cuda()
        # Forward pass
        predicted_y = net(x)
        predicted_y = predicted_y.permute(2,1,0,3).squeeze().double()
        
        predicted_y = predicted_y[:batch_size]                #FOR AUDIO ONLY : Cropping the end of the predicted fmri to match the measured bold
        
        y = y.double()
        
        if gpu:
            y = y.cuda()
        #print(f"y_real shape : ", y.shape, "and y_predicted shape : ", predicted_y.shape)         # both must have the same shape
        loss=delta*mseloss(predicted_y,y)/batch_size
        
        #gradient descent
        loss.backward()
        optimizer.step()

        #results for each batch in one array, to evaluate the r2 score in one epoch
        all_y.append(y.cpu().numpy().reshape(batch_size,-1))
        all_y_predicted.append(predicted_y.detach().cpu().numpy().reshape(batch_size,-1))
        running_loss += loss.item()
        
    r2_model = r2_score(np.vstack(all_y),np.vstack(all_y_predicted),multioutput='raw_values') 
    return running_loss/batch_nb, r2_model