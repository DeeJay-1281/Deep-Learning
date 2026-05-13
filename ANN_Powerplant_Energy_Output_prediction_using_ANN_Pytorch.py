import pandas as pd
import numpy as np

#importing data & defining x,y
df=pd.read_csv("powerplant_data.csv")#data loading & preprocessing
x=df.drop("PE",axis=1)
y=df["PE"]

from sklearn.model_selection import train_test_split #data split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler #feature scaling
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

import torch
import torch.nn as nn

x_train_tensor=torch.tensor(x_train_scaled,dtype=torch.float32) #convert numpy data to tensors
x_test_tensor=torch.tensor(x_test_scaled,dtype=torch.float32)
y_train_tensor=torch.tensor(y_train.values,dtype=torch.float32).view(-1,1)#reshape tensor in 2d col vector
y_test_tensor=torch.tensor(y_test.values,dtype=torch.float32).view(-1,1)

#2 hindden layer,6 neurons in each
#data loader class-- creats the batches for us

from torch.utils.data import TensorDataset,DataLoader

train_dataset=TensorDataset(x_train_tensor,y_train_tensor)#combine feature & labels
test_dataset=TensorDataset(x_test_tensor,y_test_tensor)


train_loader =DataLoader(train_dataset,batch_size=32,shuffle=True)#slice data in batches for propogation
test_loader =DataLoader(test_dataset,batch_size=32)


#----------------DEEP LEARNING --build ANN module---------------

class ANN(nn.Module):#nn.modle is paent class
    def __init__(self):
        super(ANN,self).__init__()#super enables ANN to ue properties of nn.modele

        self.model=nn.Sequential( #sequential->layer by layer
            #1st layer
            nn.Linear(x_train.shape[1],6),
            nn.ReLU(),
            #2nd layer
            nn.Linear(6,6),
            nn.ReLU(),

            #output layer
            nn.Linear(6,1),
        )

    def forward(self,x):#forward propogation
        return self.model(x)
    
#----------------define loss+ optimizer---------------
    
import torch.optim as optim 
model=ANN()#create neural network instance 
#loss,optimizer
criterion=nn.MSELoss()#loss function for reg,crossentropy for classification
optimizer=optim.Adam(model.parameters())#optimizer->udates weight

#---------------------training ANN model------------------------
epochs=100#100 iterations
train_losses=[]#stores training losses per epoch
validation_looses=[]
best_val_loss=float("inf")
for epoch in range(epochs):
    model.train()
    running_loss=0.0

    for xb,yb in train_loader: #xb=features of one batch , yb= labels of one batch
        optimizer.zero_grad()#clears previous grad values
        outputs=model(xb)#predict output,forward prop
        loss=criterion(outputs,yb)#computes loss
        loss.backward()
        optimizer.step()#update weight
        running_loss+=loss.item()#loss-->py flot
    epoch_train_loss=running_loss/len(train_loader)#calculate avg training loss
    train_losses.append(epoch_train_loss)

    #----------------------validation---------------------
    model.eval()
    runing_val_loss=0.0

    with torch.no_grad():# no grad compute
        for xb,yb in test_loader:
            outputs=model(xb)
            loss=criterion(outputs,yb)
            runing_val_loss+=loss.item()
    
    epoch_val_loss=runing_val_loss/len(test_loader)
    validation_looses.append(epoch_val_loss)

    #print(f"epoch {epoch+1}/{epochs}==>training loss {epoch_train_loss} & val loss= {epoch_val_loss}")
    if epoch_val_loss<best_val_loss:
        best_val_loss=epoch_val_loss
        torch.save(model.state_dict(),"best_model.pt")
    

import matplotlib.pyplot as plt
loss_df=pd.DataFrame({
    "train_loss":train_losses,
    "val_loss":validation_looses
})

plt.plot(loss_df["train_loss"],label="Training loss")
plt.plot(loss_df["val_loss"],label="validation loss")

plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend()
plt.show()

#saving & loading best model
#print(model.load_state_dict(torch.load("best_model.pt")))

model.eval()
with torch.no_grad():
    train_pred=model(x_train_tensor)
    test_pred=model(x_test_tensor)
    train_mse_loss=criterion(train_pred,y_train_tensor)
    test_mse_loss=criterion(test_pred,y_test_tensor)

print("Traing MSE : ",train_mse_loss.item())
print("Test MSE : ",test_mse_loss.item())

#evaluate model

from sklearn.metrics import r2_score
print("R2 SCORE ",r2_score(y_test,test_pred))

predicted_df=pd.DataFrame(test_pred.numpy(),columns=["Predicted"])
actual_df=pd.DataFrame(y_test.values,columns=["Actual"])
print(pd.concat([predicted_df,actual_df],axis=1))



