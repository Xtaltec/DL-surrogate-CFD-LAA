

#%% 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy
import math
from numpy.random import RandomState
import scipy.io as sio
import time
import matlab.engine

#%% Unsupervised Learning is done in Matlab
def UnsupervisedLearning(DataFile, ShapeDataFile, StressDataFile, IdxList_train, IdxList_test,SV_Train,SV_Test,nNodes):
    idx_train_mat=matlab.double(list(IdxList_train+1)) #+1 to matlab index
    idx_test_mat=matlab.double(list(IdxList_test+1)) #+1 to matlab index
    DataFlag = eng.UnsupervisedLearning(DataFile, ShapeDataFile, StressDataFile, idx_train_mat,idx_test_mat,SV_Shape,SV_Stress,nNodes)
    MatData=sio.loadmat(DataFile)
    
    X=MatData['ShapeCode_train']
    X=numpy.asmatrix(X)
    X=X.transpose()


    X_t=MatData['ShapeCode_test']
    X_t=numpy.asmatrix(X_t)
    X_t=X_t.transpose()
  
    Y= MatData['StressCode_train']
    Y=numpy.asmatrix(Y)
    Y=Y.transpose()

    Y_t= MatData['StressCode_test']
    Y_t=numpy.asmatrix(Y_t)
    Y_t=Y_t.transpose()
    
    S=MatData['StressData_train']
    S=numpy.asmatrix(S)

    S_t=MatData['StressData_test']
    S_t=numpy.asmatrix(S_t)
    #
    #
    Proj=MatData['Proj']
    Proj2=MatData['Proj2']

    MeanShape=MatData['MeanShape']
    MeanStress=MatData['MeanStress']
    
    EValues=MatData['EigenValues']
    EVectors=MatData['EigenVectors']
    
    
    
    return X, X_t, Y, Y_t, S, S_t,Proj,Proj2, MeanShape,MeanStress,EValues,EVectors
#end

#%% Error calculation
#-----------------------------------
#------------ A is ground truth, A[:,n] is vector of stress values; B is the reonstructed version of A

def MeanError(A,B):
    Abs=numpy.absolute(A-B)
    Mean=numpy.mean(Abs)
    MeanC=numpy.mean(numpy.mean(Abs,axis=0))
    mse = mean_squared_error(A,B)
    return Mean,MeanC,mse
#end
    
def ComputeError(A, B):
    MAE=numpy.zeros(A.shape[1])
    NMAE=numpy.zeros(A.shape[1])
    for n in range(0, A.shape[1]):
        a=A[:,n]
        b=B[:,n]
        c=numpy.absolute(a-b)
        a_abs=numpy.absolute(a)
        #a_max=numpy.max(a_abs[301:4700])
        a_max=numpy.max(a_abs)
        MAE[n]=numpy.mean(c)
        NMAE[n]=MAE[n]/a_max
    #end
    return MAE, NMAE
#end
    
#------------
def ComputeError_peak(A, B):
    AE=numpy.zeros(A.shape[1])
    APE=numpy.zeros(A.shape[1])
    for n in range(0, A.shape[1]):
        a=A[:,n]
        b=B[:,n]
        a_abs=numpy.absolute(a)
        b_abs=numpy.absolute(b)
        a_max=numpy.max(a_abs)
        b_max=numpy.max(b_abs)
        AE[n]=numpy.absolute(a_max-b_max)
        APE[n]=AE[n]/a_max
    #end
    return AE, APE
#end
    
def RelativeAbsoluteError(Y_t, Yp):
    
    RAE=(numpy.mean(numpy.absolute(Y_t-Yp)))/(numpy.mean(numpy.absolute(numpy.mean(Y_t)-Y_t)))

    return RAE
#end


#%% Define the Deep Learning Model: shape_encoding, nonlinear mapping, stress_decoding
#https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
    
#-----------------------------------
def CreateModel_NonlinearMapping(Xshape, Yshape):
    model = Sequential()
    model.add(Dense(5000, input_dim=Xshape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(5000, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(5000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(Yshape[1], kernel_initializer='normal', activation='linear'))

    #lr = 0.1
    #decay_rate = lr / nEpoch
    #adam=optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae','mape','cosine'])
    return model
#end
    
#-----------------------------------
#%% Learning rate scheduler
    
def lr_scheduler(epoch, lr):
    decay_rate = 0.01
    decay_step = 90
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.003
	drop = 0.7
	epochs_drop = 12.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

#-----------------------------------
#%%

ShapeDataFile='Shape_Final.mat'
StressDataFile='ECAP_Final.mat'


TempDataFile='TempData.mat'
ResultFile='DL_ECAP_SFC_CrossValidation.mat'

#Load Shape data
MatData_shape=sio.loadmat(ShapeDataFile)
ShapeData=MatData_shape['ShapeData']

#Load Stress Data
MatData_shape=sio.loadmat(StressDataFile)
StressData=MatData_shape['StressData']

# Initialize Variables
nNodes=StressData.shape[0]; #Number of nodes in geometry
SV_Shape=32; #Retained Single Values of Shape
SV_Stress=32; #Retained Single Values of Stress

nSim=ShapeData.shape[1]; # Total number of simulations
nTrain= round(nSim*0.9); # Number of training data
nTest=nSim-nTrain; # Number of testing data

#Hyperparameters
batchS=20

##Number of nodes
nDense=7398
nDense1=5000
nDense2=2466
## Normalization
Drop=0

## Tested best value to avoid overfitting look when writing on: https://github.com/keras-team/keras/issues/1971
nEpoch=200


#%%
#make sure you can see DLStress.py in the current directory of spyder

eng = matlab.engine.start_matlab()

IndexList= numpy.arange(0, nSim, 1) #Lista de numero de simulaciones

## We only have ECAP
ECodeMAE=[]; ECodeNMAE=[];
ECodeAE=[]; ECodeAPE=[];

DifMAE=[];DifNMAE=[];

ECAPMAE=[]; ECAPNMAE=[];
ECAPAE=[]; ECAPAPE=[];

IndexList_test=[];
IndexList_train=[];
mae=numpy.zeros([100,1])
rmse=numpy.zeros([100,1])
nmae=numpy.zeros([100,1])
rae=numpy.zeros([100,1])
maeC=numpy.zeros([100,1])
rmseC=numpy.zeros([100,1])
    
for i in range(0, 100):

    rng=RandomState()
    rng.shuffle(IndexList) # Randomize the simulations
    idx_train=IndexList[0:nTrain] # Train set
    idx_test=IndexList[nTrain:nSim] # Test set
    IndexList_train.append(idx_train)  # Añade los indices nada mas
    IndexList_test.append(idx_test)     # Añade los indices nada mas
    X=ShapeData[:,idx_train]  # Guardar los datos de shape para train
    X_t=ShapeData[:,idx_test]     # Guardar los datos de shape para test
    Y=StressData[:,idx_train]  # Guardar los datos de shape para train
    Y_t=StressData[:,idx_test]     # Guardar los datos de shape para test
    
    X=X.transpose()
    X_t=X_t.transpose()
    Y=Y.transpose()
    Y_t=Y_t.transpose()
    
    #%% Unsupervised learning in Matlab
    
    
    
    #%% Non-linear mapping
    
    t3=time.perf_counter()
    
    ## Normalize
    #scaler = StandardScaler()
    #X= scaler.fit_transform(X) 
    #X_t=scaler.fit_transform(X_t) 
    #Y=scaler.fit_transform(Y) 
    
    
    #rlrop = ReduceLROnPlateau(monitor='mean_squared_error', factor=0.2, patience=150)
    
    NMapper=CreateModel_NonlinearMapping(X.shape, Y.shape)
    history = NMapper.fit(X, Y, epochs=nEpoch, batch_size=batchS, verbose=0)
    #history = NMapper.fit(X, Y, epochs=nEpoch, batch_size=batchS,callbacks=[rlrop], verbose=0)
    
    Yp=NMapper.predict(X_t, batch_size=idx_test.size, verbose=0)
    #Yp=scaler.inverse_transform(Yp1)
    
    t4=time.perf_counter()
    print('Nonlinear Mapping', t4-t3)
    
    S_t=Y_t.transpose()
    Sp=Yp.transpose()
    
    #%% Compute error
    
    #compare ground-truth S and predicted Sp
        
    #Mean_Net=MeanError(S_t, Sp)
    [Mean_Total,Mean_Col,MSE]=MeanError(S_t,Sp)
    Mean_Code=MeanError(Y_t,Yp)
    #Mean_Dif=MeanError(Sp,StressReconstruction)
    
    # Code error
    [ECodeMAE_k, ECodeNMAE_k]=ComputeError(Y_t, Yp)
    ECodeMAE.append(ECodeMAE_k)
    ECodeNMAE.append(ECodeNMAE_k)
    
    #compare ground-truth S and predicted Sp
    [ECAPMAE_k, ECAPNMAE_k]=ComputeError(S_t, Sp)
    ECAPMAE.append(ECAPMAE_k)
    ECAPNMAE.append(ECAPNMAE_k)
       
    #peak stress error
    [ECAPAE_k, ECAPAPE_k]=ComputeError_peak(S_t, Sp)
    ECAPAE.append(ECAPAE_k)
    ECAPAPE.append(ECAPAPE_k)
    
    # Mean Error
    mae[i]=mean_absolute_error(S_t,Sp)
    rmse[i]=math.sqrt(mean_squared_error(S_t,Sp))
    nmae[i]=mae[i]/numpy.max(S_t)*100
    rae[i]=RelativeAbsoluteError(S_t,Sp)
    
    maeC[i]=mean_absolute_error(Y_t,Yp)
    rmseC[i]=math.sqrt(mean_squared_error(Y_t,Yp))
    
    print('Iteration:', i)
    print('MAE: ',mae[i],'RAE: ',rae[i]*100,'RMSE: ',rmse[i])

    #report
    #t6=time.perf_counter()
    #print('ComputeError')
    #print('ECode', numpy.mean(ECodeMAE), numpy.std(ECodeMAE), numpy.mean(ECodeNMAE), numpy.std(ECodeNMAE))
    #print('ECAP', numpy.mean(ECAPMAE), numpy.std(ECAPMAE), numpy.mean(ECAPNMAE), numpy.std(ECAPNMAE))
    #print('ECAPpeak', numpy.mean(ECAPAE), numpy.std(ECAPAE), numpy.mean(ECAPAPE), numpy.std(ECAPAPE))
    #end


#%% Save data
mae_M=numpy.mean(mae)
rmse_M=numpy.mean(rmse)
nmae_M=numpy.mean(nmae)
rae_M=numpy.mean(rae)
maeC_M=numpy.mean(maeC)
rmseC_M=numpy.mean(rmseC)

mae_sd=numpy.std(mae)
rmse_sd=numpy.std(rmse)
nmae_sd=numpy.std(nmae)
rae_sd=numpy.std(rae)
maeC_sd=numpy.std(maeC)
rmseC_sd=numpy.std(rmseC)


print('Total error')
print('MAE: ',mae_M)
print('RMSE: ',rmse_M)
print('NMAE: ',nmae_M)
print('RAE: ',rae_M*100)
print('MAEC: ',maeC_M)
print('RMSEC: ',rmseC_M)

print('Total error')
print('MAE: ',mae_sd)
print('RMSE: ',rmse_sd)
print('NMAE: ',nmae_sd)
print('RAE: ',rae_sd*100)
print('MAEC: ',maeC_sd)
print('RMSEC: ',rmseC_sd)


sio.savemat(ResultFile,
           {'mae':mae,'rmse':rmse,'nmae':nmae,'rae':rae,'maeC':maeC,'rmseC':rmseC,
            'mae_M':mae_M,'rmse_M':rmse_M,'nmae_M':nmae_M,'rae_M':rae_M,'maeC_M':maeC_M,'rmseC_M':rmseC_M,
            'mae_sd':mae_sd,'rmse_sd':rmse_sd,'nmae_sd':nmae_sd,'rae_sd':rae_sd,'maeC_sd':maeC_sd,'rmseC_sd':rmseC_sd})
#             'IndexList_test':IndexList_test, 'IndexList_train':IndexList_train,
#             'ECAPMAE':ECAPMAE, 'ECAPNMAE':ECAPNMAE,
#             'ECAPAE':ECAPAE,'ECAPAPE':ECAPAPE})