#%% Dimensionality reduction network 

#%% Import all required libraries and dependencies
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy
import math
from numpy.random import RandomState
import scipy.io as sio
import time
import matlab.engine
from matplotlib import pyplot

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
#------------ A is ground truth, B is the DL prediction of A
    
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
    
def ComputePercentileError(A, B):
    
    AE90=numpy.zeros(A.shape[1])
    APE90=numpy.zeros(A.shape[1])

    AE99=numpy.zeros(A.shape[1])
    APE99=numpy.zeros(A.shape[1])
    
    for n in range(0, A.shape[1]):
        a=A[:,n]
        b=B[:,n]
        a_90=numpy.percentile(a,90)
        a_99=numpy.percentile(a,99)
        
        b_90=numpy.percentile(b,90)
        b_99=numpy.percentile(b,99)
        
        AE90[n]=numpy.absolute(a_90-b_90)
        APE90[n]=AE90[n]/a_90
        
        AE99[n]=numpy.absolute(a_99-b_99)
        APE99[n]=AE99[n]/a_99
        
    return AE90, AE99, APE90,APE99
       
def RelativeAbsoluteError(Y_t, Yp):
    
    RAE=(numpy.mean(numpy.absolute(Y_t-Yp)))/(numpy.mean(numpy.absolute(numpy.mean(Y_t)-Y_t)))

    return RAE
#end

#%% Define the fully connected network to perform the non linear mapping

    
#-----------------------------------
def CreateModel_NonlinearMapping(Xshape, Yshape):
    model = Sequential()
    model.add(Dense(nDense, input_dim=Xshape[1], kernel_initializer='normal', activation=Act))

    model.add(Dense(nDense, kernel_initializer='normal', activation=Act))
    model.add(Dense(nDense, kernel_initializer='normal', activation=Act))
    model.add(Dense(nDense, kernel_initializer='normal', activation=Act))
    #model.add(Dense(nDense, kernel_initializer='normal', activation=Act))
    
    model.add(Dense(Yshape[1], kernel_initializer='normal', activation='linear'))
    
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


#-----------------------------------
    
# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.7
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

#%% Load the data

#Data  containing the spatial (x,y,z) coordinates of the nodes for all geometries
ShapeDataFile='Shape_Final.mat' # Rows correspond to the spatial coordinates of the nodes for each geometry
                                # following the order x1,y1,z1,x2,y2,z2,...,xn,yn,zn

#Data containing the ECAP grounf truth from CFD simulations for all geometries
StressDataFile='ECAP_Final.mat' # Rows correspond to the ECAP values for each node of the mesh while the 
                                # columns represent each geometry on the dataset


TempDataFile='TempData.mat' # File where all the MATLAB unsupervised learning data is going to be stored
ResultFile='DL_ECAP_result.mat' # Name of the file with the results of the DL analysis

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

# Batch size
batchS=20 

# Number of nodes
nDense=512
# Normalization
Drop=0

# Number of epochs, tested best value to avoid overfitting 
nEpoch=300

# Activation unit type
Act='relu'

#%% Initialize all the data

eng = matlab.engine.start_matlab() # Start the MATLAB engine
rng=RandomState(42) ## Maintain the same seed to be able to analize algorithm improvements
IndexList= numpy.arange(0, nSim, 1) #Lista de numero de simulaciones

# Initialize matrices for the accuracy results
ECodeMAE=[]; ECodeNMAE=[]; # Mean absolute error at the low dimensional scalars
ECodeAE=[]; ECodeAPE=[]; # Absolute error at the low dimensional scalars

DifMAE=[];DifNMAE=[];

ECAPMAE=[]; ECAPNMAE=[]; # Mean absolute error at predicted ECAP maps
ECAPAE=[]; ECAPAPE=[]; # Absolute error at predicted ECAP maps

IndexList_test=[]; # List of testing dataset
IndexList_train=[]; # List of training dataset
    
rng.shuffle(IndexList) # Randomize the simulations
idx_train=IndexList[0:nTrain] # Train set
idx_test=IndexList[nTrain:nSim] # Test set
IndexList_train.append(idx_train)  # Añade los indices nada mas
IndexList_test.append(idx_test)     # Añade los indices nada mas
ShapeData_train=ShapeData[:,idx_train]  # Save training Shape data
ShapeData_test=ShapeData[:,idx_test]     # Save testing Shape data
StressData_train=StressData[:,idx_train]  # Save training Stress data
StressData_test=StressData[:,idx_test]     # Save testing Stress data

#%% Truncated - PCA for dimensionality reduction

t1=time.perf_counter()

[X, X_t, Y, Y_t, S, S_t,Proj,Proj2, MeanShape,MeanStress,EValues,EVectors]=UnsupervisedLearning(TempDataFile, ShapeDataFile, StressDataFile, idx_train, idx_test,SV_Shape,SV_Stress,nNodes)

t2=time.perf_counter()
print('Unsupervised Learning', t2-t1)


#%% Non-linear mapping

t3=time.perf_counter()


# Create the neural network model and perform the non linear mapping
NMapper=CreateModel_NonlinearMapping(X.shape, Y.shape)

# Training of the network
history = NMapper.fit(X, Y, epochs=nEpoch, batch_size=batchS, verbose=0)

# Predict the low dimensional ECAP representations
Yp=NMapper.predict(X_t, batch_size=idx_test.size, verbose=0)


t4=time.perf_counter()
print('Nonlinear Mapping', t4-t3)



#%% ECAP decoding through PCA reconstruction

Sp=numpy.zeros([nNodes,idx_test.size]);

for k in range(0,idx_test.size):
    temp=numpy.zeros([nNodes,SV_Stress])
    for n in range(0,SV_Stress):
      temp[:,n]=(Yp[k,n]*EValues[n])*EVectors[:,n]
    q=temp.sum(axis=1)
    Sp[:,k]=q+MeanStress.reshape((nNodes), order='F')
#end

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

[AE90,AE99,APE90,APE99]=ComputePercentileError(S_t,Sp)


# Mean Error
mae=mean_absolute_error(S_t,Sp)
rmse=math.sqrt(mean_squared_error(S_t,Sp))
nmae=mae/numpy.max(S_t)*100
rae=RelativeAbsoluteError(S_t,Sp)

maeC=mean_absolute_error(Y_t,Yp)
rmseC=math.sqrt(mean_squared_error(Y_t,Yp))


print('Accuracy')
print('MAE: ',mae)
print('RAE: ',rae*100)
print('RMSE: ',rmse)
print('')
print('Accuracy Code')
print('MAE_Code: ',maeC)
print('RMSE_Code: ',rmseC)
print('')
print('Metrics')
print('mae: ', numpy.mean(history.history['mean_absolute_error'][-200:]))
print('mse: ', numpy.mean(history.history['mean_squared_error'][-200:]))


print('Percentiles')
print('ECAPpeak', numpy.mean(ECAPAE), numpy.std(ECAPAE), numpy.mean(ECAPAPE), numpy.std(ECAPAPE))
print('ECAPpeak99', numpy.mean(AE99), numpy.std(AE99), numpy.mean(APE99), numpy.std(APE99))
print('ECAPpeak90', numpy.mean(AE90), numpy.std(AE90), numpy.mean(APE90), numpy.std(APE90))
#report
#t6=time.perf_counter()
#print('ComputeError')


#end


#%% Save data to .mat

sio.savemat(ResultFile,
           {'Sp':Sp,'S_t':S_t,'IndexList_test':IndexList_test,'X':X,'X_t':X_t,'Y':Y,'Y_t':Y_t,'Yp':Yp,'MeanStress':MeanStress,'EValues':EValues,'EVectors':EVectors})
#             'IndexList_test':IndexList_test, 'IndexList_train':IndexList_train,
#             'ECAPMAE':ECAPMAE, 'ECAPNMAE':ECAPNMAE,
#             'ECAPAE':ECAPAE,'ECAPAPE':ECAPAPE})
    
#%%# plot metrics

mse=pyplot.plot(history.history['mean_squared_error'],label="mse")
mae=pyplot.plot(history.history['mean_absolute_error'],label="mae")
#mape=pyplot.plot(history.history['mean_absolute_percentage_error'],label="mape")
#cosine=pyplot.plot(history.history['cosine_proximity'],label="cosine")

pyplot.title(label=str(nDense))
pyplot.ylim((0,0.2))   # set the ylim to bottom, top
pyplot.legend()
pyplot.show()