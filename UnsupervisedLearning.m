%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                                                                   %%%%
%%%%                       Unsupervised learning                       %%%%
%%%%                                                                   %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear all
% 
% ShapeDataFile='Shape_Nel_Ncl.mat';
% StressDataFile='ECAP_Nel_Ncl.mat';
% TempDataFile='TempData.mat';
% OutputDataFile='TempData.mat';
% 
% IdxList_train=[40,24,86,51,8,108,127,96,73,7,60,10,89,94,30,33,2,59,50,26,22,101,48,68,121,122,91,16,90,118,13,99,76,43,15,93,95,45,3,100,6,92,52,97,62,111,54,85,125,27,18,105,11,66,75,63,84,61,1,56,78,110,42,41,4,17,38,5,53,116,71,128,34,28,55,35,23,74,31,119,57,98,109,32,107,14,106,19,29,49,104,82,124,126,79,69,80,20,120,72,77,25,37,81,112,46,115,39,102,65,58,12,113,88,70];
%     
% IdxList_test=[87,36,114,21,83,9,103,123,67,64,117,47,44];


function Result=UnsupervisedLearning(OutputDataFile, ShapeDataFile, StressDataFile,IdxList_train, IdxList_test,SV_Shape,SV_Stress,nNod)
%% Initialization
Result=0;

%% Load Data Files
load(ShapeDataFile)
load(StressDataFile)

%% Shape encoding

ShapeData_train=ShapeData(:,IdxList_train); %% Training shape data
ShapeData_test=ShapeData(:,IdxList_test);   %% Testing shape data

MeanShape=mean(ShapeData_train,2);  %% Mean shape
X=ShapeData_train-MeanShape; %% Substract Mean Shape

X=X/sqrt(length(IdxList_train)); %% Don't understand why?

[U, S, V]=svd(X); %%% Singular value decomposition
Lambda=diag(S); %% Singular Values

V123=sum(Lambda(1:SV_Shape).^2)/sum(Lambda.^2); %% Info retained 

PC_count=SV_Shape;
PC=U(:,1:PC_count);
Proj=zeros(nNod*3,PC_count);

for k=1:PC_count
    Proj(:,k)=U(:,k)/Lambda(k);
end

ShapeCode_train=zeros(PC_count,length(IdxList_train));
for k=1:length(IdxList_train)
    temp=ShapeData_train(:,k)-MeanShape;
    c=zeros(1,PC_count);
    for n=1:PC_count
        c(n)=sum(PC(:,n).*temp(:))/Lambda(n);
    end
    ShapeCode_train(:,k)=c;
end

ShapeCode_test=zeros(PC_count,length(IdxList_test));
for k=1:length(IdxList_test)
    temp=ShapeData_test(:,k)-MeanShape;
    c=zeros(1,PC_count);
    for n=1:PC_count
        c(n)=sum(PC(:,n).*temp(:))/Lambda(n);
    end
    ShapeCode_test(:,k)=c;
end

%% ECAP encoding

%% Stress PCA

StressData_train=StressData(:,IdxList_train);
StressData_test=StressData(:,IdxList_test);

MeanStress=mean(StressData_train,2);  %% Mean shape
X2=StressData_train-MeanStress; %% Substract Mean Shape

X2=X2/sqrt(length(IdxList_train));
%X2=normalize(X2);

[U2, S2, V2]=svd(X2); %%% Singular value decomposition
Lambda2=diag(S2); %%% Singular Values

V123_2=sum(Lambda2(1:SV_Stress).^2)/sum(Lambda2.^2); %%info retained 

PC_count2=SV_Stress;
PC2=U2(:,1:PC_count2);

Proj2=[];
for k=1:PC_count2
    Proj2(:,k)=U2(:,k)/Lambda2(k);
end

StressCode_train=zeros(PC_count2,length(IdxList_train));
for k=1:length(IdxList_train)
    temp=StressData_train(:,k)-MeanStress;
    c2=zeros(1,PC_count2);
    for n=1:PC_count2
        c2(n)=sum(PC2(:,n).*temp(:))/Lambda2(n);
    end
    StressCode_train(:,k)=c2;
end

StressCode_test=zeros(PC_count2,length(IdxList_test));
for k=1:length(IdxList_test)
    temp=StressData_test(:,k)-MeanStress;
    c=zeros(1,PC_count2);
    for n=1:PC_count2
        c(n)=sum(PC2(:,n).*temp(:))/Lambda2(n);
    end
    StressCode_test(:,k)=c;
end

EigenValues=Lambda2(1:SV_Stress);
EigenVectors=PC2;

%% Save
save(OutputDataFile, 'MeanShape','MeanStress','Proj','Proj2','ShapeCode_train', 'ShapeCode_test', 'StressData_train','StressData_test', ...
    'StressCode_train','StressCode_test','OutputDataFile', 'ShapeDataFile', 'StressDataFile',...
    'IdxList_train', 'IdxList_test','EigenValues','EigenVectors');
Result=1;



 