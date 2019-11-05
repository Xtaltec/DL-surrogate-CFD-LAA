%% Plot the ECAP of result %%

load('DL_ECAP_result.mat') %% Load ECAP prediction file for single training
load('Shape_Final.mat') %% Load Shape Ground Truth
load('ECAP_Final.mat') %% Load ECAP Ground Truth

id=IndexList_test+1; %% Sum 1 to indexlist so that it works properly in MATLAB

[nNod,nSim]=size(StressData); %% [Number of nodes, Number of instances]

vShapeData=zeros(nNod,3,nSim); %% 3D array with the x,y,z coordinates separated in 3 columns
vShapeData(:,1,:)=ShapeData(1:3:(nNod*3),:);
vShapeData(:,2,:)=ShapeData(2:3:(nNod*3),:);
vShapeData(:,3,:)=ShapeData(3:3:(nNod*3),:);

%% PLOTTING - Ground Truth
close all

for i=1:length(IndexList_test)

y=id(i);
figure()
%%% Ground Truth
subplot(1,3,1)
%rgb_GTruth=vals2colormap(S_t(:,i),'jet',[0,6]);
scatter3(vShapeData(:,1,y),vShapeData(:,2,y),vShapeData(:,3,y),30,S_t(:,i),'filled','MarkerEdgeColor','k')
title('Ground Truth')
colormap(jet);
caxis([0, 6])


%%% Predicted
subplot(1,3,2)
%rgb_Predict=vals2colormap(StressReconstruction(:,i),'jet',[0,6]);
scatter3(vShapeData(:,1,y),vShapeData(:,2,y),vShapeData(:,3,y),30,StressReconstruction(:,i),'filled','MarkerEdgeColor','k')
title('Prediction')
colormap(jet);
caxis([0, 6])

%%% Difference
subplot(1,3,3)
Dif=abs(Sp(:,i)-S_t(:,i));
%rgb_Dif_M=vals2colormap(Dif,'jet',[0,d]);
scatter3(vShapeData(:,1,y),vShapeData(:,2,y),vShapeData(:,3,y),30,Dif,'filled','MarkerEdgeColor','k')
title(['Matrix dif: ',num2str(mean(Dif))])
colormap(jet);
caxis([0, 6])

end

%% Wrong ECAP:8,9,21,23,41,43,44,45,78,101,130,155
%% Left but better check: 26,163,178
%%
%  for i=151:200
%      figure()
%      rgb_GTruth=vals2colormap(StressData(:,i),'jet',[0,6]);
%      scatter3(vShapeData(:,1,i),vShapeData(:,2,i),vShapeData(:,3,i),30,rgb_GTruth,'filled','MarkerEdgeColor','k')
%      title(num2str(i))
%  end

%% Node and average difference per ECAP value

v1_t=length(find(S_t<=1));
v2_t=length(find(S_t<=2))-v1_t;
v3_t=length(find(S_t<=3))-v2_t-v1_t;
v4_t=length(find(S_t<=4))-v3_t-v2_t-v1_t;
v5_t=length(find(S_t<=5))-v4_t-v3_t-v2_t-v1_t;
v6_t=length(find(S_t<=6))-v5_t-v4_t-v3_t-v2_t-v1_t;
vm_t=length(find(S_t>6));

v1_p=length(find(Sp<=1));
v2_p=length(find(Sp<=2))-v1_p;
v3_p=length(find(Sp<=3))-v2_p-v1_p;
v4_p=length(find(Sp<=4))-v3_p-v2_p-v1_p;
v5_p=length(find(Sp<=5))-v4_p-v3_p-v2_p-v1_p;
v6_p=length(find(Sp<=6))-v5_p-v4_p-v3_p-v2_p-v1_p;
vm_p=length(find(Sp>6));

dif_v1dif=abs(v1_t-v1_p)/v1_t;
dif_v2=abs(v2_t-v2_p)/v2_t;
dif_v3=abs(v3_t-v3_p)/v3_t;
dif_v4=abs(v4_t-v4_p)/v4_t;
dif_v5=abs(v5_t-v5_p)/v5_t;
dif_v6=abs(v6_t-v6_p)/v6_t;
dif_vm=abs(vm_t-vm_p)/vm_t;

%% Threshold for high ECAP values

%%% A threshold is chosen above which the nodes are considered to have high
%%% risk of thrombus formation
T=4;

bDif=zeros(1,length(IndexList_test));
sDif=zeros(1,length(IndexList_test));
nBig=zeros(1,length(IndexList_test));
nSm=zeros(1,length(IndexList_test));
nBig_P=zeros(1,length(IndexList_test));
nSm_P=zeros(1,length(IndexList_test));

for i=1:length(IndexList_test)

    big=find(S_t(:,i)>T); %% Id of nodes at risk
    sm=find(S_t(:,i)<=T); %% Id of nodes not at risk
    
    nBig(i)=length(big); %% Total nodes at risk
    nSm(i)=length(sm); %% Total nodes safe
    
    nBig_P(i)=length(find(Sp(big,i)>T)); %% Total predicted nodes at risk
    nSm_P(i)=length(find(Sp(sm,i)<=T)); %% Total predicted save nodes
    
    bDif(i)=mean(abs(S_t(big,i)-Sp(big,i))); %% MAE of nodes at risk
    sDif(i)=mean(abs(S_t(sm,i)-Sp(sm,i)));  %% MAE nodes not at risk
end


nBig_D=abs((nBig-nBig_P))./nBig; %% Normalized difference of nodes at risk
Predicted_risk=1-nBig_D; %% Percentage of nodes predicted in risk
Predicted_risk_mean=mean(Predicted_risk); %%Percentage of the nodes detected at risk is also predicted