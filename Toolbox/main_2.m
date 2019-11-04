clear all;
clc;
%%
path ='D:\TFM\Guadalupe\OvalCFD2';
%% LO UNICO QUE CAMBIAR PARA EL C�CULO DE CASOS 
%CASES=['1']; %CASES should be added following% CASES=[ '411';'419'; '423';'425'; '426'; '427'; '428';'429'; '431';'432';'436']; 
CASES={'Oval'};
Domain=['1']
%Domain=['2'];% Domain=['1';'1';'1';'1';'1';'2';'1';'1';'1';'4';'1'];
GGI_VTKHaemoCreator_function(path,CASES,Domain);
cd 'D:\TFM\Guadalupe\OvalCFD2'
