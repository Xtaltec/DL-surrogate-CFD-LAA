function GGI_VTKHaemoCreator_function(path,CASES,Domain)
%% Guadalupe Garc�a Isla's PROGRAM FOR COMPUTING TAWSS, OSI, RRT & ECAP 

% This program computes TAWSS, OSI, RRT & ECAP haemodynamic values from the
% resulting WSS data from a CFD simulation. These data ought to be
% extracted  from the .encas files created with the CFD simulations with
% Paraview in .csv files for SYSTOLE (SYS) and DIASTOLE (DIA). 

% It computes the haemodynamic values and stores them in a .vtk file containing
% the structure of the tetrahedral mesh used for the CFD simulation.
% These parameters are computed for systole and diastole separately and
% then for both together. Therefore, separate .csv datasets are required for SYS and DIA. 

%IMPORTANT: For successfully running the program:
% 1- Introduce the name of the Cases between ' ' and separated by a ; in the variable "CASES" below. 
% 2- Introduce the domain of interest for each case in the variable
%    "Domain" FOLLOWING THE SAME ORDER OF CASES THAN IN "CASES". 

%% The only thing to be changed for computing the haemodynamic variables (LO �NICO QUE CAMBIAR PARA EL C�CULO DE CASOS) 

cd (path);
%CASES={'423_avmin'};    %CASES should be added following% CASES={'411';'419';'423';'425';'426';'427';'428';'429';'431';'432';'436'} or CASES={'425_avav';'423_avmax';'425_avmax';'429_avmin';'429_maxav';'423_maxmax';'429_maxmax';'436_minmin'} 
%Domain=['1'];                          % Domain=['1';'1';'1';'1';'1';'2';'1';'1';'1';'4';'1'];


%%_______IMPORTANT!! READ BEFORE STARTING________%%

%INGREDIENTS (input):
%   1- Data of the parameters product of the CFD simulation of each case point along time.
%       IMPORTANT: data should be stored in the same folder than this program
%       and should be exported followinng the following structure: 
%             CASENUMBER_SYS/DIA    example: 411_SYS or 411_DIA
%       automatically, paraview will generate 7 different files (0 to 6)
%       corresponding to each domain. In addition, for each domain, a
%       different file for each timestep will be generated (.timestep). So
%       at the end the file would be:
%       CASENUMBER_SYS/DIAdomain.timestep.csv   
%       example1: 411_SYS1.20.csv       (CASES= 411; Phase= SYS; Domain= 1; Timestep= 20)
%       example2: 423_avav_DIA2.30.csv  (CASES= 423_avav; Phase= DIA; Domain= 2; Timestep= 30)
% 
%       In the variable "CASES" store only the name/number of the case, as in the
%       examples 1 & 2 above ( CASES= 411; CASES= 423_avav). From
%       CASENUMBER_SYS/DIAdomain.timestep.csv only "CASENUMBER"


%   2- .vtk File corresponding to the case geometry.
%        IMPORTANT: it should be stored in the same folder of the
%        program with the name:
%               CASENUMBER_SURFACE.vtk    
%               example1: 411_SURFACE.vtk
%               example2: 423_avav_SURFACE.vtk
%        The vtk files should be of version 3.0 and should correspond to
%        the atrial surface domain (normally 1) 
%        
%        IMPORTANT: When you extract the tetrahedral mesh surface with 
%        Paraview it is exported in a .vtk file of version 4.1. This .vtk 
%        version cannot be read by Paraview itself (-.-'). So it should be 
%        converted to a 3.0 file BEFORE RUNNING THIS PROGRAM. HOW?:
%           1st- Open with notepad the .vtk file. 
%           2nd- Change from the first line the 4.1 to a 3.0 
%           3rd- Using the search tool, find the word "METADATA"
%           4th- Eliminate the whole paragraph under the title "METADATA",
%                which is above the section "POLYGONS". 
%                Example: from the file 411_SURFACE, it was eliminated:                       
%                         METADATA
%                         INFORMATION 1
%                         NAME L2_NORM_RANGE LOCATION vtkDataArray
%                         DATA 2 0.0111138 0.0662228
%          5th- Save the .vtk file without changing or eliminating
%          anything else.
%          6th- Move it to the folder with all the .csv files & this
%          Program. The .vtk file is Ready to be used!! 


% OUTPUT: The computed variables (TAWSS, OSI, RRT & ECAP) will be written
% into the .vtk files used as input with the name
%               CASENUMBER_SURFACE.vtk 
%               example1: 411_SURFACE.vtk
%               example2: 423_avav_SURFACE.vtk
% Open the files with Paraview for their visualization. 






% NOTHING AT ALL BELOW THIS SECCTION OUGHT TO BE CHANGED (UNLESS that WSS
% values in the .csv are not located among the columns defined in
% "%%Parameters definition"> "WSSi" & "WSSf". It has happened to me only
% once, if the output TAWSS, OSI, RRT and ECAP values follow extrelly weird
% patterns this may be the cause.)




%% Parameters definition 
%Load Data
WSSi=9; %WSSi and WSSf define the range of columns of the .csv files were
WSSf=12;%our data of interest is located (WSS values). I have done this to
        %store only the necessary data for lowering the computational time. 
%Haemo
WSS=1;
WSSx=2;
WSSy=3;
WSSz=4;
%SYS_DIA_SD={'SYS'; 'DIA'; 'BOTH'};
SYS_DIA_SD={'BOTH'};
%SS_SYS_DIA=[1 41 1;40 105 105];
SS_SYS_DIA=[1;105];
%SYS_DIA=[40 65 105];
SYS_DIA=[105];
%vtk
VARIABLES={ 'TAWSS'; 'OSI'; 'RRT'; 'ECAP'};
%SDstr={'Sys';'Dia';'Both'};
SDstr={'Both'};


%% Load Data  
number_Cases=length(CASES);
fprintf('Number of cases: %d', number_Cases)
cont=0;
for C=1:number_Cases % Each case
    fprintf('\nLoading Case %s\n',CASES{C})
    cont=0;
   % for SD=1:2 %Systole or Diastole 
   for SD=1
        fprintf('\t%s\n',SYS_DIA_SD{SD})
        for t=1:SYS_DIA(SD) % Each timestep 
            cont=cont+1;
            fprintf('\t\tC%s- Timestep %d/%d\n',CASES{C},t,SYS_DIA(SD))
            TS=num2str(t-1);
            %file_name=[ CASES{C} SYS_DIA_SD{SD} Domain(C) '.' TS '.csv'];
            file_name=[ CASES{C} Domain(C) '.' TS '.csv'];
           % tempo_cell{t,1} = csvread(file_name,1,0);   
           % CPOINTS{C,1}{cont,1}=[ tempo_cell{t,1}(:,WSSi:WSSf)];%Storing only the data of interest  
           tempo_cell{t,1} = readtable(file_name);
            
           % CPOINTS{C,1}{cont,1}=[ tempo_cell{t,1}(:,9:12) tempo_cell{t,1}(:,17:19)]; 
           CPOINTS{C,1}{cont,1}=[ tempo_cell{t,1}.wall_shear tempo_cell{t,1}.x_wall_shear tempo_cell{t,1}.y_wall_shear tempo_cell{t,1}.z_wall_shear ];
            
            
        end;
    end;
  CPOINTS{C,1}{106,1}=CASES{C}; % The name of the case is stored
end;

%% Calculation of haemodynamic parameters
%_________________TAWSS & OSI__________________%
TAWSS_p=[];
fprintf('\n Calculating Haemodynamical parameters...\n')
for i= 1:length(CPOINTS)%Each case
    fprintf('\nCase %s\n',CASES{i})
   % for j=1:3% LAA Dia, Sys  or both
   for j = 1
        cont=1;
        T=0.01:0.01:(SYS_DIA(j)*10^(-2));
        fprintf('\t\t%s',SYS_DIA_SD{j})  
        TAWSS_p=zeros(length(CPOINTS{i}{1}),1);
        TAWSS_x=zeros(length(CPOINTS{i}{1}),1);
        TAWSS_y=zeros(length(CPOINTS{i}{1}),1);
        TAWSS_z=zeros(length(CPOINTS{i}{1}),1);
            for t=(1+SS_SYS_DIA(1,j)):SS_SYS_DIA(2,j)
                cont=cont+1;
                TAWSS_p=[sqrt((CPOINTS{i}{t}(:,WSS)).^2)*T(cont)-(sqrt((CPOINTS{i}{t}(:,WSS)).^2)*T(cont-1))+TAWSS_p];%Caculation of TAWSS
                                
                 TAWSS_x=[(CPOINTS{i}{t}(:,WSSx))*(T(cont)-T(cont-1))+TAWSS_x]; % Computation of TAWSSV(x)
                 TAWSS_y=[(CPOINTS{i}{t}(:,WSSy))*(T(cont)-T(cont-1))+TAWSS_y]; % Computation of TAWSSV(y)
                 TAWSS_z=[(CPOINTS{i}{t}(:,WSSz))*(T(cont)-T(cont-1))+TAWSS_z]; % Computation of TAWSSV(z)
            end;
        fprintf('- T=%d\n',cont); 
        TAWSS1=sqrt(TAWSS_x.^2+TAWSS_y.^2+TAWSS_z.^2); %Computation of TAWSSV using TAWSSV(x), TAWSSV(y) & TAWSSV(z)
        TAWSS2=TAWSS_p;
        TAWSS{i}{j}(:,1)=TAWSS_p./(SYS_DIA(j)*10^(-2)); %TAWSS (The TAWSS computed before is divided by the period SYS= 0.40, DIA=0.65 & BOTH= 1.05)
        OSI{i}{j}(:,1)=(1/2)*(1-(TAWSS1./TAWSS2)); %OSI (using TAWSSV and TAWSS before dividing by T) 

    end;
end;
%_________________RRT & ECAP_________________%
 RRT=[];
 ECAP=[];
for i= 1:length(CPOINTS)%Each case
    %for j=1:3% LAA Dia and Sys, LA Dia and Sys 
    for j = 1
        RRT{i}{j}(:,1)=[(1-2.*OSI{i}{j}).* TAWSS{i}{j}].^(-1);
        ECAP{i}{j}(:,1)=(OSI{i}{j})./ (TAWSS{i}{j});
    end;
end;
fprintf('\n');
fprintf('Saving Haemodinamic Parameters (TAWSS, OSI, RRT, ECAP) in "HaemPara_POINTS"...\n');
save('HaemPara_POINTS','TAWSS','OSI','RRT','ECAP')
%% VTK creator 
%Here all the data calculated is stored into their corresponding .vtk files
%containing their structure
cont=0;
cont2=0;
NumFields=length(VARIABLES)*length(SDstr);
HAEMPARAM=load('HaemPara_POINTS.mat'); %Struct de TAWSS, OSI, RRT and ECAP
number_Cases=length(HAEMPARAM.(VARIABLES{1}));
fprintf('Writting .vtk files:\n')
for C=1:number_Cases %Each case
    fprintf('\nCase %s\n',CASES{C})
      number_pnts=length(HAEMPARAM.(VARIABLES{1}){C}{1});%.(VARIABLE{1} name) {Case}{sys_or_dia}
      file_name=[ CASES{C} '_SURFACE.vtk'];
      fid = fopen(file_name,'a','b');
      fprintf(fid,'POINT_DATA %d\n',number_pnts) ;
      fprintf(fid,'FIELD FieldData %d\n', NumFields);
      
      for V=1:length(VARIABLES) %Each variable
          fprintf('\n\t%s', VARIABLES{V});
          for SD=1:length(SDstr)
              fprintf('\n\t\t%s', SDstr{SD});
             cont2=cont2+1;
             FieldNames{cont2}=[VARIABLES{V} '_' SDstr{SD}];
             fprintf(fid,'\n%s 1 %d double\n', FieldNames{cont2},number_pnts);
             for i=1:number_pnts %Each point 
                  cont=cont+1;
                  fprintf(fid,'%.6f ', HAEMPARAM.(VARIABLES{V}){C}{SD}(i,1));
                  if cont==9 
                      cont=0;
                      fprintf(fid,'\n');
                  end;
             end;
         end;
  
     end;
     fprintf(fid,'\n');
     fclose(fid);
end;
fprintf('\n\n\n______________COMPUTATION COMPLETED______________')

end