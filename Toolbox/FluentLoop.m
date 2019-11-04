%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                                                                   %%%%
%%%%                 Loop for automatic Fluent simulations             %%%%
%%%%                                                                   %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all

%% Establishing Fluent Matlab conection

orb=initialize_orb();
load_ansys_aas();

iCoFluentUnit=actfluentserver(orb,'aaS_FluentId.txt');
iFluentTuiInterpreter=iCoFluentUnit.getSchemeControllerInstance();

%% Directory initialization

%%% File names - For ease of automatization all files shared the same name and are identified by their simulation ID number

Name='LAA'; %General name of all files
Nastran=([Name,'_']); %General Name of the .bdf files

%%% Trace files contain all the code to automatically run all the paraview postprocessing automatically through pvpython 
nTrace='Trace.py'; % Name of original trace file
sTrace='Trace2.py'; % Name of the trace that iteratively changes

%%% Main directory where all files are loaded and saved

GDir='C:\Simulations\';
TDir='''C:\\Simulations\\';

%%% Directory for all input and output files inside the main directory

InDir=([GDir,'BDF']); % directory of the .bdf files
OutDir=([GDir,'EncaseFiles']); % directory of the .encas files
OutDirCas=([GDir,'CaseFiles']); % directory of the .cas files
OutDirData=([GDir,'DataFiles']); % directory of the .data files

%%% Trace directories - Directory for the .stl load in pvpython
encDir=(['EncasDir=','',TDir,'EncaseFiles\\']); % .encas directory for the .stl load in the trace

%% Parameters of the simulation loop

simu=150; % Simulations to be carried out
s=length(simu); % Total amount of simulations
fail=zeros(1,s+1); % Vector to save all the simulations that have failed

%%% Parameter initialization

%%% Set of variables stored on .encas
Var='density density-all velocity-magnitude x-velocity  y-velocity z-velocity absolute-pressure total-pressure rel-total-pressure vorticity x-vorticity y-vorticity z-vorticity mass-imbalance wall-shear x-wall-shear y-wall-shear z-wall-shear';

%%% Dynamic mesh smoothing parameters
diffcoeff=0; % Diffusion coefficient (Try 1.5 1.8 if possible)

%%% Dynamic mesh remeshing parameters
rint=6; % size-remeshing-interval: if it takes long change it to 6 or 7 
skew=0.9; % maximum cell skewnesh

%%% Number of steps and iterations
st=105;
it=200;

%%% Dynamic mesh zone parameters (heavily depends on your mesh)
cheight=0.001; % Cell height
Min=3.9306e-05; % minimum length scale of the mesh
Max=0.0105; % maximum length scale of the mesh

%%% Read Trace file (Save it as a character array)

fid = fopen(nTrace,'r');
p = 1;
tline = fgetl(fid);
A{p} = tline;
while ischar(tline)
    p = p+1;
    tline = fgetl(fid);
    A{p} = tline;
end
fclose(fid);

%%% Function to check mesh integrity
isaninteger = @(x)isfinite(x) & x==floor(x);

%% Simulation loop

for p=1:s
    
    %%% Initialize .bdf names
    i=simu(p);
    NastranL=([Nastran,num2str(i)]); % Name of this iteration simulation
    NastranN=([NastranL,'.bdf']); % Name of this iteration .bdf
    
    %%% Initialize input and ouput directories for the specific simulation
    
    InDirL=([InDir,'\',NastranN]); % Path to .bdf file to load for iteration
    OutDirL=([OutDir,'\',NastranL]); % Path to .encas file in which to save in this iteration
    OutDirCasL=([OutDirCas,'\',NastranL]); % Path to .cas file in which to save in this iteration
    OutDirDataL=([OutDirData,'\',NastranL]); % Path to .data file in which to save in this iteration
    
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%                                                                   %%%%
    %%%%                        Fluent TUI commands                        %%%%
    %%%%                                                                   %%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %% General parameters
    
    %%% Load NASTRAN check mesh - Try and catch was necessary due to unknown errors arising
    try
        iFluentTuiInterpreter.doMenuCommandToString(['file/import/nastran/bulkdata/ ',InDirL,' o'])
    catch 
        iFluentTuiInterpreter.doMenuCommandToString(['file/import/nastran/bulkdata/ ',InDirL,' o'])
    end
    
    %%% Time transient
    iFluentTuiInterpreter.doMenuCommandToString('define/models/unsteady-1st-order? y')
    
    %%% Scale (Removed this step because my mesh was altered by other software, might be necessary depending on the mesh)
    %iFluentTuiInterpreter.doMenuCommandToString('mesh/scale/ 0.001 0.001 0.001')
    
    %%% Define blood properties
    iFluentTuiInterpreter.doMenuCommandToString('/define/materials/change-create air blood yes constant 1060 n n yes constant 0.0035 n n n yes')
    
    %%% Define zone type
    iFluentTuiInterpreter.doMenuCommandToString('/define/boundary-conditions/zone-type 1 fluid')
    
    
    %% Mesh check
    %%% If check fails try to repair once, if it fails a second time continue to next simulation
    mesh=iFluentTuiInterpreter.doMenuCommandToString('mesh/check');
    
    if isaninteger(strfind(mesh,'Mesh check failed'))==1
        
        disp(['Bad mesh: ',num2str(i),' has failed - Repairing mesh...'])
        fail(i+1)=1;
        
        %%% Repair mesh
        iFluentTuiInterpreter.doMenuCommandToString('mesh/repair-improve/allow-repair-at-boundaries yes')
        iFluentTuiInterpreter.doMenuCommandToString('mesh/repair-improve/repair')
        
        %%% Repeat the mesh check
        mesh=iFluentTuiInterpreter.doMenuCommandToString('mesh/check');
        
        %%% If it fails once again continue to next case
        if isaninteger(strfind(mesh,'Mesh check failed'))==1
            
            disp(['Bad mesh after repair: ',num2str(i),' has failed'])
            fail(i+1)=3; % 3==Failed after mesh repair
            
            continue
            
        end
    end

    %% Boundary conditions
    
    %%% Load BCs and mitral motion 
    iFluentTuiInterpreter.doMenuCommandToString('file/read-transient-table/ Functions/function_mitral_def.prof')
    iFluentTuiInterpreter.doMenuCommandToString('define/user-defined/interpreted Functions/bc.c')
    
    %%% Separate: Separate exterior into several zones according to angle (in our case 60 is optimum)
    iFluentTuiInterpreter.doMenuCommandToString('define/boundary-conditions/modify-zones/sep-face-zone-angle 3 60 yes')
    
    %%% Merging: This might change depending on the geometry find number of surfaces
    summ=iFluentTuiInterpreter.doMenuCommandToString('report/summary no');
    
    b=strfind(summ,"default_exterior-1:0");
    c=strfind(summ,"default_exterior-1       3");
    f=length(b(b<c));
    merge='define/boundary-conditions/modify-zones/merge 3';
    
    if(f>5)
        for p=10:(4+f)
            merge=[merge ' ' num2str(p)];
        end
        
        iFluentTuiInterpreter.doMenuCommandToString(merge)
    end
    
    
    %%% Find the mitral valve (biggest zone compared to PVs)
    %%% Info of the size of each generated zone
    surf=char(iFluentTuiInterpreter.doMenuCommandToString('mesh/mesh-info'));
    
    id=char(extractBetween(surf,'zone  ','.'));
    v=char(extractBetween(surf,'.','triangular'));
    f=zeros(1,5);
    
    for k=3:7
        u=strsplit(v(k,:));
        f(k-2)=str2double(string(u(2)));
    end
    
    %%% Find the biggest zone
    mv=str2double(id(find(f==max(f))+3));
    pv=5:9;
    pv(mv-4)=[];
    
    %%% Set velocity inlet profile in PVs
    iFluentTuiInterpreter.doMenuCommandToString(['define/boundary-conditions/zone-type ',num2str(pv(1)),' velocity-inlet'])
    iFluentTuiInterpreter.doMenuCommandToString(['define/boundary-conditions/zone-type ',num2str(pv(2)),' velocity-inlet'])
    iFluentTuiInterpreter.doMenuCommandToString(['define/boundary-conditions/zone-type ',num2str(pv(3)),' velocity-inlet'])
    iFluentTuiInterpreter.doMenuCommandToString(['define/boundary-conditions/zone-type ',num2str(pv(4)),' velocity-inlet'])
    
    % , is equivalent to enter key on the Fluent console
    iFluentTuiInterpreter.doMenuCommandToString('define/boundary-conditions/set/velocity-inlet , , , , , vmag yes yes , pvs_profile q')
    
    %%% Name all the zones
    iFluentTuiInterpreter.doMenuCommandToString('define/boundary-conditions/zone-name 3 la')
    iFluentTuiInterpreter.doMenuCommandToString(['define/boundary-conditions/zone-name ',num2str(mv),' mv'])
    
    iFluentTuiInterpreter.doMenuCommandToString(['define/boundary-conditions/zone-name ',num2str(pv(1)),' pv1'])
    iFluentTuiInterpreter.doMenuCommandToString(['define/boundary-conditions/zone-name ',num2str(pv(2)),' pv2'])
    iFluentTuiInterpreter.doMenuCommandToString(['define/boundary-conditions/zone-name ',num2str(pv(3)),' pv3'])
    iFluentTuiInterpreter.doMenuCommandToString(['define/boundary-conditions/zone-name ',num2str(pv(4)),' pv4'])
    
    %% Dynamic mesh
    
    iFluentTuiInterpreter.doMenuCommandToString('define/dynamic-mesh/dynamic mesh? y n n n n')
    iFluentTuiInterpreter.doMenuCommandToString('define/dynamic-mesh/controls/remeshing? y')
    iFluentTuiInterpreter.doMenuCommandToString('define/dynamic-mesh/controls/smoothing? y')
    
    %%% Remeshing parameters
    
    % min=str2double(iFluentTuiInterpreter.doMenuCommandToString("(rpgetvar 'dynamesh/remesh/min-length-scale)")); %min-length-scale
    % max=str2double(iFluentTuiInterpreter.doMenuCommandToString("(rpgetvar 'dynamesh/remesh/max-length-scale)"))+0.0002; %max-length-scale
    
    iFluentTuiInterpreter.doMenuCommandToString('define/dynamic-mesh/controls/remeshing-parameters/remeshing methods y n n n n')
    iFluentTuiInterpreter.doMenuCommandToString(['define/dynamic-mesh/controls/remeshing-parameters/length-min ',num2str(Min)])
    iFluentTuiInterpreter.doMenuCommandToString(['define/dynamic-mesh/controls/remeshing-parameters/length-max ',num2str(Max)])
    iFluentTuiInterpreter.doMenuCommandToString(['define/dynamic-mesh/controls/remeshing-parameters/size-remesh-interval ',num2str(rint)])
    iFluentTuiInterpreter.doMenuCommandToString(['define/dynamic-mesh/controls/remeshing-parameters/cell-skew-max ',num2str(skew)])
    
    %%% Smoothing parameters
    
    iFluentTuiInterpreter.doMenuCommandToString('define/dynamic-mesh/controls/smoothing-parameters/smoothing method diffusion')
    iFluentTuiInterpreter.doMenuCommandToString('define/dynamic-mesh/controls/smoothing-parameters/diffusion-coeff-function boundary-distance')
    iFluentTuiInterpreter.doMenuCommandToString(['define/dynamic-mesh/controls/smoothing-parameters/diffusion-coeff-parameter ',num2str(diffcoeff)])
    
    %%% Zones
    
    %Deforming
    iFluentTuiInterpreter.doMenuCommandToString(['define/dynamic-mesh/zones/create 2 deforming faceted n y n ',num2str(Min),' ',num2str(Max),' ',num2str(skew),' n'])
    iFluentTuiInterpreter.doMenuCommandToString(['define/dynamic-mesh/zones/create la deforming faceted n y n ',num2str(Min),' ',num2str(Max),' ',num2str(skew),' n'])
    
    %Rigid body
    iFluentTuiInterpreter.doMenuCommandToString(['define/dynamic-mesh/zones/create mv rigid-body mv_movement_y n n , , , , , , , 1 constant ',num2str(cheight),' n n'])
    
    %Stationary
    cheight=0.001;
    iFluentTuiInterpreter.doMenuCommandToString(['define/dynamic-mesh/zones/create pv1 stationary 1 constant ',num2str(cheight),' n'])
    iFluentTuiInterpreter.doMenuCommandToString(['define/dynamic-mesh/zones/create pv2 stationary 1 constant ',num2str(cheight),' n'])
    iFluentTuiInterpreter.doMenuCommandToString(['define/dynamic-mesh/zones/create pv3 stationary 1 constant ',num2str(cheight),' n'])
    iFluentTuiInterpreter.doMenuCommandToString(['define/dynamic-mesh/zones/create pv4 stationary 1 constant ',num2str(cheight),' n'])
    
    %% Data saving and initialization
    
    %%% Gradient
    iFluentTuiInterpreter.doMenuCommandToString('solve/set/gradient-scheme Green-Gauss Node Based y')
    
    %%% Convergence criterion
    iFluentTuiInterpreter.doMenuCommandToString('solve/monitors/residual/convergence 0.005 , , ,')
    
    %%% Data saving
    iFluentTuiInterpreter.doMenuCommandToString(['file/transient-export/ensight-gold-transient ',([OutDirL,'\',NastranL]),' 2 , 1 , ',Var,' q n n , 1 y'])
    
    %% Execute commands
    
    %%% The commands to setup the mv pressure and opening always return
    %%% and error (unknown reason) so the try and catch is necessary to
    %%% continue the automatic loop 
    try
        
        iFluentTuiInterpreter.doMenuCommandToString('file/read-macro m.scm')
        iFluentTuiInterpreter.doMenuCommandToString('file/execute-macro- mv')
        
    catch
        
        disp('Execution will continue.');
        
    end
    
    %iFluentTuiInterpreter.doMenuCommandToString(['solve/execute-commands/add-edit- , 41 "time-step" "define/boundary-conditions/zone-type wall) ',num2str(mv),' pressure-outlet"'])
    %iFluentTuiInterpreter.doMenuCommandToString('/solve/execute-commands/add-edit- b 41 "time-step" "define/boundary-conditions/pressure-outlet mv yes no 1067 no yes yes no no no"')
    
    %iFluentTuiInterpreter.doMenuCommandToString('file/read-macro macros.scm')
    %iFluentTuiInterpreter.doMenuCommandToString('file/execute-macro- mitralvalve')
    
    %solve/execute-commands/add-edit command-1 41 "time-step" "define/boundary-conditions/zone-type wall) 5 pressure-outlet"
    %solve/execute-commands/add-edit command-2 41 "time-step" "define/boundary-conditions/pressure-outlet mv yes no 1067 no yes yes no no no"')
    
    %% Calculation
    
    %%% Time-step
    iFluentTuiInterpreter.doMenuCommandToString('solve/set/time-step 0.01')
    
    %%% Number of time-steps
    iFluentTuiInterpreter.doMenuCommandToString(['solve/set/number-of-time-steps ',num2str(st),''])
    
    %%% Iterations
    iFluentTuiInterpreter.doMenuCommandToString(['solve/set/max-iterations-per-time-step ',num2str(it),''])
    
    %%% Initialization
    iFluentTuiInterpreter.doMenuCommandToString('solve/initialize/compute-defaults/all-zones')
    iFluentTuiInterpreter.doMenuCommandToString('solve/initialize/initialize-flow')
    
    %%% Try to perform the simulation if not continue to next instance        
    
    try
        
        %% Calculation
        iFluentTuiInterpreter.doMenuCommandToString(['solve/dual-time-iterate ',num2str(st),' ',num2str(it),''])
        
        %% Write .case and .data
        
        iFluentTuiInterpreter.doMenuCommandToString(['file/write-case ',OutDirCasL])
        iFluentTuiInterpreter.doMenuCommandToString(['file/write-data ',OutDirDataL])
        
        %% Paraview Postprocessing
        %% Change the Trace file
        
        %%% Create directory for ECAP results
        mkdir ([OutDirL,'\ECAP\']);
        
        %%% Lines to change in the trace file
        
        enc=12; %Line were the .encas is loaded
        csv=51; %Line were the .csv is saved
        stlS=72; %Line were the .stl is saved
        stlR=78; %Line were the .stl is loaded
        vtk=103; %Line were the .vtk is saved
        
        %%% Change .encas dir
        A{enc}=([encDir,NastranL,'\\',NastranL,'.encas''']);
        
        %%% Change .csv and .stl dir
        Res=([OutDirL,'\ECAP\',Name]);
        A{csv}=(['SaveData(''',Res,'.csv'', proxy=lAA_1encas, WriteTimeSteps=1)']);
        A{stlS}= (['SaveData(''',Res,'.stl'', proxy=extractSurface1, FileType=''Ascii'')']);
        
        %%% Change .stl dir to open
        A{stlR}=([encDir,NastranL,'\\ECAP\\',Name,'1.stl''']);
        
        %%% Change .vtk dir to save
        A{vtk}=(['SaveData(''',Res,'_SURFACE.vtk'', proxy=lAA1stl, FileType=''Ascii'')']);
        
        %%% Save new Trace file
        fid = fopen(sTrace,'w');
        for k = 1:(numel(A)-1)
            fprintf(fid,'%s\n', A{k});
        end
        fclose(fid);
        
        %% Call pvpython
        
        system(['pvpython ',sTrace,' &']) %% Run the trace file through pvpython
        
    catch
        disp(['Simulation number: ',num2str(i),' has failed'])
        fail(i+1)=2; %%% 2== Failed at simulation step
    end
    
end

disp('The simulation loop has been completed! YAY!')

