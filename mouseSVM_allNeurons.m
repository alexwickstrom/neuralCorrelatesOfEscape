%mouseSVM_allNeurons.m
% Load the 3-trial mouse-vs-rat dataset, and classify within and across
% trials.

% Included is a script to plot the mean firing rates from each cell, for
% each behavioral class, across each trial.
cd('~/Desktop/Rat_ThreeSessions_ALLsegments/')

mapping=load('cellRegistered_20181101_122533.mat');
mapStr = mapping.cell_registered_struct;
cellMap = mapStr.cell_to_index_map;
eps = 0.1;

normX = @(X) (X-mean(X,1))./(std(X,0,1)+eps);

% edit to remove zeros

[zRows, zCols] = find(cellMap==0);
zRows = sort(unique(zRows));

sameUnitsMap = cellMap;
sameUnitsMap(zRows,:) = [];
cellMap = sameUnitsMap;


% cellMap shape: 164x3
% Each column is a session.
% If there's not a match:

% Row_x == neuron ID;
% Number appearing in the column = which cell, during gthat trial, is
% neuron_x

% If they're in the same row, they're the same neuron.


map1 = cellMap(:,1); % H13
map2 = cellMap(:,2); % H17
map3 = cellMap(:,3); % H18

nrnIds1 = map1(map1>0);
nrnIds2 = map2(map2>0);
nrnIds3= map3(map3>0);

nrnIDs = {nrnIds1, nrnIds2, nrnIds3}; % group them into a cell
% When inside a loop from 1 to 3:

trialDirs = {'H13_M46_S37','H17_M38_S10','H18_M26_S50'};

nrnDat = struct;
m = 10; % dimensionality of reduced data
y = struct;
for ii = 1:numel(trialDirs)
    %%%%%%%%%%%%%%%%%
    % open the data %
    cd(trialDirs{ii})
    thisDir = pwd;
    disp(thisDir)
    nrn = load('neuron.mat'); nrn = nrn.neuron;
    % extract and correctly index the neural data
    nrnDat(ii).data = nrn.C(nrnIDs{ii},:);
    nrnDat(ii).dir = trialDirs{ii};
    [numUnits, numBins] = size(nrnDat(ii).data);
    % perform PCA for dimRedux
    [coeff,score] = pca(nrnDat(ii).data');
    
    nrnDat(ii).features = score(:,1:m)';
    
    %%%%%%%%%%%%%%%%%%%%%%%
    % now make the labels %
    % load the behavioral transition times
    % get the indices for the start of each behavior
    
    labels = cell(size(nrnDat(ii).data,2),1);
    labels(:) = {'0'};
    BxTimes = load('Behavior_MS.mat');
    

    times.app = [];
    times.esc = [];
    times.frz = [];
    times.str = [];
    
    appTimes = BxTimes.approachFrameMS; % approach
    for jj = 1:size(appTimes,1)
        if appTimes(jj,2) <= numBins
            times.app = [times.app, appTimes(jj,1):appTimes(jj,2)];
        end
    end; labels(times.app) = {'approach'};
    
    escTimes = BxTimes.escapeFrameMS; % escape
    for jj = 1:size(escTimes,1)
        if escTimes(jj,2) <= numBins
            times.esc = [times.esc, escTimes(jj,1):escTimes(jj,2)];
        end
    end; labels(times.esc) = {'escape'};
    
    frzTimes = BxTimes.freezeFrameMS; % freeze
    for jj = 1:size(frzTimes,1)
        if frzTimes(jj,2) <= numBins
            times.frz = [times.frz, frzTimes(jj,1):frzTimes(jj,2)];
        end
    end; labels(times.frz) = {'freeze'};
    
    strTimes = BxTimes.stretchFrameMS; % stretch
    for jj = 1:size(strTimes,1)
        if strTimes(jj,2) <= numBins
            times.str = [times.str, strTimes(jj,1):strTimes(jj,2)];
        end
    end; labels(times.str) = {'stretch'};
    y(ii).labels = labels;
    cd('../')
end

% train an SVM; split the data.
kernel = 'linear';
for kk = 1:ii
    crossValAccs = [0];
    for fold = 1:10
        % split the data
        
        % choose raw data or features when selecting a field from nrnDat
        [Xtrain,Ytrain,Xval,Yval] = tv_split(nrnDat(kk).data,...
            y(kk).labels,0.8);
        % train
        
        Xtrain(4,:) = [];
        Xval(4,:) = [];
        t = templateSVM('Standardize',1,'KernelFunction',kernel);
        Mdl = fitcecoc(Xtrain',Ytrain,'Learners',t,...
            'Verbose',2);
        models(kk).model = Mdl;
        models(kk).dir = nrnDat(kk).dir;
        
        
        % % get accuracy % %
        % trainingAcc
        Yhat_train = predict(Mdl,Xtrain');
        [C_M_train, orderTrain] = confusionmat(Ytrain,Yhat_train);
        acc_train = sum(diag(C_M_train)/sum(sum(C_M_train)));
        
        
        % testAcc
        
        Yhat = predict(Mdl,Xval');
        [C_M,order] = confusionmat(Yval,Yhat);
        acc = sum(diag(C_M)/sum(sum(C_M)));
        models(kk).CM = C_M;
        models(kk).acc = acc;
        crossValAccs(fold) = acc;
        
    end
    models(kk).crossValAcc = mean(crossValAccs);
    disp(mean(crossValAccs))
    
    
end

%% Train across datasets

% Option A:
% Choose one to hold out as test;
% Pool the remaining two for training.

% Option B:
% Pool everyone's trials.
% Split the pooled trials randomly into T/V sets.

foldIDs = 1:3;
lam = 8e-2;

for fold = 1:length(foldIDs)
    
    otherTrials = foldIDs; % make a new copy of #s 1:n
    Xtest = nrnDat(fold).data;
    Ytest = y(fold).labels;
    
    [Xtest,Ytest] = removeZeros(Xtest,Ytest);
    
    otherTrials(fold) = [];
    Xtrain = [];
    Ytrain = [];
    % concatenate the remaining datasets
    for trial = otherTrials
        Xtrain =  [Xtrain, nrnDat(trial).data];
        
        Ytrain =  [Ytrain; y(trial).labels];
    end
    dim = 2;
%     normX = @(X) (X-mean(X,1))./(std(X,0,1)+2);
    % Normalize
    Xtrain = normX(Xtrain);%(Xtrain - mean(Xtrain,1)) %normalize(Xtrain,dim);
    Xtest = normX(Xtest);
   % Xtest = normalize(Xtest,dim);
    Xtrain(unitsToRemove,:) = [];
    Xtest(unitsToRemove,:) = [];
    kernel = 'linear';
    [Xtrain,Ytrain] = removeZeros(Xtrain,Ytrain);
    hyperOpts = struct('Optimizer','bayesopt','KFold',5,...
        'MaxObjectiveEvaluations',3,'Regularization','lasso');
%     t = templateSVM('Standardize',1,'KernelFunction',kernel,...
%         'IterationLimit',1e6);
%     t = templateSVM('KernelFunction', kernel, 'IterationLimit', 1e5, ...
%         'Standardize', 0, 'DeltaGradientTolerance', 1e-15, ...
%         'OutlierFraction', 0);

% bayesOpt Lambdas: 3e-7
% best lambda: 8e-1 for RidgeRegression
    t = templateLinear('lambda', lam, 'regularization', 'ridge', ...
            'Solver', 'bfgs', 'FitBias', 0, 'IterationLimit', 9000000, ...
            'GradientTolerance', 1e-16, 'BetaTolerance', 1e-15);
    Mdl = fitcecoc(Xtrain',Ytrain,'Learners',t,...
           'Verbose',0);%,...
          % 'OptimizeHyperparameters','auto');
     
    %'HyperparameterOptimizationOptions',hyperOpts,...
      %  'Verbose',2);
    
    modelsAcross(kk).model = Mdl;
    modelsAcross(kk).dir = nrnDat(fold).dir;
    
    % get accuracy
    Yhat_train = predict(Mdl,Xtrain');
    [C_M_train, orderTrain] = confusionmat(Ytrain,Yhat_train);
    acc_train = sum(diag(C_M_train)/sum(sum(C_M_train)));
    disp('Training CM + Accuracy:')
    disp(C_M_train)
    disp(acc_train)
    
    Yhat = predict(Mdl,Xtest');
    [C_M,order] = confusionmat(Ytest,Yhat);
    acc = sum(diag(C_M)/sum(sum(C_M)));
    modelsAcross(kk).CM = C_M;
    modelsAcross(kk).acc = acc;
    crossValAccs_across(fold) = acc;
    disp('-------')
    disp('Validation CM + Accuracy:')
    disp(C_M)
    disp(['Acc = ',num2str(acc)])
    
    [perClassAccs] = perClassAccuracies(C_M,order);
    keyboard
end

%% Troubleshoot the neural identities
MEANS = [];
for ii = 1:3
    YY = y(ii).labels;
    XX = nrnDat(ii).data;
    classInds(ii).app = find(contains(YY,'approach'));
    classInds(ii).esc = find(contains(YY,'escape'));
    classInds(ii).frz = find(contains(YY,'freeze'));
    classInds(ii).str = find(contains(YY,'stretch'));
    
    % split the data by behavior
    bxData(ii).app  = normX(XX(:,classInds(ii).app));
    bxData(ii).esc  = normX(XX(:,classInds(ii).esc));
    bxData(ii).frz  = normX(XX(:,classInds(ii).frz));
    bxData(ii).str  = normX(XX(:,classInds(ii).str));
    
    figure;
    
    ax1 = subplot(221); sc1 = imagesc(bxData(ii).app);
    cb1 = colorbar; cmap1 = caxis; title('Approach')
    ax2 = subplot(222); sc2 = imagesc(bxData(ii).esc);
    cb2 = colorbar; cmap2 = caxis; title('Escape')
    ax3 = subplot(223); sc3 = imagesc(bxData(ii).frz);
    cb3 = colorbar; cmap3 = caxis; title('Freeze')
    ax4 = subplot(224); sc4 = imagesc(bxData(ii).str);
    cb4 = colorbar; cmap4 = caxis; title('Stretch')
    
    figure;
    
    allMeans = [mean(bxData(ii).app,2), ...
        mean(bxData(ii).esc,2),...
        mean(bxData(ii).frz,2),...
        mean(bxData(ii).str,2)];
    plot(mean(bxData(ii).app,2),'LineWidth',2)
    hold on
    plot(mean(bxData(ii).esc,2),'LineWidth',2)
    plot(mean(bxData(ii).frz,2),'LineWidth',2)
    plot(mean(bxData(ii).str,2),'LineWidth',2)
    plot(mean(allMeans,2),'k--','LineWidth',2)
    grid on
    
    legend('app','esc','frz','str','AVG')
    
    MEANS = [MEANS, mean(allMeans,2)];
    
    % For this trial:
    % Get the average firing rate from each neuron, across each behavior.
    
    % MeanFiringRates = 
    
    eachDay(ii).app = mean(normX(bxData(ii).app),2);
    eachDay(ii).esc = mean(normX(bxData(ii).esc),2);
    eachDay(ii).frz = mean(normX(bxData(ii).frz),2);
    eachDay(ii).str = mean(normX(bxData(ii).str),2);
    
    
end

for ii = 1:3
    eachDay(ii).app = mean(normX(bxData(ii).app),2);
    eachDay(ii).esc = mean(normX(bxData(ii).esc),2);
    eachDay(ii).frz = mean(normX(bxData(ii).frz),2);
    eachDay(ii).str = mean(normX(bxData(ii).str),2);
end

    % get correlations: 
    % For a given cell: Get the correlations
    % 
    % For each cell:
        % Make a vector (4 x 1) for each day.
        % Then, take correlations between each vector
        % Minimum will be the lowest R value.
        % Then exclude units with R < thresh. 
numUnits = size(eachDay(ii).app,1);
corrMat = zeros(numUnits,4,3);
corrs = zeros(3,numUnits);

for ii = 1:3
    x = eachDay(ii);
    corrMat(:,:,ii) = [x.app,x.esc,x.frz,x.str];
end

for ii = 1:numUnits
    C = corr(squeeze(corrMat(ii,:,:)));
    corrs(:,ii) = [C(1,2),C(1,3),C(2,3)];
end
minCorr = min(abs(corrs));

unitsToRemove = find(minCorr<0.9*mean(minCorr));
keyboard
    
% Now make a vector of length 4, that's the mean

% Each unit should have an aveerage 

% Separate everything first by behavior type
% SEparate the units across days
% See which ones are correlated most by plotting Day1 as X axis, Day2 as Y
% axis.

%%
for ii = 1:3
    eachDay(ii).app = mean(normX(bxData(ii).app),2);
    eachDay(ii).esc = mean(normX(bxData(ii).esc),2);
    eachDay(ii).frz = mean(normX(bxData(ii).frz),2);
    eachDay(ii).str = mean(normX(bxData(ii).str),2);
end
figure; 
trls = 1:3;
plotX = [1 2; 1 3; 2 3];
for sp = 1:3
    d1 = plotX(sp,1); d2 = plotX(sp,2);

    subplot(2,2,sp)
    plot(eachDay(d1).app,eachDay(d2).app,'g.')
    hold on
    p1 =plot(eachDay(d1).esc,eachDay(d2).esc,'r.','MarkerSize',10);
    p1.Color = [p1.Color,0.6]
    plot(eachDay(d1).frz,eachDay(d2).frz,'.','Color',[1 0.6 0])%'o.')
    plot(eachDay(d1).str,eachDay(d2).str,'b.')
    plot([-1:0.1:4],[-1:0.1:4],'k--')
    title(['Day ',num2str(d1), ' vs. Day ',num2str(d2)])
    legend('App','esc','frz','str')
    grid
    
end
    




%%
figure; 
mp = plot(MEANS,'LineWidth',2);
alph = 0.6;
for ii=1:3
    mp(ii).Color = [mp(ii).Color, alph];
end
legend('Trial1','Trial2','Trial3')
grid

% future things to try:
%{

 1) Try to train/predict on bins BEFORE Bx onset
 2) Try a CNN
 3) Pool across all trials, and split into train/val. sets
 4) Try to discriminate close vs. far escape
    a) Within
    b) Across trials
 5) Train/test only on the first N bins from each of the Bx epochs
 6) Shock dataset too.
%}

%%
function [Xtrain,Ytrain,Xval,Yval] = tv_split(x,y,ratio)

[x,y] = removeZeros(x,y);

trainInds = sort(randperm(numel(y),floor(ratio*numel(y))));

Ytrain = y(trainInds); % choose some for training set.
y(trainInds) = [];     % then erase them.
Yval = y;           % The remainder = validation set.

try
    Xtrain = x(:,trainInds);
catch
    keyboard
end
x(:,trainInds) = [];
Xval = x;
end

function [x,y] = removeZeros(x,y)
zeroClass = find(contains(y,'0'));
try
    y(zeroClass) = [];
    x(:, zeroClass) = [];
catch
    keyboard
end
end

function [perClassAccs] = perClassAccuracies(CM,order)
%CM(I,J) = CM(Y_true, Y_hat)
    % represents the count of instances whose
    % TRUE labels are group I
    %  and whose predicted group labels are group J.

perClassAccs = {};
classNames = {'approach','escape','freeze','stretch'};
for whichClass = 1:4
    thisClassName = classNames{whichClass};
    classIndx = find(contains(order,thisClassName));
    numTruePos = CM(classIndx,classIndx);
    BxRow = CM(classIndx,:);
    BxCol = CM(:,classIndx);
    perClassAccs{whichClass} = {thisClassName,squeeze(numTruePos/sum(BxRow))};
    disp(perClassAccs{whichClass})
end

end
