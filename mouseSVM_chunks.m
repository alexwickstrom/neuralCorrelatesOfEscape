% Friday November 2

% "It's always hard to pantomime things that are 12-dimensional"

%{
TODO:
Make sure that TV function is doing the right thing.
SHould split the data into approx. equal amounts within each class.
%}
cd('~/Desktop/newMouseDat') % directory where the data is stored
mapping=load('cellRegistered_20181031_213822.mat');
mapStr = mapping.cell_registered_struct;
cellMap = mapStr.cell_to_index_map;
eps = 0.1;

normX = @(X) (X-mean(X,2))./(std(X,0,2)+eps);
% softNorm = @(X) (X./max(X,[],2)

% edit to remove zeros

[zRows, zCols] = find(cellMap==0);
zRows = sort(unique(zRows));

sameUnitsMap = cellMap;
sameUnitsMap(zRows,:) = [];
cellMap = sameUnitsMap;

map1 = cellMap(:,1); % H13
map2 = cellMap(:,2); % H17
map3 = cellMap(:,3); % H18

nrnIds1 = map1(map1>0);
nrnIds2 = map2(map2>0);
nrnIds3 = map3(map3>0);

nrnIDs = {nrnIds1, nrnIds2, nrnIds3}; % group them into a cell
% When inside a loop from 1 to 3:

trialDirs = {'H13_M46_S37','H17_M38_S10','H18_M26_S50'}; % name of the folders

nrnDat = struct;
m = 10; % dimensionality of reduced data
y = struct;
numTrials = numel(trialDirs);
for ii = 1:numTrials
    %%%%%%%%%%%%%%%%%
    % open the data %
    cd(trialDirs{ii})
    thisDir = pwd;
    disp(thisDir)
    nrn = load('neuron.mat'); nrn = nrn.neuron;
    % extract and correctly index the neural data
    nrnDat(ii).data = normX(nrn.C(nrnIDs{ii},:));
    
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
    
    % save the in- and out-times from each Bx epoch
    allTimes(ii).app = appTimes;
    allTimes(ii).esc = escTimes;
    allTimes(ii).frz = frzTimes;
    allTimes(ii).str = strTimes;
end

% reshape the dataset into chunks

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
chunkSize = 5;


[Xchunks,Ylabels] = chunkData(nrnDat,allTimes,y,chunkSize);

%%
close all
makeAveragePlots(nrnDat,allTimes,unitsToRemove)


%% Train across datasets

% Option A:
% Choose one to hold out as test;
% Pool the remaining two for training.

% Option B:
% Pool everyone's trials.
% Split the pooled trials randomly into T/V sets.
clc
foldIDs = [2,3];
shouldRemoveNeuron = 1;
bestAcc = 0;
trainValAccs = [];
% higher than 5e-3
lam = 0.04; % 0.107
threeAccTotal = [];
iters = 1:3:300;
for numIters = iters % get TV accuracies by training once; then twice; etc.
    threeAccs = [];
    for fold = 1:length(foldIDs)
        
        otherTrials = foldIDs; % make a new copy of #s 1:n
        
        testID = otherTrials(fold);
        otherTrials(fold) = [];
        Xtest = Xchunks(testID).data;
        Ytest = Ylabels(testID).y;
        
        % [Xtest,Ytest] = removeZeros(Xtest,Ytest);

        Xtrain = [];
        Ytrain = [];
        % concatenate the remaining datasets
        for trial = otherTrials
            
            Xtrain =  cat(3,Xtrain,Xchunks(trial).data);
            
            Ytrain =  [Ytrain; Ylabels(trial).y];
        end

        dim = 2;
       
        %
        % Normalize
        %  Xtrain = normX(Xtrain);%(Xtrain - mean(Xtrain,1)) %normalize(Xtrain,dim);
        %  Xtest = normalize(Xtest,dim);
        % remove one noisy cell
        if shouldRemoveNeuron
            try
                Xtrain(unitsToRemove,:,:) = [];
                Xtest(unitsToRemove,:,:) = [];
            catch
                Xtrain(4,:) = [];
                Xtest(4,:) = [];
            end
        end

        % erase freeze and stretch
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        frzIndsTr = contains(Ytrain,'freeze');
        frzIndsTst = contains(Ytest,'freeze');
        
        strIndsTr = contains(Ytrain,'stretch');
        strIndsTst = contains(Ytest,'stretch');
        
        trainIndsToDelete = frzIndsTr + strIndsTr;
        testInds = frzIndsTst + strIndsTst;
        
        
        Xtrain(:,:,find(trainIndsToDelete)) = [];
        Ytrain(find(trainIndsToDelete)) = [];
  
        Xtest(:,:,find(testInds)) = [];
        Ytest(find(testInds)) = [];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % make sure same # in each class
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        appInds = find(contains(Ytrain,'approach'));
        numApp = numel(appInds);
        escInds = find(contains(Ytrain,'escape'));
        numEsc = numel(escInds);
        
        [minClass,indx] = min([numApp,numEsc]);
        
        try
        if indx == 1 % delete some from Escape
            delInds = escInds(minClass+1:end);% = [];
        else
            delInds = appInds(minClass+1:end);
        end
        catch
            keyboard
        end
        
        try
        Xtrain(:,:,delInds) = [];
        catch
            keyboard
        end
        try
        Ytrain(delInds) = [];
        catch
            keyboard
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        appInds = find(contains(Ytrain,'approach'));
        numApp = numel(appInds);
        escInds = find(contains(Ytrain,'escape'));
        numEsc = numel(escInds);
        
        %if numApp ~= numEsc
           %. disp('Something''s wrong with the way you erased data...')
            %    keyboard
        %end
        
        %%% Add random noise to training data
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
       Xtrain2 = cat(3,Xtrain,Xtrain+0.01*(1+randn(size(Xtrain))));
       Ytrain2 = [Ytrain;Ytrain];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        Xtrain = reshape(Xtrain,size(Xtrain,3),size(Xtrain,2)*size(Xtrain,1));
        Xtest = reshape(Xtest,size(Xtest,3),size(Xtest,2)*size(Xtest,1));
        
        kernel = 'gaussian';
        % [Xtrain,Ytrain] = removeZeros(Xtrain,Ytrain);
        hyperOpts = struct('Optimizer','bayesopt','KFold',5,...
            'MaxObjectiveEvaluations',3,'Regularization','lasso');
        %     t = templateSVM('Standardize',1,'KernelFunction',kernel,...
        %         'IterationLimit',1e6);
        s = templateSVM('KernelFunction', kernel, 'IterationLimit', 35, ...
            'Standardize', 1, 'DeltaGradientTolerance', 1e-21, ...
            'OutlierFraction', 0.5);
        t = templateLinear('lambda', lam, 'regularization', 'ridge', ...
            'Solver', 'bfgs', 'FitBias', 1, 'IterationLimit', numIters, ...
            'GradientTolerance', 1e-15, 'BetaTolerance', 1e-15);
        
        if size(Xtrain,1) ~= size(Ytrain,1)
            disp(size(Xtrain))
            disp(size(Ytrain))
            keyboard
        end
        Mdl = fitcecoc(Xtrain,Ytrain,'Learners',t,...
            'Verbose',0);
        %         'OptimizeHyperparameters','auto',...
        
        
        %'HyperparameterOptimizationOptions',hyperOpts,...
        %  'Verbose',2);
        
        modelsAcross(fold).model = Mdl;
        modelsAcross(fold).dir = nrnDat(fold).dir;
        
        % get accuracy
        Yhat_train = predict(Mdl,Xtrain);
        [C_M_train, orderTrain] = confusionmat(Ytrain,Yhat_train);
        acc_train = sum(diag(C_M_train)/sum(sum(C_M_train)));
        disp(C_M_train)
        disp(acc_train)
        
        Yhat = predict(Mdl,Xtest);
        [C_M,order] = confusionmat(Ytest,Yhat);
        acc = sum(diag(C_M)/sum(sum(C_M)));
        if acc > bestAcc
            bestAcc = acc;
            bestLam = lam;
        end
        modelsAcross(fold).CM = C_M;
        modelsAcross(fold).acc = acc;
        crossValAccs(fold) = acc;
        
        
        disp(C_M)
        disp(acc)
        threeAccs = [threeAccs; acc];
        perClassAccs = perClassAccuracies(C_M,order);
        
        
    end
    trainValAccs = [trainValAccs; acc_train, acc];
    threeAccTotal = [threeAccTotal, threeAccs];
end

figure; plot(iters,trainValAccs,'LineWidth',2)
legend('Train','Val')
xlabel('Iteration')
title(num2str(lam))
grid on
%% Troubleshoot the neural identities
MEANS = [];
for ii = 1:3
    YY = y(ii).labels;
    XX = nrnDat(ii).data;
    classInds(ii).app = find(contains(YY,'approach'));
    classInds(ii).esc = find(contains(YY,'escape'));
    classInds(ii).frz = find(contains(YY,'freeze'));
    classInds(ii).str = find(contains(YY,'stretch'));
    
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
    
    %     figure;
    %
    % allMeans = [mean(bxData(ii).app,2), ...
    %         mean(bxData(ii).esc,2),...
    %         mean(bxData(ii).frz,2),...
    %         mean(bxData(ii).str,2)];
    %     plot(mean(bxData(ii).app,2),'LineWidth',2)
    %     hold on
    %     plot(mean(bxData(ii).esc,2),'LineWidth',2)
    %     plot(mean(bxData(ii).frz,2),'LineWidth',2)
    %     plot(mean(bxData(ii).str,2),'LineWidth',2)
    %     plot(mean(allMeans,2),'k--','LineWidth',2)
    %     grid on
    %
    %     legend('app','esc','frz','str','AVG')
    
    %    MEANS = [MEANS, mean(allMeans,2)];
end
close all
%%

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











figure;
trls = 1:3;
plotX = [1 2; 1 3; 2 3];
for sp = 1:3
    d1 = plotX(sp,1); d2 = plotX(sp,2);
    
    subplot(2,2,sp)
    
    app1 = eachDay(d1).app;
    app2 = eachDay(d2).app;
    
    
    appPoints = [app1, app2]';
    for ii = 1:size(app1,1)
        appDists(ii) = pdist([app1(ii), app2(ii);app1(ii),app1(ii)],'Euclidean');
    end
    APP(sp).dists = appDists;
    
    
    p1 = plot(app1,app2,'g.','MarkerSize',10);
    p1.Color = [p1.Color,0.6];
    
    hold on
    
    esc1 = eachDay(d1).esc;
    esc2 = eachDay(d2).esc;
    for ii = 1:size(esc1,1)
        escDists(ii) = pdist([esc1(ii), esc2(ii);esc1(ii),esc1(ii)],'Euclidean');
    end
    ESC(sp).dists = escDists;
    p1 =plot(esc1,esc2,'r.','MarkerSize',10);
    p1.Color = [p1.Color,0.6];
    
    frz1 = eachDay(d1).frz;
    frz2 = eachDay(d2).frz;
    for ii = 1:size(frz1,1)
        frzDists(ii) = pdist([frz1(ii), frz2(ii);frz1(ii),frz1(ii)],'Euclidean');
    end
    FRZ(sp).dists = frzDists;
    p1 = plot(frz1,frz2,'.','Color',[1 0.6 0],'MarkerSize',10);
    p1.Color = [p1.Color,0.6];
    
    str1 = eachDay(d1).str;
    str2 = eachDay(d2).str;
    for ii = 1:size(str1,1)
        strDists(ii) = pdist([str1(ii), str2(ii);str1(ii),str1(ii)],'Euclidean');
    end
    STR(sp).dists = strDists;
    p1 = plot(str1,str2,'b.','MarkerSize',10);
    p1.Color = [p1.Color,0.6];
    lP = plot([-1:0.1:4],[-1:0.1:4],'k--','MarkerSize',10);
    lP.Color = [p1.Color,0.6];
    title(['Day ',num2str(d1), ' vs. Day ',num2str(d2)])
    xlabel(['Day ',num2str(d1)])
    ylabel(['Day ',num2str(d2)])
    
    legend('App','esc','frz','str')
    grid
    
    
    
end
suptitle('Firing Rates')
%%
figure
for SP = 1:4
    subplot(2,2,SP)
    lw = 1.2;
    if SP == 1
        for jj = 1:3
            plot(APP(jj).dists,'LineWidth',lw); hold on
        end
        title('Approach')
    elseif SP == 2
        
        for jj = 1:3
            plot(ESC(jj).dists,'LineWidth',lw); hold on
        end
        title('Escape')
    elseif SP == 3
        for jj = 1:3
            plot(FRZ(jj).dists,'LineWidth',lw); hold on
        end
        title('Freeze')
    else
        for jj = 1:3
            plot(STR(jj).dists,'LineWidth',lw); hold on
        end
        title('Stretch')
    end
    legend('Day1 vs. Day2','Day1 vs. Day3','Day2 vs. Day3')
    grid on
    xlabel('Unit')
    ylabel('Residual')
end
suptitle('Residuals')
%%
% get all of the distances into a matrix
APPdists = squeeze(cell2mat(struct2cell(APP)))';
ESCdists = squeeze(cell2mat(struct2cell(ESC)))';
FRZdists = squeeze(cell2mat(struct2cell(FRZ)))';
STRdists = squeeze(cell2mat(struct2cell(STR)))';

figure;
plot(max(APPdists))
hold on
plot(max(ESCdists))
plot(max(FRZdists))
plot(max(STRdists))

tooBig = find(max(ESCdists)>1);
tooBig = [tooBig, find(max(APPdists)>1)];
tooBig = [tooBig, find(max(FRZdists)>1)];
tooBig = [tooBig, find(max(STRdists)>1)];
unitsToRemove = unique(tooBig);

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
% TV Split
function [Xtrain,Ytrain,Xval,Yval] = tv_split(x,y,ratio)
%%%%%%
% Make sure to have proportional class representation in training and
% validation sets

try
    
    trainInds = sort(randperm(numel(y),floor(ratio*numel(y))));
    
catch
    keyboard
end

Ytrain = y(trainInds); % choose some for training set.
y(trainInds) = [];     % then erase them.
Yval = y;           % The remainder = validation set.

try
    Xtrain = x(:,:,trainInds);
    
catch
    keyboard
end
x(:,:,trainInds) = [];
Xval = x;

Xtrain = permute(Xtrain,[3 2 1]);
Xval = permute(Xval,[3 2 1]);

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

function [xChunks,yChunks] = chunkData(nrnDat,allTimes,y,chunkSize)


binAdds = chunkSize-1;

for trial = 1:length(nrnDat)
    
    chunkIndcs = [];
    
    xBins = nrnDat(trial).data;
    yBins = y(trial).labels;
    
    % get the count of each behavior type across this trial
    IOtimes = allTimes(trial);
    
    
    IOmat = cell2mat(struct2cell(IOtimes));
    
    numEpochs = size(IOmat,1);
    numUnits = size(xBins,1);
    
    chunkBins = repmat(IOmat(:,1),1,chunkSize) + [0:binAdds];
    
    
    try
        % get y labels
        yLabels0 = yBins(IOmat(:,1));
        numBins = size(yLabels0,1);
        slide = 3;
        numStrides = 4;
        yLabels = reshape(repmat(yLabels0,1,numStrides),numBins*numStrides,1);
        
        % just use yLabels0 if you want to avoid sliding window
        yChunks(trial).y = yLabels;% yBins(IOmat(:,1));
        
        % loop over the chunk bins and assign all bins within a row to a chunk
        CHUNKS0 = [];
        
        
        J = repmat([1:numEpochs],1,numStrides); % vector of repeating epoch #s
        
        for ii = 2:numStrides:numStrides*numEpochs
            if (chunkBins(J(ii-1),1)-4) > 0
                CHUNKS0(:,:,ii-1) = xBins(:,chunkBins(J(ii-1),:)-2);
                CHUNKS0(:,:,ii) = xBins(:,chunkBins(J(ii-1),:)+0); % sliding window of 2 bins for next one
                CHUNKS0(:,:,ii+1) = xBins(:,chunkBins(J(ii-1),:)+2); % sliding window of 2 bins for next one
                CHUNKS0(:,:,ii+2) = xBins(:,chunkBins(J(ii-1),:)+4); % sliding window of 2 bins for next one
            end
        end
        disp(ii)
        
        xChunks0 = xBins(:,chunkBins); % get
        xChunks1 = xBins(:,chunkBins+1); % next window of chunks, slid over by one bin
        
        % FIRST: Reshape xChunks0 and xChunks1; THEN concatenate them along the
        % correct dimension.
        
        try
            xReshape0 = reshape(xChunks0,numUnits,chunkSize,numEpochs);
            xReshape1 = reshape(xChunks1,numUnits,chunkSize,numEpochs);
            
        catch
            keyboard
        end
        
        % xChunks0 = cat(3,xReshape0,xReshape1); % uncomment to skip sliding window
        
        %%%%%% UNCOMMENT to put back to the old broken way?
        xChunks0 = CHUNKS0;
        %xChunks(trial).dat = reshape(xChunks0,numUnits,chunkSize,numEpochs*slide);
        xChunks(trial).data = xChunks0;
        disp(size(xChunks0))
        disp(size(yLabels))
        
    catch
        keyboard
    end
    
end
end

function [perClassAccs] = perClassAccuracies(CM,order)
%CM(I,J) = CM(Y_true, Y_hat)
% represents the count of instances whose
% TRUE labels are group I
%  and whose predicted group labels are group J.

perClassAccs = {};
classNames = unique(order);%{'approach','escape','freeze','stretch'};
for whichClass = 1:length(classNames)
    thisClassName = classNames{whichClass};
    classIndx = find(contains(order,thisClassName));
    numTruePos = CM(classIndx,classIndx);
    BxRow = CM(classIndx,:);
    BxCol = CM(:,classIndx);
    perClassAccs{whichClass} = {thisClassName,squeeze(numTruePos/sum(BxRow))};
   % disp(perClassAccs{whichClass})
end

end

function [] = makeAveragePlots(nrnDat,allTimes,remove)

allAverages = struct;
for day = 1:3
    thisDat =nrnDat(day).data;
    thisDat(:,remove) = [];
    theseTimes = allTimes(day);
    
    appMeans = [];
    for jj = 1:size(theseTimes.app,1)
        appTimes = theseTimes.app(jj,:);
        appTimes(1) = appTimes(1) - 14;
        appTimes(2) = appTimes(1) + 2*14;
        try
            if appTimes(1) > 0
                thisAppMean = mean(nrnDat(day).data(:,appTimes(1):appTimes(2)));
            else
                continue;
            end
        catch
            keyboard
        end
        appMeans = [appMeans;thisAppMean];
    end
    
    escMeans = [];
    for jj = 1:size(theseTimes.esc,1)
        escTimes = theseTimes.esc(jj,:);
        escTimes(1) = escTimes(1) - 14;
        escTimes(2) = escTimes(1) + 2*14;
        if escTimes(1) > 0
            thisEscMean = mean(nrnDat(day).data(:,escTimes(1):escTimes(2)));
        else, continue
        end
        escMeans = [escMeans;thisEscMean];
    end
    
    frzMeans = [];
    for jj = 1:size(theseTimes.frz,1)
        frzTimes = theseTimes.frz(jj,:);
        frzTimes(1) = frzTimes(1) - 14;
        frzTimes(2) = frzTimes(1) + 2*14;
        if frzTimes(1) > 0
            thisFrzMean = mean(nrnDat(day).data(:,frzTimes(1):frzTimes(2)));
        else
            continue
        end
        frzMeans = [frzMeans;thisFrzMean];
    end
    
    strMeans = [];
    for jj = 1:size(theseTimes.str,1)
        strTimes = theseTimes.str(jj,:);
        strTimes(1) = strTimes(1) - 14;
        strTimes(2) = strTimes(1) + 2*14;
        if strTimes(1) > 0
            thisStrMean = mean(nrnDat(day).data(:,strTimes(1):strTimes(2)));
        else
            continue
        end
        strMeans = [strMeans;thisStrMean];
    end
    allAverages(day).app = mean(appMeans);
    allAverages(day).esc = mean(escMeans);
    allAverages(day).frz = mean(frzMeans);
    allAverages(day).str = mean(strMeans);
    
end

appAv = [];
escAv = [];
frzAv = [];
strAv = [];
for day = 1:3
    figure;

    for sp = 1:4
        subplot(2,2,sp)
        if sp == 1
            plot(allAverages(day).app)
            title('Approach')
        elseif sp == 2
            plot(allAverages(day).esc)
            title('Escape')
        elseif sp == 3
            plot(allAverages(day).frz)
            title('Freeze')
        else
            plot(allAverages(day).str)
            title('Stretch')
        end

    end
    suptitle(['Day ',num2str(day)])
    appAv = [appAv;allAverages(day).app];
    escAv = [escAv;allAverages(day).esc];
    frzAv = [frzAv;allAverages(day).frz];
    strAv = [strAv;allAverages(day).str];
    
end
% plot the averages across all days
figure; 
    for sp = 1:4
        subplot(2,2,sp)
        if sp == 1
            plot(mean(appAv))
            title('Approach')
        elseif sp == 2
            plot(mean(escAv))
            title('Escape')
        elseif sp == 3
            plot(mean(frzAv))
            title('Freeze')
        else
            plot(mean(strAv))
            title('Stretch')
        end

    end
end