%function [] = makeAveragePlots(nrnDat,allTimes,remove)
% Friday November 2

% This program loads the data sent on Monday, Nov. 19, which is from
% a new mouse with more confident mini-scope implant location

cd('~/Desktop/newMouseDat/newPAG/VertSessions') % directory where the data is stored
%%
mapping=load('cellRegistered_20181119_111616.mat');
mapStr = mapping.cell_registered_struct;
cellMap = mapStr.cell_to_index_map;
eps = 0.01;

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
map4 = cellMap(:,4);

nrnIds1 = map1(map1>0);
nrnIds2 = map2(map2>0);
nrnIds3 = map3(map3>0);
nrnIds4 = map4(map4>0);

nrnIDs = {nrnIds1, nrnIds2, nrnIds3, nrnIds4}; % group them into a cell
% When inside a loop from 1 to 3:

trialDirs = {'H12_M37_S44','H14_M55_S37','H16_M53_S51','H19_M1_S10'}; % name of the folders
keyboard

highAccUnits = [58, 45, 42;  84, 68, 68;  92, 81, 76;  118, 98, 102; ...
    122, 105, 109;  136, 118, 130];
%goodIDs = {highAccUnits(:,1),highAccUnits(:,2),highAccUnits(:,3)};

nrnDat = struct;
m = 10; % dimensionality of reduced data
y = struct;
numDays = numel(trialDirs);
for ii = 1:numDays
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
    [escCoeff,score] = pca(nrnDat(ii).data');
    
   % nrnDat(ii).features = score(:,1:m)';
    
    %%%%%%%%%%%%%%%%%%%%%%%
    % now make the labels %
    % load the behavioral transition times
    % get the indices for the start of each behavior
    
    labels = cell(size(nrnDat(ii).data,2),1);
    labels(:) = {'0'};
    
    % goodLabels = cell(size(goodDat(ii).data,2),1);
    % goodLabels(:) = {'0'};
    
    BxTimes = load('Behavior_MS.mat');
    
    
    times.app = [];
    times.esc = [];
    times.str = [];
    times.vrt = [];
    
    appTimes = BxTimes.approachFrameMS; % approach
    for jj = 1:size(appTimes,1)
        if appTimes(jj,2) <= numBins
            times.app = [times.app, appTimes(jj,1):appTimes(jj,2)];
        end
    end; labels(times.app) = {'approach'};
    goodLabels(times.app) = {'approach'};
    
    
    escTimes = BxTimes.escapeFrameMS; % escape
    for jj = 1:size(escTimes,1)
        if escTimes(jj,2) <= numBins
            times.esc = [times.esc, escTimes(jj,1):escTimes(jj,2)];
        end
    end; labels(times.esc) = {'escape'};
    
    
    vertTimes = BxTimes.vertFrameMS; % freeze
    for jj = 1:size(vertTimes,1)
        if vertTimes(jj,2) <= numBins
            times.vrt = [times.vrt, vertTimes(jj,1):vertTimes(jj,2)];
        end
    end; labels(times.vrt) = {'vert'};
    
    % nothing
    
    
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
    
    allTimes(ii).vrt = vertTimes;
    
    allTimes(ii).str = strTimes;
end



%% Troubleshoot the neural identities
MEANS = [];
for ii = 1:numDays
    YY = y(ii).labels;
    XX = nrnDat(ii).data;
    classInds(ii).app = find(contains(YY,'approach'));
    classInds(ii).esc = find(contains(YY,'escape'));
    %  classInds(ii).frz = find(contains(YY,'freeze'));
    classInds(ii).str = find(contains(YY,'stretch'));
    
    bxData(ii).app  = normX(XX(:,classInds(ii).app));
    bxData(ii).esc  = normX(XX(:,classInds(ii).esc));
    % bxData(ii).frz  = normX(XX(:,classInds(ii).frz));
    bxData(ii).str  = normX(XX(:,classInds(ii).str));
    
    
end

% Find units to remove

for ii = 1:numDays
    eachDay(ii).app = mean(normX(bxData(ii).app),2);
    eachDay(ii).esc = mean(normX(bxData(ii).esc),2);
    %eachDay(ii).frz = mean(normX(bxData(ii).frz),2);
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
%%

allAverages = struct;
%%%%%%%%%%%%%%%%%%%%%%
% TRIMMING PARAMETERS
timeBeforeBxOnset = 2;
Sr = 1; % sample rate == 10
numSecsPerBx = 5;
%%%%%%%%%%%%%%%%%%%%%%
for day = 1:numDays
    thisDat = nrnDat(day).data;
    
    theseTimes = allTimes(day);
    
    appMeans = [];
    appTrials = [];
    for jj = 1:size(theseTimes.app,1)
        % get the in / out times for each trial of this behavior
        appTimes = theseTimes.app(jj,:);
        
        appTimes(1) = appTimes(1) - timeBeforeBxOnset; % Go back a few bins if necessary
        appTimes(2) = appTimes(1) + numSecsPerBx*Sr;   % Go forward in time however many bins
        
        try
            if appTimes(1) > 0
                appTrial = nrnDat(day).data(:,appTimes(1):appTimes(2));
                thisAppMean = mean(appTrial);
            else
                continue;
            end
        catch
            keyboard
        end
        appMeans = [appMeans;thisAppMean];
        appTrials(:,:,jj) = appTrial;
    end
    
    
    
    escMeans = [];
    escTrials = [];
    for jj = 1:size(theseTimes.esc,1)
        escTimes = theseTimes.esc(jj,:);
        escTimes(1) = escTimes(1) - timeBeforeBxOnset;
        escTimes(2) = escTimes(1) + numSecsPerBx*Sr;
        if escTimes(1) > 0
            escTrial = nrnDat(day).data(:,escTimes(1):escTimes(2));
            thisEscMean = mean(escTrial);
        else, continue
        end
        escMeans = [escMeans;thisEscMean];
        escTrials(:,:,jj) = escTrial;
    end
    
    vrtMeans = [];
    vrtTrials = [];
    for jj = 1:size(theseTimes.vrt,1)
        vrtTimes = theseTimes.vrt(jj,:);
        vrtTimes(1) = vrtTimes(1) - timeBeforeBxOnset; % assuming 7Hz frame-rate: 2 seconds
        vrtTimes(2) = vrtTimes(1) + numSecsPerBx*Sr;
        if vrtTimes(1) > 0
            if vrtTimes(2) <= size(nrnDat(day).data,2)
                vrtTrial = nrnDat(day).data(:,vrtTimes(1):vrtTimes(2));
                thisVrtMean = mean(vrtTrial);
            end
        else
            continue
        end
        vrtMeans = [vrtMeans;thisVrtMean];
        vrtTrials(:,:,jj) = vrtTrial;
    end
    
    strMeans = [];
    strTrials = [];
    for jj = 1:size(theseTimes.str,1)
        strTimes = theseTimes.str(jj,:);
        strTimes(1) = strTimes(1) - timeBeforeBxOnset;
        strTimes(2) = strTimes(1) + numSecsPerBx*Sr; % used to be 2*14 = 4 seconds total
        if strTimes(1) > 0
            strTrial = nrnDat(day).data(:,strTimes(1):strTimes(2));
            thisStrMean = mean(strTrial);
        else
            continue
        end
        strMeans = [strMeans;thisStrMean];
        strTrials(:,:,jj) = strTrial;
    end
    allAverages(day).app = mean(appMeans);
    allAverages(day).esc = mean(escMeans);
    allAverages(day).vrt = mean(vrtMeans);
    allAverages(day).str = mean(strMeans);
    
    allTrials(day).app = appTrials;
    allTrials(day).esc = escTrials;
    allTrials(day).vrt = vrtTrials;
    allTrials(day).str = strTrials;
    
    allAverages(day).appMeans = appMeans;
    allAverages(day).escMeans = escMeans;
    allAverages(day).vrtMeans = vrtMeans;
    allAverages(day).strMeans = strMeans;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the trimmed version of the data.
X_trim = struct;
Y_trim = struct;
for day = 1:numDays
    bxNames = fieldnames(allTrials(day));
    trimDatCell = struct2cell(allTrials(day));
    % Preallocate + Erase
    binIDs = [];
    chkIDs = [];
    trimDataToday_bins = [];
    trimDataToday_chunks = [];
    for bx = 1:4
        trimDataMat = cell2mat(trimDatCell(bx));
        matSize = size(trimDataMat);
        % Treat chunks and bins differently
        numChks_thisBx = matSize(end);
        numBins_thisBx = prod(matSize(2:end)); % how many bins total for this behavior on this day
        reshapedBx_Bins = reshape(trimDataMat,matSize(1),numBins_thisBx);
        binIDs = [binIDs; bx*ones(numBins_thisBx,1)];
        chkIDs = [chkIDs; bx*ones(numChks_thisBx,1)];
        
        trimDataToday_bins = [trimDataToday_bins,reshapedBx_Bins]; % Keep adding more bins to the right of this matrix
        trimDataToday_chunks = cat(3,trimDataToday_chunks,trimDataMat);
    end
    
    trim_binLabels = cell(length(binIDs),1);
    trim_chunkLabels = cell(length(chkIDs),1);
    
    % Now assign labels inside the cell
    for bx = 1:4
        trim_binLabels(binIDs==bx) = bxNames(bx);
        trim_chunkLabels(chkIDs==bx) = bxNames(bx);
    end
    X_trim(day).bins = trimDataToday_bins;
    X_trim(day).chunks = trimDataToday_chunks;
    Y_trim(day).binLabels = trim_binLabels;
    Y_trim(day).chunkLabels = trim_chunkLabels;
end

appAv = [];
escAv = [];
vrtAv = [];
strAv = [];
for day = 1:numDays
 
    appAv = [appAv;allAverages(day).app];
    escAv = [escAv;allAverages(day).esc];
    vrtAv = [vrtAv;allAverages(day).vrt];
    strAv = [strAv;allAverages(day).str];
    
end
% plot the averages across all days
figure;
lw = 1.4;
x_len = size(appAv,2);
x_vec = [1:x_len] - (timeBeforeBxOnset + 1);
for sp = 1:4
    subplot(2,2,sp)
    if sp == 1
        plot(x_vec,mean(appAv),'LineWidth',lw)
        title('Approach')
        xlabel('time bin')
    elseif sp == 2
        plot(x_vec,mean(escAv),'LineWidth',lw)
        title('Escape')
    elseif sp == 3
        plot(x_vec,mean(vrtAv),'LineWidth',lw)
        title('Vert')
    else
        plot(x_vec,mean(strAv),'LineWidth',lw)
        title('Stretch')
    end
    
end
suptitle('Averages across all days')

%% train within days
%
% train an SVM; split the data.
kernel = 'linear';
for kk = 1:numDays
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
        % Fit the model
        Mdl = fitcecoc(Xtrain',Ytrain,'Learners',t,...
            'Verbose',2);
        models(kk).model = Mdl;
        models(kk).dir = nrnDat(kk).dir;
        
        
        % % get accuracy % %
        % trainingAcc
        Yhat_train = predict(Mdl,Xtrain');
        [C_M_train, orderTrain] = confusionmat(Ytrain,Yhat_train);
        acc_train = sum(diag(C_M_train)/sum(sum(C_M_train)));
        
        % testAccCM
        
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


%% train across days

foldIDs = 1:4;
lam = 2e-1;

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
    Xtrain = normX(Xtrain); %(Xtrain - mean(Xtrain,1)) %normalize(Xtrain,dim);
    Xtest = normX(Xtest);

    kernel = 'linear';
    [Xtrain,Ytrain] = removeZeros(Xtrain,Ytrain);
    hyperOpts = struct('Optimizer','bayesopt','KFold',5,...
        'MaxObjectiveEvaluations',3,'Regularization','lasso');
    %   t = templateSVM('Standardize',1,'KernelFunction',kernel,...
    %         'IterationLimit',1e6);
    t = templateSVM('KernelFunction', kernel, 'IterationLimit', 2e5, ...
        'Standardize', 0, 'DeltaGradientTolerance', 1e-15, ...
        'OutlierFraction', 0);
    
    % bayesOpt Lambdas: 3e-7
    % best lambda: 8e-1 for RidgeRegression
    l = templateLinear('lambda', lam, 'regularization', 'ridge', ...
        'Solver', 'bfgs', 'FitBias', 0, 'IterationLimit', 20000000, ...
        'GradientTolerance', 1e-16, 'BetaTolerance', 1e-15);
    Mdl = fitcecoc(Xtrain',Ytrain,'Learners',t,...
        'Verbose',0);%,...
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



%% format for threatAxis

exampleTrial = allTrials(1).app(:,:,1);
[N,T] = size(exampleTrial);
D = numDays; % three days
conds = {'app','esc','str','vrt'};
S = numel(conds);

% trialNum: N x S x D
% number of trials for each neuron in each S,D condition (is
% usually different for different conditions and different sessions)

% trialNum(i,B,D) = number of trials for neuron i on day D for behavior B.
trialNum = zeros(N,S,D);
for B = 1:S
    for day = 1:D
        Bx = conds{B};
        thisExp = getfield(allTrials(day),Bx);
        
        trialNum(:,B,day) = size(thisExp,3);
    end
end

maxTrialNum = max(max(max(trialNum)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% firingRates: N x S x D x T x maxTrialNum

% all single-trial data together, massive array. Here
% maxTrialNum is the maximum value in trialNum. E.g. if the number of
% trials per condition varied between 1 and 20, then maxTrialNum = 20. For
% the neurons and conditions with less trials, fill remaining entries in
% firingRates with zeros or nans.

firingRates = NaN*zeros(N,S,D,T,maxTrialNum);
% 30 x 4 x 3 x 29 x 25
for B = 1:S
    Bx = conds{B};
    for day = 1:D
        for trial = 1:maxTrialNum
            try
                thisExp = getfield(allTrials(day),Bx);
                firingRates(:,B,day,:,trial) = thisExp(:,:,trial);
            catch
                if size(thisExp,3) >= trial
                    disp('Something actually broke ... ')
                    keyboard
                end
                continue
            end
        end
    end
end

firingRatesAverage = nanmean(firingRates,5);
firingRatesStd = nanstd(firingRates,0,5);

%% Plot threat axis
%% Single PCs: Plot Projections
addpath(genpath('/Users/awick/Documents/MATLAB/KaoLab/toolboxes/dr_toolbox'))
% Start with [mean(A), mean(E), mean(F), mean(S)] to get PC1.
% ALSO get PC2.

% To plot: Get projection of {each time bin onto PC1;} {each tb onto PC2},
% then plot that as a 2D point.
% Label PC1, PC2 x/y axes.

% Then, for each trial, project each time bin as a N-dim vector onto PC1,
% and

proj =@(a,b) dot(a,b)/sqrt(sum(b.^2));
%colors = {'r','g','b','k'};
% Green Red Orange Yellow
colors =  [ 0.0 0.8 0.0; ... % Green
    0.95 0.2 0.0; ... % Red
    .1 0.5 0.9;  ... % Orange
    0.8 0.85 0.2; ... % Yellow
    ];
catData = [];
allProj = nan*zeros(4,3,10,9); %???
for day = 1:D
    % 2 PCs
    ProjFig2 = figure;
    pAx2 = gca;
    % 1 PC
    ProjFig1 = figure;
    pAx1 = gca;
    for bx = 1:4
        C = colors(bx,:);
        % firingRatesAverage:     30     4     3    29 --
        % Average across all trials for each day.
        catData = [catData, squeeze(firingRatesAverage(:,bx,day,:))];
        allPCs = pca(catData); % should be N x D ... first dim = numUnits
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Local Linear Embedding goes here
        numDims = 12;
        k = 7; % neighbors
        
        % Original data size is N x T -- neurons by bins
        catData = normalize(catData); 
        lle_x = lle(catData,numDims,k);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        PC1 = allPCs(:,1);
        PC2 = allPCs(:,2);
       % PC3 = allPCs(:,3);
      %  PC4 = allPCs(:,4);
        
        % Now loop over the number of trials.
        theseTrials = squeeze(firingRates(:,bx,day,:,:));
        [N,T,numDays] = size(theseTrials);
        nansGone = theseTrials;
        nanInds = find(isnan(theseTrials));
        %nansGone(find(isnan(theseTrials))) = [];
        if size(nanInds,1) > 0
            nansGone(nanInds) = [];
            
            numLeft = size(nansGone,2);
            nansGone = reshape(nansGone,N,T,numLeft/(N*T));
            
        end
        
        theseTrials = nansGone;
        numDays = size(theseTrials,3);
        
        h = zeros(numDays,1);
        scale = 1;
      %  if day == 1
      %      scale = -1;
      %  end
        for trial = 1:numDays
            pX = [];
            pY = [];
            p3 = [];
         %   p4 = [];
            for t = 1:T % what is this?/???
                
                proj_x = proj(theseTrials(:,t,trial),PC1);
                proj_y = proj(theseTrials(:,t,trial),PC2);
            %    proj_3 = proj(theseTrials(:,t,trial),PC3);
            %      proj_4 = proj(theseTrials(:,t,trial),PC4);
                
                pX = [pX, proj_x];
                pY = [pY, proj_y];
             %   p3 = [p3, proj_3];
           %     p4 = [p4, proj_4];
            end
            set(0, 'CurrentFigure', ProjFig2)
            % h = plot(scale*pX, pY,'.','Color',C,'MarkerSize',10); hold on
            h = plot(scale*pX, pY,'.','Color',C,'MarkerSize',10); hold on
            set(0, 'CurrentFigure', ProjFig1)
            %h1 = plot( pX, zeros(size(pY)),'.','Color',C,'MarkerSize',15); hold on
            
            h1 = scatter(scale*pX,zeros(size(pY)),200,C,'filled','MarkerFaceAlpha',0.45); hold on
            
            theseBxTrials(trial,:) = scale*pX;
            %allProj(bx,day,trial,:) = pX;
        end
        
        nBins = 20;
        numBins = floor(numel(theseBxTrials/2));
        [counts,edges] = histcounts(theseBxTrials,numBins, 'Normalization','probability');
        hf = histfit(theseBxTrials(:),10,'kernel');
        smoothX = hf(2).XData;
        smoothY = hf(2).YData;
        smoothY = smoothY/norm(smoothY);%0.5*normalize(smoothY,'range');
        delete(hf)
        ha = area(scale*smoothX,smoothY);%,'Color',C,'LineWidth',2);
        ha.FaceColor = C; ha.FaceAlpha = 0.3;
        h2 = fitdist(theseBxTrials(:),'kernel','kernel','epanechnikov');
        
    end
    
    for hh = 1:4
        legendColors = [ 0.0 0.8 0.0; ... % Green
            0.8 0.85 0.2; ... % Yellow
            .1 0.5 0.9;   ... % Orange 1 0.5 0
            0.95 0.2 0.0; ... % Red
            ];
        h(hh) = plot(NaN,NaN,'.','Color',legendColors(hh,:),'MarkerSize',15);
    end
    
    fName = strcat('newStuffthreatAxis_day',num2str(day));
    legend(h,'Approach','Stretch','Vert','Escape')
    title(['Day ',num2str(day)])
    print(gcf,fName,'-dsvg')
    print(gcf,fName,'-dpng')
    keyboard
end


%%
% Train on only two tasks: Approach and Stretch
kernel = 'linear';
numDirs = numel(trialDirs);
for kk = 1:numDirs
    crossValAccs = [0];
    
    bestFoldAcc = 0.5;
    for fold = 1:10
        
        % DATA PREPARATION
  
        % split the data: Use the TRIMMED versions (trimmed around behavior
        % onset)
        [Xtrain,Ytrain,Xval,Yval] = tv_split(X_trim(kk).bins,...
            Y_trim(kk).binLabels,0.8);
        
        % Erase the 'Escape' data from training + validation sets
        bxToRemove = 'esc';
        escInds = find(contains(Ytrain,bxToRemove));
        Ytrain(escInds) = [];
        Xtrain(:,escInds) = [];
        eI = find(contains(Yval,bxToRemove));
        Yval(eI) = [];
        Xval(:,eI) = [];
        
        
        % Erase the 'Vert' data from training + validation sets
        bxToRemove = 'vrt';
        vertInds = find(contains(Ytrain,bxToRemove));
        Ytrain(vertInds) = [];
        Xtrain(:,vertInds) = [];
        vI = find(contains(Yval,bxToRemove));
        Yval(vI) = [];
        Xval(:,vI) = [];
        
        
        % TRAIN
        [Mdl,fitInfo] = fitclinear(Xtrain',Ytrain);

        models(kk).model = Mdl;
        models(kk).dir = nrnDat(kk).dir;
        
        % % get accuracy % %
        % trainingAcc
        Yhat_train = predict(Mdl,Xtrain');
        [C_M_train, orderTrain] = confusionmat(Ytrain,Yhat_train);
        acc_train = sum(diag(C_M_train)/sum(sum(C_M_train)));
        
        if acc_train > bestFoldAcc
            bestFoldAcc = acc_train;
            bestFoldModel = Mdl;
            models(kk).bestX = Xtrain;
            models(kk).bestY = Ytrain;
        end
        
        % testAccCM
        Yhat = predict(Mdl,Xval');
        [C_M,order] = confusionmat(Yval,Yhat);
        acc = sum(diag(C_M)/sum(sum(C_M)));
        models(kk).CM = C_M;
        models(kk).acc = acc;
        
        crossValAccs(fold) = acc;
       
        
    end % End of k-fold within day.
    models(kk).bestModel = bestFoldModel;
    models(kk).crossValAcc = mean(crossValAccs);
    disp(mean(crossValAccs))
    
    allCV_Accs = [models.crossValAcc];
   
    SVM_axes(kk).AppStretch = bestFoldModel.Beta;
    
    % Give the best model, as well as the data this model was trained on, 
    % into this function.
    % Also give ALL of the trimmed data, so that the next comparisons can
    % be made as well ( Escape vs. Vert)
    [Xtrain_SV, Ytrain_SV, SVM_axes] = plotThreatAxis(kk, bestFoldModel, models(kk).bestX, models(kk).bestY, X_trim, Y_trim,...
        timeBeforeBxOnset,numSecsPerBx,SVM_axes);
   
end
 [maxVal, bestDay] = max(allCV_Accs);
%SVM_axes.AppStretch = bestFoldModel.Beta;
close(gcf)


%% Now analyze the two SVM axes: AppStretch on top, EscVert on bottom.

% Function to compute orthoganal vector
orthXY = @(X,Y)X - dot(X,Y)*Y/(norm(Y)^2);
% Just do SVD instead; treat the left singular vectors as PCs. 


alph = 0.4;
mSize = 80;
for day = 1:4

% Get orthogonal versions of these two axes. 
appStretchAx = orthXY(SVM_axes(day).AppStretch,SVM_axes(day).EscVert)';
escVertAx = SVM_axes(day).EscVert';

% Now project all Behavior onto them. 
% For each behavior bin:
% 1) Store result of projection onto App axis
% 2) Store result of projection onto Esc axis
% 3) Plot those projections as a point (appProj, escProj)
% Then see if they cluster out. 

X_DAT = nrnDat(kk).data;
Y_DAT = y(kk).labels;
[X_DAT,Y_DAT] = removeZeros(X_DAT,Y_DAT);

numEx = numel(Y_DAT);
allProj_SV = zeros(size(Ytrain_SV));
C3 = [];

cNames = unique(Y_DAT);
colors2D = distinguishable_colors(4);

for ii = 1:numEx 
    thisPt = X_DAT(:,ii);
    label = Y_DAT{ii};
    
    classID = find(contains(cNames,label));
    
    allProj_App(ii) = dot(thisPt,appStretchAx);
    allProj_Esc(ii) = dot(thisPt,escVertAx);
    
    C3(ii,:) = colors2D(classID,:);
end

figure;
sctr3 = scatter(allProj_App,allProj_Esc,mSize,C3,'filled');
set(sctr3,'MarkerFaceAlpha',1.1*alph)

hold on
h = zeros(1,4);
for hh = 1:4
    h(hh) = scatter(NaN,NaN,mSize,colors2D(hh,:),'filled');
    set(h(hh),'MarkerFaceAlpha',1.1*alph)
end
grid on;
legend(h,cNames)
xlabel('Approach vs. Stretch')
ylabel('Escape vs. Vert')
title(['Projections: Day', num2str(day)])

% 
fName3 = strcat('2D_projections_',num2str(timeBeforeBxOnset),'_',num2str(numSecsPerBx));
% print(gcf,fName3,'-dpng')

%% 2D Histogram

colors_dist = linspecer(4);
sHist = scatterhist(allProj_App,allProj_Esc,'Group',Y_DAT,'kernel','on',...
    'Marker','xods','Color',colors_dist,'LineStyle',{'-','-','-','-'});
xlabel('Approach-Stretch Axis')
ylabel('Escape-Vert Axis')
title(['Projections: Day ',num2str(day)])
grid on

for l = 2:max(size(sHist(3).Children))
    try
        sHist(2).Children(l).YData = -sHist(2).Children(l).YData;
        sHist(3).Children(l).YData = -sHist(3).Children(l).YData;
    catch
        keyboard
    end
end

sHist(2).YLim = [-fliplr((sHist(2).YLim))];
sHist(2).Position = sHist(2).Position + [0.0 0.01 0.0 0];
sHist(3).YLim = [-fliplr((sHist(3).YLim))];
sHist(3).Position = sHist(3).Position + [0.01 0 0 0];

set(gcf,'Color','w')
end
%print(gcf,strcat(fName3,'_Hists'),'-dpng')
%% Function Defs
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
classNames = {'approach','escape','vert','stretch'};
for whichClass = 1:numel(classNames)
    thisClassName = classNames{whichClass};
    classIndx = find(contains(order,thisClassName));
    numTruePos = CM(classIndx,classIndx);
    BxRow = CM(classIndx,:);
    BxCol = CM(:,classIndx);
    perClassAccs{whichClass} = {thisClassName,squeeze(numTruePos/sum(BxRow))};
    disp(perClassAccs{whichClass})
    
end

end
function[accs] = newClassAccuracies(CM,order)
[classNames,sortInds] = sort(unique(order));

for whichClass = 1:numel(classNames)
    thisClassName = classNames{whichClass};
    classIndx = find(contains(order,thisClassName));
    numTruePos = CM(classIndx,classIndx);
    BxRow = CM(classIndx,:);
    BxCol = CM(:,classIndx);
    perClassAccs{whichClass} = {thisClassName,squeeze(numTruePos/sum(BxRow))};
    disp(perClassAccs{whichClass})
    keyboard
end
end



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


function [Xtrain_SV, Ytrain_SV, SVM_axes] = plotThreatAxis(kk, Mdl, Xtrain, Ytrain, X_trim, Y_trim,timeBeforeBxOnset,numSecsPerBx,SVM_axes)

close all
beta = Mdl.Beta; % Model classification axis
bias = Mdl.Bias;
subPlotsFig = figure;
colors = [1 0 0; 0 0 1]; % [R G B]
numEx = size(Ytrain,1);
allProj = zeros(size(Ytrain));
C = zeros(numEx,3);


% First get projections of the TRAINING data
for ii = 1:numEx
    thisPt = Xtrain(:,ii);
    label = Ytrain{ii};
    cNames = unique(Ytrain);
    if strcmp(label,cNames{1})
        classID = 1;
    elseif strcmp(label,cNames{2})
        classID = 2;
    end
    
    allProj(ii) = dot(thisPt,beta);
    C(ii,:) = colors(classID,:);
end
% Plotting Parameters
alph = 0.4; % transparency
mSize = 80; % marker size
subplot(212)


% Scatter the points
sctr = scatter(allProj,zeros(size(allProj)),mSize,C,'filled');
set(sctr,'MarkerFaceAlpha',alph);
hold on;

% Now generate a histogram
appIndx = C(:,1)==1;
strIndx = logical(1 - appIndx);

appPts = allProj(appIndx);
strPts = allProj(strIndx);

histogram(appPts,25,'FaceColor',[1 0 0],'FaceAlpha',alph)
histogram(strPts,25,'FaceColor',[0 0 1],'FaceAlpha',alph)

hold on;
h = zeros(2, 1);
h(1) = scatter(NaN,NaN,8*mSize,'r','s','filled','MarkerFaceAlpha',alph);
h(2) = scatter(NaN,NaN,8*mSize,'b','s','filled','MarkerFaceAlpha',alph);
legend(h,cNames{1},cNames{2})
title('Training Data Projections')
grid on
xlabel('Projection')
ylabel('Count')


% Get the data again, but erase Approach and Escape fields.
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Now train on Escape vs. Vert
[Xtrain_SV,Ytrain_SV,Xval_SV,Yval_SV] = tv_split(X_trim(kk).bins,...
    Y_trim(kk).binLabels,0.9); % leave out just 10% of data for validation.

vertInds = find(contains(Ytrain_SV,'app'));
Ytrain_SV(vertInds) = [];
Xtrain_SV(:,vertInds) = [];
vI = find(contains(Yval_SV,'app'))   ;
Xval_SV(:,vI) = [];
Yval_SV(vI) = [];

escInds = find(contains(Ytrain_SV,'str'));
Ytrain_SV(escInds) = [];
Xtrain_SV(:,escInds) = [];
eI = find(contains(Yval_SV,'str'))   ;
Xval_SV(:,eI) = [];
Yval_SV(eI) = [];

% Now actually train it to get another axis.
[Mdl2,fitInfo] = fitclinear(Xtrain_SV',Ytrain_SV);
SVM_axes(kk).EscVert = Mdl2.Beta;
% % get accuracy % %
% trainingAcc
Yhat_trainSV = predict(Mdl2,Xtrain_SV');
[C_M_train, orderTrain] = confusionmat(Ytrain_SV,Yhat_trainSV);
acc_train = sum(diag(C_M_train)/sum(sum(C_M_train)));

% testAccCM
Yhat = predict(Mdl2,Xval_SV');
[C_M,order] = confusionmat(Yval_SV,Yhat);
acc = sum(diag(C_M)/sum(sum(C_M)));

% Now plot stretch and vert on this axis.
% sv_fig = figure;

numEx = size(Ytrain_SV,1);
allProj_SV = zeros(size(Ytrain_SV));
C2 = [];
for ii = 1:numEx 
    thisPt = Xtrain_SV(:,ii);
    label = Ytrain_SV{ii};
    cNames = unique(Ytrain_SV);
    if strcmp(label,cNames{1})
        classID = 1;
    elseif strcmp(label,cNames{2})
        classID = 2;
    end
    
    allProj_SV(ii) = dot(thisPt,beta);
    C2(ii,:) = colors(classID,:);
end

Ca = [C, 0.5*ones(size(C,1),1)];



%figure
subplot(211)
mSize = 80;
sctr = scatter(allProj_SV,zeros(size(allProj_SV)),mSize,C2,'filled');

escIndx = C2(:,1)==1;
appIndx = C2(:,1) == 0;
 
escPts = allProj_SV(escIndx);
appPts = allProj_SV(appIndx);

meanEsc = mean(escPts);
meanApp = mean(appPts);


set(sctr,'MarkerFaceAlpha',0.4);
legend(cNames{1},cNames{2})
% Custom Legend
grid on
hold on;
h = zeros(2, 1);
h(1) = scatter(NaN,NaN,mSize,'r','filled');
h(2) = scatter(NaN,NaN,mSize,'b','filled');

% plot(meanEsc,0,'rx','markersize',30)
% plot(meanApp,0,'bx','markersize',30)
lgnd = legend(h, cNames{1},cNames{2},'mean1','mean2');
title('Projections of Held-out Data onto SVM Axis')

%

subplot(211)
histogram(appPts,25,'FaceColor',[0 0 1],'FaceAlpha',0.4)
hold on
histogram(escPts,25,'FaceColor',[1 0 0],'FaceAlpha',0.4)
h = zeros(2, 1);
h(1) = scatter(NaN,NaN,80,'r','s','filled');
h(2) = scatter(NaN,NaN,80,'b','s','filled');
legend(h, cNames{1},cNames{2});
title('Distribution of Projections of Unseen Data on Trained SVM Axis')

fName2 = strcat('Day',num2str(kk),'_','threatAxisProjHist_',num2str(timeBeforeBxOnset),'_',num2str(numSecsPerBx));
print(gcf,fName2,'-dpng')
%legend(cNames{1},cNames{2})
end
