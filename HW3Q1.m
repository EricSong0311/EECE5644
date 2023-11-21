
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Expected risk minimization with 2 classes
clear all;
close all;
%Switches to bypass parts 1 and 2 for debugging
dimensions=3; %Dimension of data
numLabels=4;
Lx={'L0', 'L1','L2','L3'};
% For min-Perror design, use 0-1 loss
lossMatrix = ones(numLabels,numLabels)-eye(numLabels);
muScale=5;
SigmaScale=4;
%Define data
D.d100.N=100;
D.d500.N=500;
D.d1k.N=1e3;
D.d5k.N=5e3;
D.d10k.N=1e4;
D.d100k.N=100e3;
dTypes=fieldnames(D);
%Define Statistics
p=ones(1,numLabels)/numLabels; %Prior
%Label data stats
mu.L0=muScale*[1 1 0]';
RandSig=SigmaScale*rand(dimensions,dimensions);
Sigma.L0(:,:,1)=RandSig*RandSig'+eye(dimensions);
mu.L1=muScale*[1 0 0]';
RandSig=SigmaScale*rand(dimensions,dimensions);
Sigma.L1(:,:,1)=RandSig*RandSig'+eye(dimensions);
mu.L2=muScale*[0 1 0]';
RandSig=SigmaScale*rand(dimensions,dimensions);
Sigma.L2(:,:,1)=RandSig*RandSig'+eye(dimensions);
mu.L3=muScale*[0 0 1]';
RandSig=SigmaScale*rand(dimensions,dimensions);
Sigma.L3(:,:,1)=RandSig*RandSig'+eye(dimensions);

mu.L0
mu.L1
mu.L2
mu.L3
Sigma.L0(:,:,1)
Sigma.L1(:,:,1)
Sigma.L2(:,:,1)
Sigma.L3(:,:,1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Generate Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ind=1:length(dTypes)
    D.(dTypes{ind}).x=zeros(dimensions,D.(dTypes{ind}).N); %Initialize Data
[D.(dTypes{ind}).x,D.(dTypes{ind}).labels,...
D.(dTypes{ind}).N_l,D.(dTypes{ind}).p_hat]=...
genData(D.(dTypes{ind}).N,p,mu,Sigma,Lx,dimensions);
end
%Plot Training Data
figure;
for ind=1:length(dTypes)
subplot(3,2,ind);
plotData(D.(dTypes{ind}).x,D.(dTypes{ind}).labels,Lx);
legend 'show';
title([dTypes{ind}]);
end
%Plot Validation Data
figure;
plotData(D.(dTypes{ind}).x,D.(dTypes{ind}).labels,Lx);
legend 'show';
title([dTypes{end}]);   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Determine Theoretically Optimal Classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ind=1:length(dTypes)-1
    [D.(dTypes{ind}).opt.PFE, D.(dTypes{ind}).opt.decisions]=...
        optClass(lossMatrix,D.(dTypes{ind}).x,mu,Sigma,...
        p,D.(dTypes{ind}).labels,Lx);

% Call the plotMAPResult function here
    plotMAPResult(D.(dTypes{ind}).x, D.(dTypes{ind}).labels, D.(dTypes{ind}).opt.decisions, Lx);

    opPFE(ind)=D.(dTypes{ind}).opt.PFE;
    fprintf('Optimal pFE, N=%1.0f: Error=%1.2f%%\n',...
    D.(dTypes{ind}).N,100*D.(dTypes{ind}).opt.PFE);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Train and Validate Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numPerc=15; %Max number of perceptrons to attempt to train
k=10; %number of folds for kfold validation
for ind=1:length(dTypes)-1
%kfold validation is in this function
[D.(dTypes{ind}).net,D.(dTypes{ind}).minPFE,...
D.(dTypes{ind}).optM,valData.(dTypes{ind}).stats]=...
kfoldMLP_NN(numPerc,k,D.(dTypes{ind}).x,...
D.(dTypes{ind}).labels,numLabels);
%Produce validation data from test dataset
valData.(dTypes{ind}).yVal=D.(dTypes{ind}).net(D.d100k.x);
[~,valData.(dTypes{ind}).decisions]=max(valData.(dTypes{ind}).yVal);
valData.(dTypes{ind}).decisions=valData.(dTypes{ind}).decisions-1;
%Probability of Error is wrong decisions/num data points
valData.(dTypes{ind}).pFE=...
sum(valData.(dTypes{ind}).decisions~=D.d100k.labels)/D.d100k.N;
outpFE(ind,1)=D.(dTypes{ind}).N;
outpFE(ind,2)=valData.(dTypes{ind}).pFE;
outpFE(ind,3)=D.(dTypes{ind}).optM;
fprintf('NN pFE, N=%1.0f: Error=%1.2f%%\n',...
D.(dTypes{ind}).N,100*valData.(dTypes{ind}).pFE);
end
%----------------------------------
% 循环训练的 MLP 模型
for ind=1:length(dTypes)-1
    % 使用 MLP 模型进行分类
    yVal = D.(dTypes{ind}).net(D.d100k.x);
    [~, predictedLabels] = max(yVal);
    predictedLabels = predictedLabels - 1;

    % 绘制分类结果
    plotResult(D.d100k.x, D.d100k.labels, predictedLabels, Lx);

end
%----------------------------------
%This code was used to plot the results from the data generated in the main
%function
%Extract cross validation results from structure
for ind=1:length(dTypes)-1
[~,select]=min(valData.(dTypes{ind}).stats.mPFE);
M(ind)=(valData.(dTypes{ind}).stats.M(select));
N(ind)=D.(dTypes{ind}).N;
end
%Plot number of perceptrons vs. pFE for the cross validation runs
for ind=1:length(dTypes)-1
    figure;
    plot(valData.(dTypes{ind}).stats.M, valData.(dTypes{ind}).stats.mPFE, 'o-', 'LineWidth', 2);
    xlabel('Number of Perceptrons');
    ylabel('pFE');
    title(['Probability of Error vs. Number of Perceptrons for ' dTypes{ind}]);
    grid on;
end

% for ind=1:length(dTypes)-1
% figure;
% stem(valData.(dTypes{ind}).stats.M,valData.(dTypes{ind}).stats.mPFE);
% xlabel('Number of Perceptrons');
% ylabel('pFE');
% title(['Probability of Error vs. Number of Perceptrons for ' dTypes{ind}]);
% end
%Number of perceptrons vs. size of training dataset
figure,semilogx(N(1:end),M(1:end),'o','LineWidth',2)
grid on;
xlabel('Number of Data Points')
ylabel('Optimal Number of Perceptrons')
ylim([0 15]);
xlim([50 10^4]);
title('Optimal Number of Perceptrons vs. Number of Data Points');
%Prob. of Error vs. size of training data set
figure,semilogx(outpFE(1:end,1),outpFE(1:end,2),'o-','LineWidth',2)
xlim([90 10^4]);
hold all;semilogx(xlim,[opPFE(end) opPFE(end)],'r--','LineWidth',2)
legend('NN pFE','Optimal pFE')
grid on
xlabel('Number of Data Points')
ylabel('pFE')
title('Probability of Error vs. Data Points in Training Data');

%-----------------------------------------------------------
figure;
semilogx(N, outpFE(:,2), 'o-', 'LineWidth', 2); % 绘制 MLP 的错误率
hold on;
plot(xlim, [mean(opPFE) mean(opPFE)], 'r--', 'LineWidth', 2); % 理论最优分类器的错误率
xlabel('Number of Training Samples');
ylabel('Probability of Error');
title('Test Probability of Error vs Training Data Size');
legend('MLP Classifier', 'Theoretical Optimal Classifier');
grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function performs the cross validation and model selection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [minPFE,decisions]=optClass(lossMatrix,x,mu,Sigma,p,labels,Lx)
% Determine optimal probability of error
symbols='ox+*v';
numLabels=length(Lx);
N=length(x);
for ind = 1:numLabels
pxgivenl(ind,:) =...
evalGaussian(x,mu.(Lx{ind}),Sigma.(Lx{ind})); % Evaluate p(x|L=l)
end
px = p*pxgivenl; % Total probability theorem
classPosteriors = pxgivenl.*repmat(p',1,N)./repmat(px,numLabels,1); % P(L=l|x)
% Expected Risk for each label (rows) for each sample (columns)
expectedRisks =lossMatrix*classPosteriors;
% Minimum expected risk decision with 0-1 loss is the same as MAP
[~,decisions] = min(expectedRisks,[],1);
decisions=decisions-1; %Adjust to account for L0 label
fDecision_ind=(decisions~=labels);%Incorrect classificiation vector
minPFE=sum(fDecision_ind)/N;
%Plot Decisions with Incorrect Results
figure;
for ind=1:numLabels
class_ind=decisions==ind-1;
plot3(x(1,class_ind & ~fDecision_ind),...
x(2,class_ind & ~fDecision_ind),...
x(3,class_ind & ~fDecision_ind),...
symbols(ind),'Color',[0.39 0.83 0.07],'DisplayName',...
['L' num2str(ind) ' Correct Classification']);
hold on;
plot3(x(1,class_ind & fDecision_ind),...
x(2,class_ind & fDecision_ind),...
x(3,class_ind & fDecision_ind),...
['r' symbols(ind)],'DisplayName',...
['L' num2str(ind) ' Incorrect Classification']);
hold on;
end
xlabel('x1');
ylabel('x2');
zlabel('x3');
grid on;
title('X Vector with Incorrect Classifications');
legend 'show';
if 0
%Plot Decisions with Incorrect Decisions
figure;
for ind2=1:numLabels
subplot(3,2,ind2);
for ind=1:numLabels
class_ind=decisions==ind-1;
plot3(x(1,class_ind),x(2,class_ind),x(3,class_ind),...
'.','DisplayName',['Class ' num2str(ind)]);
hold on;
end
plot3(x(1,fDecision_ind & labels==ind2),...
x(2,fDecision_ind & labels==ind2),...
x(3,fDecision_ind & labels==ind2),...
'kx','DisplayName','Incorrectly Classified','LineWidth',2);
ylabel('x2');
grid on;
title(['X Vector with Incorrect Decisions for Class ' num2str(ind2)]);
if ind2==1
legend 'show';
elseif ind2==4
xlabel('x1');
end
end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,labels,N_l,p_hat]= genData(N,p,mu,Sigma,Lx,d)
%Generates data and labels for random variable x from multiple gaussian
%distributions
numD = length(Lx);
cum_p = [0,cumsum(p)];
u = rand(1,N);
x = zeros(d,N);
labels = zeros(1,N);
for ind=1:numD
pts = find(cum_p(ind)<u & u<=cum_p(ind+1));
N_l(ind)=length(pts);
x(:,pts) = mvnrnd(mu.(Lx{ind}),Sigma.(Lx{ind}),N_l(ind))';
labels(pts)=ind-1;
p_hat(ind)=N_l(ind)/N;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotData(x,labels,Lx)
%Plots data
for ind=1:length(Lx)
pindex=labels==ind-1;
plot3(x(1,pindex),x(2,pindex),x(3,pindex),'.','DisplayName',Lx{ind});
hold all;
end
grid on;
xlabel('x1');
ylabel('x2');
zlabel('x3');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%
function [outputNet,outputPFE, optM, stats]=kfoldMLP_NN(numPerc,k,x,labels,numLabels)
%Assumes data is evenly divisible by partition choice which it should be
N=length(x);
numValIters=10;
%Create output matrices from labels
y=zeros(numLabels,length(x));
for ind=1:numLabels
y(ind,:)=(labels==ind-1);
end
%Setup cross validation on training data
partSize=N/k;
partInd=[1:partSize:N length(x)];
%Perform cross validation to select number of perceptrons
for M=1:numPerc
for ind=1:k
index.val=partInd(ind):partInd(ind+1);
index.train=setdiff(1:N,index.val);
%Create object with M perceptrons in hidden layer
net=patternnet(M);
% net.layers{1}.transferFcn = 'softplus';%didn't work
%Train using training data
net=train(net,x(:,index.train),y(:,index.train));
%Validate with remaining data
yVal=net(x(:,index.val));
[~,labelVal]=max(yVal);
labelVal=labelVal-1;
pFE(ind)=sum(labelVal~=labels(index.val))/partSize;
end
%Determine average probability of error for a number of perceptrons
avgPFE(M)=mean(pFE);
stats.M=1:M;
stats.mPFE=avgPFE;
end
%Determine optimal number of perceptrons
[~,optM]=min(avgPFE);
%Train one final time on all the data
for ind=1:numValIters
netName(ind)={['net' num2str(ind)]};
finalnet.(netName{ind})=patternnet(optM);
% finalnet.layers{1}.transferFcn = 'softplus';%Set to RELU
finalnet.(netName{ind})=train(net,x,y);
yVal=finalnet.(netName{ind})(x);
[~,labelVal]=max(yVal);
labelVal=labelVal-1;
pFEFinal(ind)=sum(labelVal~=labels)/length(x);
end
[minPFE,outInd]=min(pFEFinal);
stats.finalPFE=pFEFinal;
outputPFE=minPFE;
outputNet=finalnet.(netName{outInd});
end
%-----------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to plot MLP classification results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotResult(x, trueLabels, predictedLabels, Lx)
    % Define markers and colors for correct and incorrect classifications
    markers = {'o', '+', '*', 'x', 's', 'd', '^', 'v', '>', '<'};
    colors = {'b', 'g', 'r', 'c', 'm', 'y', 'k', [0.5 0.5 0.5]}; % Add more colors if needed

    % Initialize figure
    figure;
    hold on;
    grid on;

    numClasses = length(Lx);
    for c = 1:numClasses
        % Indices for correct and incorrect classifications for this class
        correctIndices = (trueLabels == c-1) & (predictedLabels == c-1);
        incorrectIndices = (trueLabels ~= c-1) & (predictedLabels == c-1);

        % Plot correctly classified samples
        plot3(x(1, correctIndices), x(2, correctIndices), x(3, correctIndices), ...
              markers{c}, 'Color', colors{c}, 'DisplayName', [Lx{c} ' Correct']);

        % Plot incorrectly classified samples
        plot3(x(1, incorrectIndices), x(2, incorrectIndices), x(3, incorrectIndices), ...
              markers{c}, 'Color', colors{c}, 'LineStyle', 'none', 'MarkerFaceColor', 'none', ...
              'DisplayName', [Lx{c} ' Incorrect']);
    end

    % Labels and title
    xlabel('x1');
    ylabel('x2');
    zlabel('x3');
    title('MLP Classification Results');
    legend('show');
    view(3); % Set the view to 3D
    hold off;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to plot MAP classification results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotMAPResult(x, trueLabels, mapDecisions, Lx)
    % Define markers and colors for correct and incorrect classifications
    markers = {'o', '+', '*', 'x', 's', 'd', '^', 'v', '>', '<'};
    colors = {'b', 'g', 'r', 'c', 'm', 'y', 'k', [0.5 0.5 0.5]}; % Add more colors if needed

    % Initialize figure
    figure;
    hold on;
    grid on;

    numClasses = length(Lx);
    for c = 1:numClasses
        % Indices for correct and incorrect classifications for this class
        correctIndices = (trueLabels == c-1) & (mapDecisions == c-1);
        incorrectIndices = (trueLabels ~= c-1) & (mapDecisions == c-1);

        % Plot correctly classified samples
        plot3(x(1, correctIndices), x(2, correctIndices), x(3, correctIndices), ...
              markers{c}, 'Color', colors{c}, 'DisplayName', [Lx{c} ' Correct']);

        % Plot incorrectly classified samples
        plot3(x(1, incorrectIndices), x(2, incorrectIndices), x(3, incorrectIndices), ...
              markers{c}, 'Color', colors{c}, 'LineStyle', 'none', 'MarkerFaceColor', 'none', ...
              'DisplayName', [Lx{c} ' Incorrect']);
    end

    % Labels and title
    xlabel('x1');
    ylabel('x2');
    zlabel('x3');
    title('MAP Classification Results');
    legend('show');
    view(3); % Set the view to 3D
    hold off;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


