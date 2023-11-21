clear all ;
close all ;
clc ;
%%
n=2; experiments = 100;
N = [10 100 1000] ;% number of iid samples
num_GMM_picks = zeros(length(N) , 10) ;

% True mu and Sigma values for 4 component GMM
mu_true(:, 1) = [0; 0];   
mu_true(:, 2) = [1; 1];   
mu_true(:, 3) = [5; -5];  
mu_true(:, 4) = [-7; 3];  

% Co-variance matrices
Sigma_true(:, :, 1) = [2 0.5; 0.5 1]; 
Sigma_true(:, :, 2) = [1.5 0.4; 0.4 1]; 
Sigma_true(:, :, 3) = [3 -1; -1 2];     
Sigma_true(:, :, 4) = [2 0; 0 2];     

alpha_true = [0.23 0.32 0.25 0.20]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number of 2-dimentional sample
D.d10.N=10;
D.d100.N=100;
D.d1k.N=1e3;
dTypes=fieldnames(D);
%matrix characterastic
gmmParameters.priors = alpha_true;
gmmParameters.meanVectors = mu_true;
gmmParameters.covMatrices = Sigma_true;
for ind=1:length(dTypes)
    [x,labels] = generateDataFromGMM(D.(dTypes{ind}).N, gmmParameters);
    for j = 1:4;
        Nclass(j,1) = length(find(labels==j));
    end
    figure;
    plot(x(1,labels==1),x(2,labels==1),'g.','DisplayName','Component 1');
    axis equal;
    hold on;
    plot(x(1,labels==2),x(2,labels==2),'b.','DisplayName','Component 2');
    axis equal;
    hold on;
    plot(x(1,labels==3),x(2,labels==3),'r.','DisplayName','Component 3');
    axis equal;
    hold on;
    plot(x(1,labels==4),x(2,labels==4),'y.','DisplayName','Component 4');
    axis equal;
    hold on;
    xlabel('x_1');ylabel('x_2');zlabel('x_3');
    grid on;
    title(sprintf('Generated data with %d samples',D.(dTypes{ind}).N));
    legend 'show';
end
%%%%%%%%%%%%%%%%%%%%%%%%%
figure
for I = 1 : experiments
% True mu and Sigma values for 4 component GMM
mu_true ( : , 1 ) = [ 7 ; 0 ] ; mu_true ( : , 2 ) = [ 6 ; 6 ] ;
mu_true ( : , 3 ) = [ 0 ; 0 ] ; mu_true ( : , 4 ) = [ -1 ; 7 ] ;
Sigma_true ( : , : , 1 ) = [ 5 1 ; 1 4 ] ; Sigma_true ( : , : , 2 ) = [ 3 1 ; 1 3 ] ;
Sigma_true ( : , : , 3 ) = [ 5 1 ; 1 3 ] ; Sigma_true ( : , : , 4 ) = [ 4 -2; -2 3 ];
alpha_true = [ 0.2 0.23 0.27 0.3 ] ;
% Generate Gaus s ians with N samples and run cross-validation
for i = 1 : length(N)
x = generate_samples(n , N( i ) , mu_true , Sigma_true , cumsum( alpha_true ) );
% Stor e GMM with highest performance for each iteration
GMM_pick = cross_val(x) ;
num_GMM_picks ( i ,GMM_pick) = num_GMM_picks ( i ,GMM_pick)+1;
end

% Plot frequency of model selection
bar(num_GMM_picks' ) ;
legend ( '10 Training Samples ' , ' 100 Training Samples ' , '1000 Training Samples','10000 Training Samples' ) ;
title ( 'GMM Model Order Selection ' ) ;
xlabel ( 'GMM Model Order ' ) ; ylabel ( 'Frequency of Selection' ) ;
end
%---------------------------------

% After the experiment loop
for i = 1:length(N)
    % Calculate the frequency of each model being picked
    freq_GMM_picks = num_GMM_picks(i, :) / experiments;
    num_GMM_picks_summary(i, :) = freq_GMM_picks;
end

% Display results in a table
T = array2table(num_GMM_picks_summary, ...
    'VariableNames', {'Model1', 'Model2', 'Model3', 'Model4', 'Model5', 'Model6', 'Model7', 'Model8', 'Model9', 'Model10'}, ...
    'RowNames', {'N=10', 'N=100', 'N=1000'});
disp(T);

% Plotting the results
figure;
for i = 1:length(N)
    subplot(1, length(N), i);
    bar(num_GMM_picks_summary(i, :));
    title(['GMM Model Selection for N=' num2str(N(i))]);
    xlabel('Model Order');
    ylabel('Frequency of Selection');
end
%---------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%
% Question Functions
function x = generate_samples(n , N, mu, Sigma , p_cumulative )
% Draws N samples from each class pdf to create GMM
x = zeros(n ,N) ;
for i = 1 :N
% Generate random probability
num = rand( 1 , 1 ) ;
% As sign point to 1 o f 4 Gaussians based on probability
if (num > p_cumulative( 1 ) ) == 0
x ( : , i ) = mvnrnd(mu( : , 1 ) , Sigma ( : , : , 1 ) , 1 )' ;
elseif (num > p_cumulative( 2 ) ) == 0
x ( : , i ) = mvnrnd(mu( : , 2 ) , Sigma ( : , : , 2 ) , 1 )' ;
elseif (num > p_cumulative( 3 ) ) == 0
x ( : , i ) = mvnrnd(mu( : , 3 ) , Sigma ( : , : , 3 ) , 1 )' ;
else
x ( : , i ) = mvnrnd(mu( : , 4 ) , Sigma ( : , : , 4 ) , 1 )' ;
end
end
end


function best_GMM = cross_val(x)
% Performs EM algorithm to estimate parameter s and evaluete performance
% on each data set B times , with 1 through M GMM models considered
B = 10 ; M = 10 ; % repetitions per data set ; max GMM considered
perf_array= zeros(B,M) ; % save space for per formance evaluation

% Test each data set 10 times
for b = 1 :B
% Pick random data points to fill training and validation set and
% add noise
set_size = 500;
train_index = randi( [ 1 , length( x ) ] , [ 1 , set_size ] ) ;
train_set = x( : , train_index ) + (1e-3)*randn ( 2 , set_size ) ;
val_index = randi( [ 1 , length( x ) ] , [ 1 , set_size ] ) ;
val_set = x( : , val_index ) + (1e-3)*randn( 2 , set_size ) ;
for m = 1 :M
% Non-Built-In : run EM algorith to estimate parameters
%[ alpha ,mu, sigma ] = EMforGMM(m, trainset , setsize , valset ) ;
% Built-In function : run EM algorithm to estimate parameters
GMModel = fitgmdist( train_set' ,M, 'RegularizationValue' ,1e-10) ;
alpha = GMModel.ComponentProportion ;
mu = (GMModel .mu)' ;
sigma = GMModel.Sigma ;
% Calculate log-likelihood per formance with new parameters
perf_array(b ,m) = sum(log(evalGMM( val_set , alpha ,mu, sigma ) ) ) ;
end
end

% Calculate average performance for each M and find be s t f i t
avg_perf = sum( perf_array )/B;
best_GMM = find(avg_perf == max(avg_perf ) , 1) ;
end

function [alpha_est ,mu,Sigma]=EMforGMM (M, x , N, val_set )
% Uses EM algorithm to estimate the parameters of a GMM 
% number of components based on pre-existing training data 
delta = 0.04 ; % tolerance for EM stopping criterion
reg_weight = 1e-2; % regularization parameter for covariance estimates
d = size(x , 1 ) ; % dimensionality of data

% Start with equal alpha estimates
alpha_est = ones(1 ,M) /M;

% Set initial mu as random M value pairs from data array
shuffledIndices = randperm(N) ;
mu = x( : , shuffledIndices (1 :M) ) ;

% Assign each sample to the nearest mean
[ ~ , assignedCentroidLabels ] = min( pdist2 (mu' , x' ) , [ ],1 ) ;
% Use sample covariances of initial as signments as initial covariance estimates
for m = 1 :M
Sigma (:,:,m) = cov(x(:, find(assignedCentroidLabels==m) )' ) + reg_weight*eye(d , d) ;
end

% Run EM algorithuntilit converges
t = 0 ;
Converged = 0 ;
while ~Converged
% Calculate GMM distribution according to parameters
for l = 1 :M
temp ( l , : ) = repmat( alpha_est ( l ) ,1 ,N).* evalGaussian (x ,mu( : , l ) , Sigma( : , : , l ) ) ;
end
pl_given_x = temp./sum( temp , 1 ) ;

% Calculate new alpha values
alpha_new = mean( pl_given_x , 2 ) ;

% Calculate new mu values
w = pl_given_x ./ repmat( sum( pl_given_x , 2 ) ,1 ,N) ;
mu_new = x*w' ;

% Calculate new Sigma values
for l = 1 :M
v = x-repmat (mu_new ( : , l ) ,1 ,N) ;
u = repmat(w( l , : ) ,d , 1 ).* v ;
Sigma_new ( : , : , l ) = u*v' + reg_weight*eye (d , d); % adding a small regularization term
end

% Change in each parameter
Dalpha = sum(abs( alpha_new-alpha_est' ) ) ;
Dmu = sum( sum( abs (mu_new-mu) ) ) ;
DSigma = sum( sum( abs ( abs ( Sigma_new-Sigma ) ) ) ) ;
% Check if converged
Converged = ( ( Dalpha+Dmu+DSigma )<delta ) ;
% Update old parameters
alpha_est = alpha_new ; mu = mu_new; Sigma = Sigma_new ;
%log_lik = sum( log (evalGMM( val_set , alpha_est ,mu, Sigma ) ) )
%Converged = ( log_lik <-2.3) ;
t = t+1;
end
end

function g = evalGaussian (x ,mu, Sigma )
% Evaluates the Gaussian pdf N(mu, Sigma ) at each coumn of X
[ n ,N] = size ( x ) ;
invSigma = inv ( Sigma ) ;
C = (2*pi )^(-n/2) *det ( invSigma )^(1/2) ;
E = -0.5*sum( ( x-repmat (mu, 1 ,N) ).*( invSigma *(x-repmat (mu, 1 ,N) ) ) , 1 ) ;
g = C*exp (E) ;
end
function gmm = evalGMM(x , alpha ,mu, Sigma )
% Evaluat e s GMM on the g r id based on parameter va lue s given
gmm = zeros ( 1 , size (x , 2 ) ) ;
for m = 1 : length ( alpha )
gmm = gmm + alpha(m)*evalGaussian(x ,mu( : ,m) , Sigma ( : , : ,m) ) ;
end
end

function [x,labels] = generateDataFromGMM(N,gmmParameters)
% Generates N vector samples from the specified mixture of Gaussians
% Returns samples and their component labels
% Data dimensionality is determined by the size of mu/Sigma parameters
priors = gmmParameters.priors; % priors should be a row vector
meanVectors = gmmParameters.meanVectors;
covMatrices = gmmParameters.covMatrices;
n = size(gmmParameters.meanVectors,1); % Data dimensionality
C = length(priors); % Number of components
x = zeros(n,N); labels = zeros(1,N);
% Decide randomly which samples will come from each component
u = rand(1,N); thresholds = [cumsum(priors),1];
for l = 1:C
indl = find(u <= thresholds(l)); Nl = length(indl);
labels(1,indl) = l*ones(1,Nl);
u(1,indl) = 1.1*ones(1,Nl); % these samples should not be used again
x(:,indl) = mvnrnd(meanVectors(:,l),covMatrices(:,:,l),Nl)';
end
end
function displayProgress(t,x,alpha,mu,Sigma)
figure(1),
if size(x,1)==2
subplot(1,2,1), cla,
plot(x(1,:),x(2,:),'b.');
xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours'),
axis equal, hold on;
rangex1 = [min(x(1,:)),max(x(1,:))];
rangex2 = [min(x(2,:)),max(x(2,:))];
[x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
contour(x1Grid,x2Grid,zGMM); axis equal,
subplot(1,2,2),
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%