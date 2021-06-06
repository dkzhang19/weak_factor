%%
clear;
close all;
warning('off');

%% Calibration 
load('Calibrated_factors.mat');
%%
l = 1; % task ID
%l = str2num(getenv('SLURM_ARRAY_TASK_ID'));

M = 1000;  % # of MCs
T = 120;   % # of time periods
n = 2000;   % # of stocks
p = 4;     % # of latent factors
d = 4;     % # of factor proxies
pmax = 6;     % the largest # of factors
q = 2;      % is # of lags used in Newy-West standard errors

a = 0.5;   % parameter for the strength of the weak factor

Sigmav = [Sigmav(1:3,1:3),zeros(3,1);zeros(1,3),5];
beta0 = [1, zeros(1,p-1)];
Sigmau = 12*eye(n);
Sigmaw = 0*eye(d);
gamma = [gamma(1:3),0.4]';
xi = zeros(d,1);

rng(123); % Control the random seed
beta = repmat(beta0,n,1) + randn(n,p);
I = (1:n)>a*n;
e1 = randn(n,1);
e1(I,:) = e1(I,:)*0.1;    
% beta(:,4) = -beta(:,3) + e1; % Correlated factor loadings
beta(:,4) = e1;   % Weak factor loadings

eta = eye(4);
% eta = [1 0 0 0; 0 1 0 0; 0 0 0 1]; 

% True parameters
Sigma = beta*Sigmav*beta'+Sigmau;
mu = beta*gamma;
sdf_loading = Sigma\mu;

% Initialization

% Risk premium estimator
Gammatrue = eta*gamma;  

Gammahat_SPCA = zeros(d,M);
Gammahat_PCA = zeros(d,M);
Gammahat_PLS = zeros(d,M);
Gammahat_rpPCA = zeros(d,M);
Gammahat_Lasso = zeros(d,M);
Gammahat_Ridge = zeros(d,M);
Gammahat_four = zeros(d,M);
Gammahat_Fama = zeros(d,M);

% Estimated # of factors
phat = zeros(1,M);

% Tuning parameters
tuningrange_SPCA = 100:100:500; % tuning range for SPCA (N0)
v_r = 10.^(2:0.2:4); % tuning range for Ridge
v_l = 10.^(-0.4:0.2:0.8); % tuning range for Lasso
mu_rp = -1:10; % tuning range for rpPCA

%% MC
tic
for iMC = 1:M
    rng((l-1)*M+iMC);
    disp(iMC);
    
    vt = Sigmav^0.5*randn(p,T);          % factor innovations
    ut = Sigmau^0.5*randn(n,T);          % residual innovations
    wt = Sigmaw^0.5*randn(d,T);          % proxies residual innovations
    rt = repmat(beta*gamma,1,T) + beta*vt + ut; % returns
    gt = repmat(xi,1,T) + eta*vt + wt; % proxies
    
    
    %% Estimation
    % SPCA
    param_spca.pmax = pmax; param_spca.rt = rt; param_spca.gt = gt;
    
    SPCAres = kfoldcv_tsr2(3,3,@SPCA_cv,param_spca,tuningrange_SPCA);
    
    Gammahat_SPCA(:,iMC) = SPCAres.Gammahat_nozero(:,end); % risk premia estimator
    phat(:,iMC) = SPCAres.pmax; % estimated # of factors
    
    %% PCA
    PCAres = PCA_std(rt,gt,p,q);
    Gammahat_PCA(:,iMC) = PCAres.Gammahat_nozero; % risk premia estimator
    
  %% PLS
    PLSres = PLS_sdf(rt,gt,p);
    Gammahat_PLS(:,iMC) = PLSres.Gammahat_nozero;  % risk premia estimator
  
    %% rpPCA
    param_rp.pmax = p; param_rp.rt = rt; param_rp.gt = gt;
    sr_rpPCA_temp = zeros(length(mu_rp),1);
    
    for jjj = 1:length(mu_rp)
        param_rp.tuning = mu_rp(jjj);
        rpPCAres = rpPCA_cv(param_rp); 
        b_rpPCA = rpPCAres.b(end,:);
        sr_rpPCA_temp(jjj,1) = ((b_rpPCA*mu)'*(b_rpPCA*Sigma*b_rpPCA')^(-1)*(b_rpPCA*mu))^0.5; % calculate the SR for each tuning parameter
    end
    
    [sr_max,ind] = max(sr_rpPCA_temp); % Select the one with the largest OOS SR
    
    param_rp.tuning = mu_rp(ind);
    rpPCAres = rpPCA_cv(param_rp);
    Gammahat_rpPCA(:,iMC)   = rpPCAres.Gammahat_nozero(:,end); % risk premia estimator
    
    %% Ridge
    
    Ridgeres = Ridge_rp(rt,v_r);
    Best_r = Best_likelihood(rt,Ridgeres.b,Sigma,mu); % select the tuning with the largest likelihood (given true Sigma and mu)
    
    Gammahat_Ridge(:,iMC) = risk_premium(gt,Best_r.sdf_op);  % risk premia estimator
    
    %% Lasso    

    Lassores          = Lasso_rp(rt,v_l);
    Best_l = Best_likelihood(rt,Lassores.b,Sigma,mu); % select the tuning with the largest likelihood (given true Sigma and mu)
    
    Gammahat_Lasso(:,iMC)   = risk_premium(gt,Best_l.sdf_op); % risk premia estimator
    
    %% Four-split
    four_split_res = four_split(rt,gt,1,[1,0,0,0]);
    Gammahat_four(:,iMC) =  four_split_res.Gammahat;  % risk premia estimator
    
    %% Two-pass
    FMres = FM(rt,gt);
    Gammahat_Fama(:,iMC) = FMres.Gammahat_nozero;  % risk premia estimator
    
end
toc
%% Risk premium estimator for V
k = 4;
gfama = Gammahat_Fama(k,:);
gspca = Gammahat_SPCA(k,:);
gpca = Gammahat_PCA(k,:);
gfour = Gammahat_four(k,:);
grppca = Gammahat_rpPCA(k,:);
gpls = Gammahat_PLS(k,:);
glasso = Gammahat_Lasso(k,:);
gridge = Gammahat_Ridge(k,:);

RE = [gfama;gspca;gpca;gfour;grppca;gpls;glasso;gridge];

% save('case_a.mat','RE')
