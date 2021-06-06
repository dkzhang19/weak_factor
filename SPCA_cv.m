function res = SPCA_cv(param)
% This function performs supervised PCA estimates of risk premium

% In this version, we do SPCA for all p<=pmax. In each step, we select only
% N0 test assets with the largest correlation. Thus, we get d by p risk
% premia estimates.

%% INPUT

% Gammahat_nozero    is d by pmax matrix of risk premia estimates
% b                  is pmax by N matrix of SDF loading
% mimi               is d by N by pmax weights matrix for mimicking portfolio
% alphahat           is d by pmax vector of pricing errors

%% OUTPUT

% Gammahat_nozero    is d by pmax matrix of risk premia estimates
% b                  is pmax by N vector of SDF loading



%% INITIALIZATION

rt = param.rt;
gt = param.gt;
pmax = param.pmax;
N0 = param.tuning;

T  =  size(rt,2);
n  =  size(rt,1);
d  =  size(gt,1);

%% ESTIMATION
rtbar               =      rt - repmat(mean(rt,2),1,T);% 
gtbar               =      gt - repmat(mean(gt,2),1,T);% 
etahat              =      [];% is estimated eta
gammahat            =      [];% is estimated gamma
Index               =      [];% is the subset we choose at each step
k                   =      0;% is # of steps
vhat                =      [];
B = [];
bhat = zeros(pmax,n);
mrt                 =      mean(rt,2);
Gammahat_nozero     =      nan(d,pmax);
mimi                =      nan(d,n,pmax);

while(k<pmax)
    COR = abs(corr(rtbar',gtbar'));
    L = max(COR,[],2);
    [bb,i] = sort(L);
    if N0<n
        II = (L >= bb(n-N0));
    else
        II = L>-1;
    end
    k = k + 1;
    B = [B;zeros(1,n)];
    Index(:,k) = 0;
    Index(II,k) = 1;
    Top(:,k) = find(II);
    COR2 = corr(rtbar',gtbar');
    TopCorr(:,k) = COR2(II);
    
% perform PCA
    [U,S,V]             =      svds(rtbar(II,:),1); 
    B(k,II) = U(:,1)'/S(1,1);
    gammahat            =      [gammahat; U(:,1)' * mrt(II)/S(1,1)];
    etahat              =      [etahat,gtbar  *  V(:,1) ];
% projection  
    gtbar      =       gtbar   -  gtbar * V(:,1) * V(:,1)'; %disp(norm(gtbar,2));
    mrt        =       mrt    - rtbar * V(:,1) * gammahat(k);
    rtbar      =       rtbar   -  rtbar * V(:,1) * V(:,1)'; 
    vhat(k,:)  =       V(:,1)';
% Estimation of risk premium
    Gammahat_nozero(:,k) = etahat * gammahat;
end

for k = 1:pmax
    B_sub = B(1:k,:);
    fhat =  B_sub*rt;
    fhatbar = fhat - mean(fhat,2);
    mimi(:,:,k) = (gt - mean(gt,2))*fhatbar'*(fhatbar*fhatbar')^(-1)*B_sub;
    bhat(k,:) = mean(fhat,2)'*(fhatbar*fhatbar'/T)^(-1)*B_sub;
end
%%  Results
res.Gammahat_nozero = Gammahat_nozero;
res.b = bhat;
res.mimi = mimi;
res.alphahat = repmat(mean(gt,2),1,pmax) - Gammahat_nozero;
res.Top=Top;
res.TopCorr=TopCorr;

