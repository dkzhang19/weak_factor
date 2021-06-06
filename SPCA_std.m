function res = SPCA_std(param)
% This function performs supervised PCA estimates of risk premium

% In this version, we do SPCA for p = pmax. Thus, we get d by 1 risk
% premia estimates.

%% INPUT

% rt          is n by T matrix
% gt          is d by T factor proxies
% pmax        is # of factors
% tuning      is # of test assets we use at each step

%% OUTPUT

% Gammahat_nozero    is d by 1 vector of risk premia estimates
% b                  is 1 by N vector of SDF loading
% mimi               is d by N weights matrix for mimicking portfolio
% avarhat_nozero     is d by 1 vector of varaince estimator for rp
% alphahat           is d by 1 vector of pricing errors
% avarhat_alpha      is d by 1 vector of varaince estimator for alpha

%% INITIALIZATION

rt = param.rt;
gt = param.gt;
pmax = param.pmax;
N0 = param.tuning;
q = param.q;

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
mrt                 =      mean(rt,2);
    
while(k<pmax)

    COR   = abs(corr(rtbar',gtbar'));

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

% perform PCA

    [U,S,V]             =      svds(rtbar(II,:),1); 

    B(k,II) = U(:,1)'/S(1,1);
    gammahat            =      [gammahat; U(:,1)' * mrt(II)/S(1,1)];
    etahat              =      [etahat,gtbar  *  V(:,1) ];
    
% projection  
    gtbar      =       gtbar   -  gtbar * V(:,1) * V(:,1)'; 
    mrt        =       mrt    - rtbar * V(:,1) * gammahat(k);
    rtbar      =       rtbar   -  rtbar * V(:,1) * V(:,1)'; 
    vhat(k,:)  =       V(:,1)';
end

Sigmavhat             =       vhat*vhat'/T;
what                  =       gtbar;
phat                  =       k;

% Newy-West Estimation
Pi11hat               =        zeros(d*phat,d*phat);
Pi12hat               =        zeros(d*phat,phat);
Pi22hat               =        zeros(phat,phat);

Pi13hat               =        zeros(d*phat,d);
Pi33hat               =        zeros(d,d);
for t = 1:T

    Pi11hat           =        Pi11hat  +  vec(what(:,t) * vhat(:,t)')*vec(what(:,t) * vhat(:,t)')'/T;
    Pi12hat           =        Pi12hat  +  vec(what(:,t) * vhat(:,t)')*vhat(:,t)'/T;
    Pi22hat           =        Pi22hat  +  vhat(:,t)     * vhat(:,t)'/T;

    Pi13hat           =        Pi13hat  +  vec(what(:,t) * vhat(:,t)')*what(:,t)'/T;
    Pi33hat           =        Pi33hat  +  what(:,t)     * what(:,t)'/T;

    for s = 1:min(t-1,q) 

        Pi11hat       =        Pi11hat  + 1/T*(1-s/(q+1))* (vec(what(:,t) * vhat(:,t)')*vec(what(:,t-s) * vhat(:,t-s)')'+vec(what(:,t-s) * vhat(:,t-s)')*vec(what(:,t) * vhat(:,t)')');
        Pi12hat       =        Pi12hat  + 1/T*(1-s/(q+1))* (vec(what(:,t) * vhat(:,t)')*vhat(:,t-s)' + vec(what(:,t-s) * vhat(:,t-s)')*vhat(:,t)');
        Pi22hat       =        Pi22hat  + 1/T*(1-s/(q+1))* (vhat(:,t)     * vhat(:,t-s)' + vhat(:,t-s) * vhat(:,t)' );

        Pi13hat       =        Pi13hat  + 1/T*(1-s/(q+1))* (vec(what(:,t) * vhat(:,t)')*what(:,t-s)' + vec(what(:,t-s) * vhat(:,t-s)')*what(:,t)');
        Pi33hat       =        Pi33hat  + 1/T*(1-s/(q+1))* (what(:,t)     * what(:,t-s)' + what(:,t-s) * what(:,t)' );

    end        
end
avarhat_nozero = diag(kron(gammahat'*inv(Sigmavhat),eye(d))*Pi11hat*kron(inv(Sigmavhat)*gammahat,eye(d))/T + ...
                 kron(gammahat'*inv(Sigmavhat),eye(d))*Pi12hat*etahat'/T + (kron(gammahat'*inv(Sigmavhat),eye(d))*Pi12hat*etahat')'/T + ...
                 etahat*Pi22hat*etahat'/T);
% risk premia estimator
Gammahat_nozero = etahat * gammahat;
fhat =  B*rt;
fhatbar = fhat - mean(fhat,2);
mimi = (gt - mean(gt,2))*fhatbar'*(fhatbar*fhatbar')^(-1)*B;
bhat = mean(fhat,2)'*(fhatbar*fhatbar'/T)^(-1)*B;

% estimate alpha = E(g) - eta*gamma
alphahat = mean(gt,2) - Gammahat_nozero;
avarhat_alpha = diag(kron(gammahat'*inv(Sigmavhat),eye(d))*Pi11hat*kron(inv(Sigmavhat)*gammahat,eye(d))/T - ...
                 kron(gammahat'*inv(Sigmavhat),eye(d))*Pi13hat/T - (kron(gammahat'*inv(Sigmavhat),eye(d))*Pi13hat)'/T + ...
                 Pi33hat/T);
%% Results
res.Gammahat_nozero = Gammahat_nozero;
res.b = bhat;
res.mimi = mimi;
res.avarhat_nozero = avarhat_nozero;
res.avarhat_alpha = avarhat_alpha;
res.alphahat = alphahat;
