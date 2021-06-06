function res=kfoldcv_tsr2(M,K,func1,param,tuningrange)
% This function is used to select the best tuning parameters based on time
% series R2. 

%% Input
% M             repeat K-fold CV M times
% K             K-fold
% func1         risk permium estimator fuction
% param         parameters for func1
% tuningrange

if ~isfield(param,'pmax')
   param.pmax=1;
end

gt = param.gt;
rt = param.rt;

T = size(rt,2);
tsr2 = zeros(K,M,length(tuningrange),param.pmax);

np = param.pmax;
tr = length(tuningrange);
for m = 1:M
   indices = crossvalind('Kfold',T,K);
   for i=1:K
       test =(indices==i);
       train = ~test;
       rt_test = rt(:,test);
       rt_train = rt(:,train);
       muhat = mean(rt_test,2);
       rhatbar = rt_test-muhat;
       gt_test = gt(:,test);
       ghatbar = gt_test -mean(gt_test,2);
       for jj = 1:tr
            prm = param;
            prm.rt = rt_train;
            prm.gt = gt(:,train);
            prm.tuning = tuningrange(jj);
            res = func1(prm);
            mimi = res.mimi;
            for p = 1:np
                tsr2(i,m,jj,p) = mean(1-sum((ghatbar'-rhatbar'*mimi(:,:,p)').^2)./sum((ghatbar').^2));
            end
       end
   end
end

re.tsr2 = reshape(mean(tsr2,[1,2]),[length(tuningrange),param.pmax]);
[i1,i2] = find(re.tsr2==max(re.tsr2,[],'all'));
re.tuning = tuningrange(i1(1));
re.pmax = i2(1);

prm = param;
prm.pmax = re.pmax;
prm.tuning = re.tuning;
res = func1(prm);
res.pmax = prm.pmax;
res.tuning = prm.tuning;
res.tsr2 = re.tsr2;

