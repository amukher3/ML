function [alpha_mle,tau_mle,tauOne_mle,tauTwo_mle,OnsetSample,SclTimeDomain,NumOfWindows,SamplingFreqn,locPeakDetection] = EstimationProblem(user,DataIndex,Delta,T);
[Window,OnsetSample,SclTimeDomain,WindowSize,SamplingFreqn,locPeakDetection] = MakingWindowsAroundPeaks(user,DataIndex,Delta,T);
A = [0,0,-1,1]; %for inequality constraint : 0*Alpha+0*Tau-TauOne+TauTwo<=k
b = (-.1); 
lb= [0.01,0.01,1,.1]; % lower bound of the estimates:[alpha tau TauOne TauTwo]
ub= [10,T,10,5]; %upper bound of the estimates
%lb= [0.01,0.01,1,.01]; % For Simulated set
%ub= [100,T,100,5]; 
Aeq = []; %for equality constraint
beq = []; %for equality constrain
iter=15;
NumOfParameters=4;
NumOfWindows = size(Window,2);
x=zeros(iter,NumOfParameters);
x0(1,:) = [lb(1),1,2,.75]; % initial estiamtes of the parameters
x0(2,:) = [(ub(1)-lb(1))/3,T/3,5,1];
x0(3,:) = [(ub(1)-lb(1))/2,T/2,6,2]; 
x0(4,:) = [(ub(1)-lb(1))*3/4,T*3/4,4,2]; 
x0(5,:) = [ub(1),T,4,5]; 
x0(6,:) = [(ub(1)-lb(1))*5/6,T*5/6,3,2]; 
x0(7,:) = [(ub(1))*2/3,T*2/3,2,1]; 
x0(8,:) = [(ub(1))/4,T/4,4,3]; 
x0(9,:) = [(ub(1))/3,T/2,6,4]; 
x0(10,:) = [(ub(1))*rand,T/2,2,.5]; 
x0(11,:) = [(ub(1))*rand,T*rand,3,2]; % initial estiamtes of the parameters   
x0(12,:) = [(ub(1))*rand,1,6,3];
x0(13,:) = [(ub(1))*rand,2,8,6];
x0(14,:) = [(lb(1)),3,10,7];
x0(15,:) = [(ub(1)),4,5,2];
alpha_mle = zeros(1,NumOfWindows);
tau_mle = zeros(1,NumOfWindows);
tauOne_mle = zeros(1,NumOfWindows);
tauTwo_mle = zeros(1,NumOfWindows);
for runs=1:NumOfWindows  % Num of Windows in a user
  samples(:,runs) = Window(:,runs);
for i=1:iter % No. of independent initialisations 
[x(i,:),val(runs,i),flag(i)] = fmincon(@(t_prime)MaxAposterioriProb(t_prime,...
    samples(:,runs),T),x0(i,:),A,b,Aeq,beq,lb,ub);
end
[~,I] = min(val(runs,:));  % checking for the minm. value of the function.
mle_estimates(runs,:)= x(I,:);  % getting the estimate values(alpha,tau,tauOne,tauTwo) for the corresponding minm. value of the fit.
alpha_mle(runs)= mle_estimates(runs,1);
tau_mle(runs) = mle_estimates(runs,2);
tauOne_mle(runs)= mle_estimates(runs,3);
tauTwo_mle(runs)=mle_estimates(runs,4);
end