function [pop] = get_burnIn(pDeath,nPop,pMut,pTrans,transMode,copyAll,copyThresholdHigh,copyThresholdLow,b)
% Burn in rule: we assume equilibrium has been reached when all initially present variant types have gone extinct, ie. when all variant types have undergone neutral 
% dynamic. This is a relatively strict rule that potentially takes a large amount of time. 

pop = zeros(2,nPop);
h = ones(1,nPop);
initialType = 2; % number of types initially present
for i=1:initialType
    h((i-1)*nPop/initialType+1:i*nPop/initialType) = i;
end
pop(1,:) = h; % initial types
value = initialType;
pop(2,1:nPop) = ceil(rand(1,nPop)*50); % initial birth dates
t = max(pop(2,:));

while  min(pop(1,:))<initialType+1 % until all types have undergone neutral dynamics
    t = t+1;
    [pop,value] = get_dynamics(t,pop,value,pDeath,nPop,pMut,b,pTrans,transMode,copyAll,copyThresholdHigh,copyThresholdLow,0,0,0,0,0,1,1);
end

% ALTERNATIVE BURN IN RULE: here it is assumed that that equilibrium state has been reached if two the heterogeneity index of two populations, both initialised with opposing initial conditions 
% (one with maximum heterogeneity, one with minimum heterogeneity), have crossed. 

    % %initialisation of population 1
    % pop1 = zeros(2,nPop);
    % h = ones(1,nPop);
    % initialType = 2; %number of types initially present
    % for i=1:initialType
    %     h((i-1)*nPop/initialType+1:i*nPop/initialType) = i;
    % end
    % pop1(1,:) = h; %initial types
    % value1 = initialType;
    % pop1(2,1:nPop) = ceil(rand(1,nPop)*50); %initial birth dates
    % t1 = max(pop1(2,:));
    % %initialisation of population 2
    % pop2 = zeros(2,nPop);
    % h = ones(1,nPop);
    % initialType = nPop/10; %number of types initially present
    % for i=1:initialType
    %     h((i-1)*nPop/initialType+1:i*nPop/initialType) = i;
    % end
    % pop2(1,:) = h; %initial types
    % value2 = initialType;
    % pop2(2,1:nPop) = ceil(rand(1,nPop)*50); %initial birth dates
    % t2 = max(pop2(2,:));
    % if t1~=t2
    %     pop2(2,ceil(rand*nPop))=t1;
    % end
    % t = t1;
    
    % diffPop1Pop2 = 5;
    % while diffPop1Pop2>0
    %     t = t+1;
    %     [pop1,value1,] = get_dynamics(t,pop1,value1,pDeath,nPop,pMut,pTrans,transMode,copyAll,copyThresholdHigh,copyThresholdLow,0,0,0,0,0);
    %     type = unique(pop1(1,:));
    %     h = hist(pop1(1,:),type)./nPop;
    %     divPop1 = sum(h.^2);
    %     [pop2,value2,] = get_dynamics(t,pop2,value2,pDeath,nPop,pMut,pTrans,transMode,copyAll,copyThresholdHigh,copyThresholdLow,0,0,0,0,0);
    %     type = unique(pop2(1,:));
    %     h = hist(pop2(1,:),type)./nPop;
    %     divPop2 = sum(h.^2);
    %     diffPop1Pop2 = divPop1-divPop2;
    % end
    
    % for i = t+1:t+500
    %    [pop1,value1,] = get_dynamics(i,pop1,value1,pDeath,nPop,pMut,pTrans,transMode,copyAll,copyThresholdHigh,copyThresholdLow,0,0,0,0,0);
    %    [pop2,value2,] = get_dynamics(i,pop2,value2,pDeath,nPop,pMut,pTrans,transMode,copyAll,copyThresholdHigh,copyThresholdLow,0,0,0,0,0);
    % end
