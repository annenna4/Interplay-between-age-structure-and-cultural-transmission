% performes Ewens-Watterson test based on populations generated in main_ageSim

clear all
close all

nPop = 10^5; % population size
pMut = 5*10^-4; % innovation rate (per transmission event)
pDeath = 0.1; % death rate
pTrans = 0.01; % probability that an individual engages in CT per time step
transMode = 0; % transmission mode: 0 - no transmission, 1 - horizontal, 2 - oblique, 3 - age-neutral
b = 0; % strength of frequency-dependent transmission, b>0 conformity, b<0 anti-conformity, b=0 unbiased transmission

copyAll = 0; % if copyAll = 1 then copying happens from all age groups
copyThresholdHigh = 6; % upper bound of the age of the copying pool
copyThresholdLow = 0; % lower bound of the age of the copying pool

nSamV = [50, 100, 200, 500, 1000, 2000]; % sample size
sampleMax = 10; % number of samples drawn from each population
set = 1;

for transMode = 0:0
    
    % loading populations (nSim = 100) at a single point in time generated using the
    % specified paramter constellation 
    if copyAll == 0
        name = sprintf('./data/populations/popAge_N%02d_pMut%02d_pDeath%02d_b%02d_thLow%01d_thHigh%01d_pTrans%02d_transMode%01d_set%01d.mat',nPop,pMut,pDeath,b,copyThresholdLow,copyThresholdHigh,pTrans,transMode,set);
        if exist(name)==0 
            name = sprintf('./data/populations/popAge_N%02d_pMut%02d_pDeath%02d_thLow%01d_thHigh%01d_pTrans%02d_transMode%01d_set1.mat',nPop,pMut,pDeath,copyThresholdLow,copyThresholdHigh,pTrans,transMode);
        end
        pop = load(name);
        pop = pop.popTotal;
    else
        name = sprintf('./data/populations/popAge_N%02d_pMut%02d_pDeath%02d_b%02d_pTrans%02d_transMode%01d_set1_ALL.mat',nPop,pMut,pDeath,b,pTrans,transMode);
        if exist(name) == 0
            name = sprintf('./data/populations/popAge_N%02d_pMut%02d_pDeath%02d_pTrans%02d_transMode%01d_set1_ALL.mat',nPop,pMut,pDeath,pTrans,transMode);
        end
        pop = load(name);
        pop = pop.popTotal;
    end

%     name = sprintf('./data/populations/popWF_N%02d_pMut%02d.mat',nPop,pMut);
%     pop = load(name);
%     pop = pop.popTotal;

    EcountV=zeros(length(nSamV),sampleMax*size(pop,1)/2);
    FcountV=zeros(length(nSamV),sampleMax*size(pop,1)/2);
    
    for sim = 1:size(pop,1)/2
        
        sim
        
        h = pop(2*sim-1,:);
        %h = pop(sim,:);
        
        for j = 1:length(nSamV)
            nSam = nSamV(j);
            for i = 1:sampleMax
                sam = randsample(h,nSam,'false'); % drawing sample from population 
                typeSam = unique(sam); 
                k = length(typeSam); % number of different variants in sample  
                hh = hist(sam,typeSam); % absolute frequencies of variants in sample
                hhrel = hh./nSam;
                simpson0 = sum(hhrel.^2); % empirical diversity level
                prob0 = 1/prod(hh); % not normalised probability of sample under neutrality 
                [Ecount,Fcount] = get_WFpredictions(nSam,k,10^6,simpson0,prob0); % number of neutral samples (out of 10^6) that have a probability less than the observed sample 
                EcountV(j,(sim-1)*sampleMax+i) = Ecount;
                FcountV(j,(sim-1)*sampleMax+i) = Fcount;
            end
        end
        
    end
    
    if copyAll == 0
        name = sprintf('./data/CompToNeutrality/EcountB_N%02d_pMut%02d_pDeath%02d_b%02d_thLow%01d_thHigh%01d_pTrans%02d_transMode%01d.mat',nPop,pMut,pDeath,b,copyThresholdLow,copyThresholdHigh,pTrans,transMode);
        save(name,'EcountV');
        name = sprintf('./data/CompToNeutrality/FcountB_N%02d_pMut%02d_pDeath%02d_b%02d_thLow%01d_thHigh%01d_pTrans%02d_transMode%01d.mat',nPop,pMut,pDeath,b,copyThresholdLow,copyThresholdHigh,pTrans,transMode);
        save(name,'FcountV');
    else
        name = sprintf('./data/CompToNeutrality/Ecount_N%02d_pMut%02d_pDeath%02d_b%02d_pTrans%02d_transMode%01d_ALL.mat',nPop,pMut,pDeath,b,pTrans,transMode);
        save(name,'EcountV');
        name = sprintf('./data/CompToNeutrality/Fcount_N%02d_pMut%02d_pDeath%02d_b%02d_pTrans%02d_transMode%01d_ALL.mat',nPop,pMut,pDeath,b,pTrans,transMode);
        save(name,'FcountV');
    end

%         name = sprintf('./data/CompToNeutrality/Ecount_N%02d_pMut%02d_WF.mat',nPop,pMut);
%         save(name,'EcountV');
%         name = sprintf('./data/CompToNeutrality/Fcount_N%02d_pMut%02d_WF.mat',nPop,pMut);
%         save(name,'FcountV');

end

