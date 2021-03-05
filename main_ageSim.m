% Simulation of age-structured copying model + estimation of PD and its power law behaviour
% Output: cultural composition of population as .mat files, PD (and thresholded PDs) as csv file for input in R script 
%         -> following folder structure needed ./data/populations , ./data/frequencies , ./data/progeny

clear all
close all


nPop = 10^5; % population size
pMut = 5*10^-4; % mutation rate (per transmission event)
pDeath = 0.1; % per capita death rate
b = 0.0; % strength of frequency-dependent transmssion, b<0 negative bias, b>0 positive bias
transMode = 0; % transmission mode for CT during life time: 0 - no transmission, 1 - horizontal, 2 - oblique, 3 - age-neutral
pTrans = 0.01; % probability that an individual engages in CT per time step

copyAll = 0; % if copyAll = 1 then copying happens from all age groups
copyThresholdHigh = 2; % upper bound of the age of the copying pool
copyThresholdLow = 0; % lower bound of the age of the copying pool

tMax = 10000; % time steps to be run after equilibrium has been reached
itMax = 1; % number of simulations

set = 1;
figureIndex = 1;

popTotal = zeros(itMax*2,nPop);
variantAgeMode = 0; % variantAgeMode = 1 tracks life time of variant
PDmode = 1; % PDmode = 1 generates the progeny distribution 
lagBetweenSamples = 0; % lag between two consecutive samples; 0 (and 1 but to sample every time step, use 0) means every time point sampled, CAUTION: 2 means every second times step, 3 every third, etc.
sampleFrac = 1; % fraction of the progeny in each time step that are recorded
saveMode = 0; % 0: do nothing, 1: saves the cultural composition of the last population as a.mat file
freqMode = 0; % 0: do nothing, 1: records the frequencies of all cultural variant types in the interval [1,tMax] and saves it as .txt file

for sim = 1:itMax
    
    sim
    
    fprintf("Burn-in period\n")
    [pop] = get_burnIn(pDeath,nPop,pMut,pTrans,transMode,copyAll,copyThresholdHigh,copyThresholdLow,b);
    
    
        % re-name variant types for convenience
        names = unique(pop(1,:)); % unique names in pop
        for i = 1:length(names)
            index = find(pop(1,:) == names(i)); % find indeces of all instances of name names(i)
            pop(1,index) = ones(1,length(index))*i; % re-name it with name i
        end
        value = numel(names);
        valueIni = value;
        h = min(pop(2,:))-1;
        pop(2,:) = pop(2,:)-h;
        tini = max(pop(2,:));
        
        fprintf("Generating populations \n")
        ageCount = [];
        if lagBetweenSamples == 0 & sampleFrac == 1
            namesFreq = zeros(1,value);
        else 
            namesFreq = zeros(2,value);
        end
        freqTrait = zeros(tMax,value);

        for t = tini+1:tini+tMax

            % recording the frequencies of all present variant types at t
            if freqMode == 1
                freqTrait = horzcat(freqTrait, zeros(tMax,value-size(freqTrait,2)));  
                type = unique(pop(1,:));
                h = hist(pop(1,:),type)./nPop;
                freqTrait(t-tini,type) = h;
            end

            [pop,value,ageCount,namesFreq] = get_dynamics(t,pop,value,pDeath,nPop,pMut,b,pTrans,transMode,copyAll,copyThresholdHigh,copyThresholdLow,valueIni,variantAgeMode,ageCount,PDmode,namesFreq,lagBetweenSamples,sampleFrac);

        end
    
        %[fx,dx] = ecdf(ageCount(2,:));
        %figure(1)
        %plot(dx,fx); hold on;

        if PDmode == 1
            % calculation of progeny distribution
            if sampleFrac < 1 | lagBetweenSamples > 0
                namesFreq = namesFreq(1,:);
            end
            namesFreq = nonzeros(namesFreq);
            namesFreq = reshape(namesFreq,numel(namesFreq),1);
            type = unique(namesFreq);
            c1 = hist(namesFreq,type)';

            % structure of PD as needed for R function Neutral_Spec_Rate_MLE
            % --------------------------------------------------------------
            c = [[type] [(c1)]]; c(c(:,2)<10^-10,:) = [];
            cthresh2 = [[type(2:end)] [(c1(2:end))]]; c(c(:,2)<10^-10,:) = []; % singletons removed
            cthresh6 = [[type(6:end)] [(c1(6:end))]]; c(c(:,2)<10^-10,:) = []; % all variant types with less than 5 variants removed
            
            cHeader = {'"births"' '"species"'};
            textHeader = strjoin(cHeader, ',');
            if copyAll == 0
                name = sprintf('./data/progeny/PD_N%02d_pMut%02d_pDeath%02d_b%02d_thLow%01d_thHigh%01d_pTrans%02d_transMode%01d.csv',nPop,pMut,pDeath,b,copyThresholdLow,copyThresholdHigh,pTrans,transMode);
                name2 = sprintf('./data/progeny/PDthresh2_N%02d_pMut%02d_pDeath%02d_b%02d_thLow%01d_thHigh%01d_pTrans%02d_transMode%01d.csv',nPop,pMut,pDeath,b,copyThresholdLow,copyThresholdHigh,pTrans,transMode);
                name6 = sprintf('./data/progeny/PDthresh6_N%02d_pMut%02d_pDeath%02d_b%02d_thLow%01d_thHigh%01d_pTrans%02d_transMode%01d.csv',nPop,pMut,pDeath,b,copyThresholdLow,copyThresholdHigh,pTrans,transMode);
            else
                name = sprintf('./data/progeny/PD__N%02d_pMut%02d_pDeath%02d_b%02d_pTrans%02d_transMode%01d_ALL.csv',nPop,pMut,pDeath,b,pTrans,transMode);
                name2 = sprintf('./data/progeny/PDthresh2__N%02d_pMut%02d_pDeath%02d_b%02d_pTrans%02d_transMode%01d_ALL.csv',nPop,pMut,pDeath,b,pTrans,transMode);
                name6 = sprintf('./data/progeny/PDthresh6__N%02d_pMut%02d_pDeath%02d_b%02d_pTrans%02d_transMode%01d_ALL.csv',nPop,pMut,pDeath,b,pTrans,transMode);
            end
            fid = fopen(name,'w');
            fprintf(fid,'%s\n',textHeader);
            fclose(fid);
            dlmwrite(name,c,'-append');
            fid = fopen(name2,'w');
            fprintf(fid,'%s\n',textHeader);
            fclose(fid);
            dlmwrite(name2,cthresh2,'-append');
            fid = fopen(name6,'w');
            fprintf(fid,'%s\n',textHeader);
            fclose(fid);
            dlmwrite(name6,cthresh6,'-append');
            %---------------------------------------------------------------

            c = c1./length(namesFreq);
            c = [[type; type(end)+1] 1-[0; cumsum(c)]]; c(c(:,2)<10^-10,:) = [];
                        
            figure(figureIndex)
            loglog(c(:,1),c(:,2)); hold on;
            figureIndex = figureIndex+1;   
            %fprintf("Estimation of powerlaw - progeny distribution \n")
            %[a,b,c] = plfit(namesFreq) % using estimator developed by Clauset et al. 
            %coeffPL(sim,:) = [a,b];
    
        end
        
        % saving cultural composition of last population
        if saveMode == 1
            popTotal(2*sim-1:2*sim,1:nPop) = pop;
            if mod(sim,10) == 0
                if copyAll == 0
                    name = sprintf('./data/populations/popAge_N%02d_pMut%02d_pDeath%02d_b%02d_thLow%01d_thHigh%01d_pTrans%02d_transMode%01d_set%01d.mat',nPop,pMut,pDeath,b,copyThresholdLow,copyThresholdHigh,pTrans,transMode,set);
                    save(name,'popTotal');
                    name = sprintf('./data/populations/popAgePL_N%02d_pMut%02d_pDeath%02d_b%02d_thLow%01d_thHigh%01d_pTrans%02d_transMode%01d_set%01d.mat',nPop,pMut,pDeath,b,copyThresholdLow,copyThresholdHigh,pTrans,transMode,set);
                    save(name,'coeffPL');
                else
                    name = sprintf('./data/populations/popAge_N%02d_pMut%02d_pDeath%02d_b%02d_pTrans%02d_transMode%01d_set%01d_ALL.mat',nPop,pMut,pDeath,b,pTrans,transMode,set);
                    save(name,'popTotal');
                    name = sprintf('./data/populations/popAgePL_N%02d_pMut%02d_pDeath%02d_b%02d_pTrans%02d_transMode%01d_set%01d_ALL.mat',nPop,pMut,pDeath,b,pTrans,transMode,set);
                    save(name,'coeffPL');
                end
            end
        end

        % saving the recoreded frequencies
        if freqMode == 1
            figure(figureIndex)
            for i = valueIni+1:size(freqTrait,2)
                plot([1:tMax],freqTrait(:,i)); hold on; 
            end
            figureIndex = figureIndex+1;
            if copyAll == 0
                name = sprintf('./data/frequencies/freqTime_N%02d_pMut%02d_pDeath%02d_b%02d_thLow%01d_thHigh%01d_pTrans%02d_transMode%01d_set%01d.csv',nPop,pMut,pDeath,b,copyThresholdLow,copyThresholdHigh,pTrans,transMode,set);
                writematrix(freqTrait,name);
            else    
                name = sprintf('./data/frequencies/freqTime_N%02d_pMut%02d_pDeath%02d_b%02d_pTrans%02d_transMode%01d_set%01d_ALL.txt',nPop,pMut,pDeath,b,pTrans,transMode,set);
                writematrix(freqTrait,name);
            end
        end

end
