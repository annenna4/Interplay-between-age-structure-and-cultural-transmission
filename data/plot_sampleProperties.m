% Plots sample level statitsics of samples used for the neutrality tests

clear all
close all

nPop = 10^5; % population size
pMut = 5*10^-4; % innovation rate (per transmission event)
pDeath = 0.1; % death rate

if pDeath == 0.1 % only needed for convenience when loading files in
    x = 01;
elseif pDeath == 0.02
    x = 2;
end

copyThresholdHigh = 21; % upper bound of the age of the copying pool
copyThresholdLow = 0; % lower bound of the age of the copying pool

nSamV = [50, 100, 200, 500, 1000, 2000]; % sample size
bV = [-0.001 -0.0008 -0.0006 -0.0004 -0.0002 -0.0001 0.0001 0.0002 0.0003 0.0004 0.0005]; % strength of frequency-dependent transmission 
b = 0.0003;

%for s = 1:numel(bV)
    
    %b = bV(s);
    
    for i = 1:numel(nSamV) % loop over all sample sizes 
        
        nSam = nSamV(i);
        
        % loading samples
        name = sprintf('./thHigh%01d_pDeath0%01d/samples_N%02d_pMut%02d_pDeath%02d_b%02d_thLow%01d_thHigh%01d_nSample%02d.txt',copyThresholdHigh,x,nPop,pMut,pDeath,b,copyThresholdLow,copyThresholdHigh,nSam)
        samples = load(name);
        
        for j = 1:size(samples,2) % calculating sample properties
            
            h = nonzeros(samples(:,j));
            numTraits(j) = numel(h); % number of traits in sample j
            div(j) = sum(h.^2); % diversity level of sample j
            maxFreq(j)= max(h); % % maximum frequency in sample j
            
        end
        
            figure(1)
            [ydim,xdim] = ecdf(numTraits); % calculating cumulative density function 
            plot(xdim,ydim,'LineWidth',2); hold on;
            h = [xdim ydim];
            name = sprintf('./thHigh%01d_pDeath0%01d/ecdfNumTraits_N%02d_pMut%02d_pDeath%02d_b%02d_thLow%01d_thHigh%01d_nSample%02d.txt',copyThresholdHigh,x,nPop,pMut,pDeath,b,copyThresholdLow,copyThresholdHigh,nSam);
            save(name,'-ASCII','h');
        
            figure(2)
            [ydim,xdim] = ecdf(div); % calculating cumulative density function 
            plot(xdim,ydim,'LineWidth',2); hold on;
            h = [xdim ydim];
            name = sprintf('./thHigh%01d_pDeath0%01d/ecdfDiversity_N%02d_pMut%02d_pDeath%02d_b%02d_thLow%01d_thHigh%01d_nSample%02d.txt',copyThresholdHigh,x,nPop,pMut,pDeath,b,copyThresholdLow,copyThresholdHigh,nSam);
            save(name,'-ASCII','h');
        
            figure(3)
            [ydim,xdim] = ecdf(maxFreq); % calculating cumulative density function 
            plot(xdim,ydim,'LineWidth',2); hold on;
            h = [xdim ydim];
            name = sprintf('./thHigh%01d_pDeath0%01d/ecdfMaxFreq_N%02d_pMut%02d_pDeath%02d_b%02d_thLow%01d_thHigh%01d_nSample%02d.txt',copyThresholdHigh,x,nPop,pMut,pDeath,b,copyThresholdLow,copyThresholdHigh,nSam);
            save(name,'-ASCII','h');
        
    end
    
    figure(1)
    legend('nSam = 50','nSam = 100','nSam = 200','nSam = 500','nSam = 1000','nSam = 2000','location','southeast')
    xlabel('Number of traits in sample')
    ylabel('Probability P(x) of number of traits < x')
    
    figure(2)
    legend('nSam = 50','nSam = 100','nSam = 200','nSam = 500','nSam = 1000','nSam = 2000','location','southeast')
    xlabel('Diversity level')
    ylabel('Probability P(x) of diversity levels < x')
    
    figure(3)
    legend('nSam = 50','nSam = 100','nSam = 200','nSam = 500','nSam = 1000','nSam = 2000','location','southeast')
    xlabel('Maximum frequency in the sample')
    ylabel('Probability P(x) of maximum frequency < x')
    
%end
