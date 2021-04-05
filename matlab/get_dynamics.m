function [pop,value,ageCount,namesFreq] = get_dynamics(t,pop,value,pDeath,nPop,pMut,b,pTrans,transMode,copyAll,copyThresholdHigh,copyThresholdLow,valueIni,variantAgeMode,ageCount,PDmode,namesFreq,lagBetweenSamples,sampleFrac)

poph = pop; % in order to allow consistency with the WF model we assume that names are copied from the previous period
% death
indexDeath = [];
nBirth = binornd(nPop,pDeath); % number of death = number of birth 
indexDeath = randsample(nPop,nBirth); % generating indices of individuals to be removed
pop(:,indexDeath) = []; % remove 'dead' individuals from population

% reproduction
nMut = binornd(nBirth,pMut); % number of innovations
if copyAll == 0
    copyIndex = find(poph(2,:)>(t-copyThresholdHigh) & poph(2,:)<(t-copyThresholdLow) ); % defining copy pool 
    if isempty(copyIndex)
        copyIndex = find(poph(2,:)>(t-(copyThresholdHigh+1)) & poph(2,:)<(t-(copyThresholdLow-1)));
    end
    % choosing role models from copy pool
    type = unique(poph(1,copyIndex)); % variant types present in copy pool
    if length(type)>1
        h = hist(poph(1,copyIndex),type); % frequencies of all variant types currently present
        h = (h./nPop).^(1+b); % b~=0 includes frequency-dependent transmission 
        hAdd = type(randp(h,[nBirth-nMut,1])); 
    else %randp works differently if h is a number
        hAdd = type*ones(1,nBirth-nMut);
    end
else
    % choosing role models from copy pool
    type = unique(poph(1,:)); % variant types present in population 
    if length(type)>1
        h = hist(poph(1,:),type); % frequencies of all variant types currently present
        h = (h./nPop).^(1+b); % b~=0 includes frequency-dependent transmission 
        hAdd = type(randp(h,[nBirth-nMut,1]));
    else %randp works differently if h is a number
        hAdd = type*ones(1,nBirth-nMut);
    end
end
poph = [];
n = nPop-nBirth;
pop(1,n+1:n+nBirth-nMut) = hAdd; % adding copied types
pop(1,n+nBirth-nMut+1:nPop) = value + [1:nMut]; % adding innovations
pop(2,n+1:nPop) = t*ones(1,nBirth); % adding birth date

%tracking the age of the variants
if variantAgeMode == 1 && nMut>0
    ageCount(1:2,size(ageCount,2)+1:size(ageCount,2)+nMut) = [value + [1:nMut];zeros(1,nMut)];
    %typesNew = unique(pop(1,:)); %present at the moment
    typesNew = unique(hAdd);
    h1 = find(typesNew>valueIni); % indices of innovations in present set of variants
    h = intersect(ageCount(1,:),typesNew(h1));
    ageCount(2,h-valueIni) = ageCount(2,h-valueIni)+1;
end

if PDmode == 1
    if lagBetweenSamples == 0 & sampleFrac == 1 % progeny count taken at consecutive time points and count contains all progeny
        names = unique(hAdd); % copied types
        if numel(names)>1
            [progFreq] = hist(hAdd,names);
            namesFreq(names) = namesFreq(names) + progFreq; % updating progeny count
        else
            namesFreq(names) = namesFreq(names) + numel(hAdd); % updating progeny count
        end
        namesFreq = [namesFreq [ones(1,nMut)]]; % adding innovations to progeny count
    elseif  mod(t,lagBetweenSamples) == 0 
        hAdd = [hAdd [value+1:value+nMut]];
        if sampleFrac < 1
            hAdd = randsample(hAdd,round(numel(hAdd)*sampleFrac)); % generating the progeny sample
        end
        names = unique(hAdd); % types in the sample
        [commonNames,indexCommon1,indexCommon2] = intersect(namesFreq(2,:),names); % types (and indices) that are in both, current sample and previous progeny count
        [differentNames,indexDiff] = setdiff(names,namesFreq(2,:)); % types (and indices) of types that are only in current sample, i.e. innovations
        if numel(names)>1
            [progFreq] = hist(hAdd,names); % frequency count 
            if isempty(indexDiff) == 0 % innovations in hAdd
                namesFreq = [namesFreq zeros(2,numel(indexDiff))]; 
                namesFreq(:,size(namesFreq,2)-numel(indexDiff)+1:size(namesFreq,2)) = [progFreq(indexDiff);names(indexDiff)]; % adding innovations to progeny count
            end
            namesFreq(1,indexCommon1) = namesFreq(1,indexCommon1)+progFreq(indexCommon2); % updating progeny count
        else % if there is only one type in hAdd
            if isempty(indexDiff) == 0
                namesFreq = [namesFreq [numel(hAdd); names]]; % adding innovation to progeny count
            else
                namesFreq(1,names(indexCommon2)) = namesFreq(1,names(indexCommon2)) + numel(hAdd); % updating progeny count 
            end
        end
    end
end

value = value + nMut;

%cultural transmission during an individual's life time (only active for transMode>0) 
% ---- NOT USED CURRENTLY ---- NEEDS TO BE CHECKED FOR CONSISTENCY!!!
if transMode > 0
    nTrans = binornd(nPop,pTrans);
    indexTrans = randsample(nPop,nTrans);
    if transMode == 1 %oblique transmssion
        for i = 1:length(indexTrans)
            indexInteraction = find(pop(2,:)<pop(2,indexTrans(i))); % find older individuals
            if isempty(indexInteraction) == 0
                if rand > pMut
                    h = randi(length(indexInteraction));
                    pop(1,indexTrans(i)) = pop(1,indexInteraction(h));
                else
                    pop(1,indexTrans(i)) = value+1;
                    value = value+1;
                end
            end
        end
    elseif transMode == 2 %horizontal transmission
        for i = 1:length(indexTrans)
            indexInteraction = find(pop(2,:)>pop(2,indexTrans(i))-5 & pop(2,:)<pop(2,indexTrans(i))+5); % find individuals in same age group
            if isempty(indexInteraction) == 0
                if rand > pMut
                    h = randi(length(indexInteraction));
                    pop(1,indexTrans(i)) = pop(1,indexInteraction(h));
                else
                    pop(1,indexTrans(i)) = value+1;
                    value = value+1;
                end
            end
        end
    elseif transMode == 3 %age-neutral transmission
        for i = 1:length(indexTrans)
            if rand > pMut
                h = randi(nPop);
                pop(1,indexTrans(i)) = pop(1,h);
            else
                pop(1,indexTrans(i)) = value+1;
                value = value+1;
            end
        end
    end
end

