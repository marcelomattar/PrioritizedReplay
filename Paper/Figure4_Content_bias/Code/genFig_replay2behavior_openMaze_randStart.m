load_existing_data = true;
addpath('../../../');

if load_existing_data
    load('../../Figure2_Forward_vs_Reverse/Code/genFig_FvsR_openMaze.mat','simData','params')
else
    %% STATE-SPACE PARAMETERS
    setParams;
    params.maze             = zeros(6,9); % zeros correspond to 'visitable' states
    params.maze(2:4,3)      = 1; % wall
    params.maze(1:3,8)      = 1; % wall
    params.maze(5,6)        = 1; % wall
    %params.s_end            = [1,9;6,9]; % goal state (in matrix notation)
    params.s_end            = [1,9]; % goal state (in matrix notation)
    params.s_start          = [3,1]; % beginning state (in matrix notation)
    params.s_start_rand     = true; % Start at random locations after reaching goal
    
    %% OVERWRITE PARAMETERS
    params.N_SIMULATIONS    = 10000; % number of times to run the simulation
    params.MAX_N_STEPS      = 1e5; % maximum number of steps to simulate
    params.MAX_N_EPISODES   = 50; % maximum number of episodes to simulate (use Inf if no max) -> Choose between 20 and 100
    params.nPlan            = 20; % number of steps to do in planning (set to zero if no planning or to Inf to plan for as long as it is worth it)
    
    params.setAllGainToOne  = false; % Set the gain term of all items to one (for illustration purposes)
    params.setAllNeedToOne  = false; % Set the need term of all items to one (for illustration purposes)
    params.rewSTD           = 0.1; % reward standard deviation (can be a vector -- e.g. [1 0.1])
    params.softmaxT         = 0.2; % soft-max temperature -> higher means more exploration and, therefore, more reverse replay
    params.gamma            = 0.90; % discount factor
    
    params.updIntermStates  = true; % Update intermediate states when performing n-step backup
    params.baselineGain     = 1e-10; % Gain is set to at least this value (interpreted as "information gain") -> Use 1e-3 if LR=0.8
    
    params.alpha            = 1.0; % learning rate for real experience (non-bayesian)
    params.copyQinPlanBkps  = false; % Copy the Q-value (mean and variance) on planning backups (i.e., LR=1.0)
    params.copyQinGainCalc  = true; % Copy the Q-value (mean and variance) on gain calculation (i.e., LR=1.0)
    
    rng(mean('replay'));
    for k=1:params.N_SIMULATIONS
        simData(k) = replaySim(params);
    end
end

saveStr = input('Do you want to produce figures (y/n)? ','s');
if strcmp(saveStr,'y')
    saveBool = true;
else
    saveBool = false;
end


%% ANALYSIS PARAMETERS
minNumCells = 5;
minFracCells = 0;
runPermAnalysis = true; % Run permutation analysis (true or false)
nPerm = 500; % Number of permutations for assessing significance of an event
numStepsToCheck = 20;


%% RUN ANALYSES

% Get action consequences from stNac2stp1Nr()
for s=1:numel(params.maze)
    [I,J] = ind2sub(size(params.maze),s);
    st=nan(1,2);
    st(1)=I; st(2) = J;
    for a=1:4
        [~,~,stp1i] = stNac2stp1Nr(st,a,params);
        nextState(s,a) = stp1i;
    end
end

% Build null model
validStates = find(params.maze==0); % can start at any non-wall state
validStates = validStates(~ismember(validStates,sub2ind(size(params.maze),params.s_end(:,1),params.s_end(:,2)))); %... but remove the goal states from the list
chanceLevel = 1/numel(validStates);


for k=1:length(simData)
    prev2For_count = nan(0,1);
    next2For_count = nan(0,1);
    prev2Rev_count = nan(0,1);
    next2Rev_count = nan(0,1);
    stepsBack2For_count = nan(0,numStepsToCheck);
    stepsFor2For_count = nan(0,numStepsToCheck);
    stepsBack2Rev_count = nan(0,numStepsToCheck);
    stepsFor2Rev_count = nan(0,numStepsToCheck);
    numReverseEvents = 0;
    numForwardEvents = 0;
    forwardNull = nan(0,1);
    reverseNull = nan(0,1);
    
    fprintf('Simulation #%d\n',k);
    candidateEvents = find(cellfun('length',simData(k).replay.state)>=max(sum(params.maze(:)==0)*minFracCells,minNumCells));
    lapNum = [0;simData(k).numEpisodes(1:end-1)] + 1;
    lapNum_events = lapNum(candidateEvents);
    agentPos = simData(k).expList(candidateEvents,1);
    
    for e=1:length(candidateEvents)
        eventState = simData(k).replay.state{candidateEvents(e)};
        eventAction = simData(k).replay.action{candidateEvents(e)};
        
        % Identify break points in this event, separating event into
        % sequences
        eventDir = cell(1,length(eventState)-1);
        breakPts = 0; % Save breakpoints that divide contiguous replay events
        for i=1:(length(eventState)-1)
            % If state(i) and action(i) leads to state(i+1): FORWARD
            if nextState(eventState(i),eventAction(i)) == eventState(i+1)
                eventDir{i} = 'F';
            end
            % If state(i+1) and action(i+1) leads to state(i): REVERSE
            if nextState(eventState(i+1),eventAction(i+1)) == eventState(i)
                eventDir{i} = 'R';
            end
            
            % Find if this is a break point
            if isempty(eventDir{i})
                breakPts = [breakPts (i-1)];
            elseif i>1
                if ~strcmp(eventDir{i},eventDir{i-1})
                    breakPts = [breakPts (i-1)];
                end
            end
            if i==(length(eventState)-1)
                breakPts = [breakPts i];
            end
        end
        
        % Break this event into segments of sequential activity
        for j=1:(numel(breakPts)-1)
            thisChunk = (breakPts(j)+1):(breakPts(j+1));
            if (length(thisChunk)+1) >= minNumCells
                % Extract information from this sequential event
                replayDir = eventDir(thisChunk);
                replayState = eventState([thisChunk (thisChunk(end)+1)]);
                replayAction = eventAction([thisChunk (thisChunk(end)+1)]);
                
                % Assess the significance of this event
                %allPerms = cell2mat(arrayfun(@(x)randperm(length(replayState)),(1:nPerm)','UniformOutput',0));
                sigBool = true; %#ok<NASGU>
                if runPermAnalysis
                    fracFor = nanmean(strcmp(replayDir,'F'));
                    fracRev = nanmean(strcmp(replayDir,'R'));
                    disScore = fracFor-fracRev;
                    dirScore_perm = nan(1,nPerm);
                    for p=1:nPerm
                        thisPerm = randperm(length(replayState));
                        replayState_perm = replayState(thisPerm);
                        replayAction_perm = replayAction(thisPerm);
                        replayDir_perm = cell(1,length(replayState_perm)-1);
                        for i=1:(length(replayState_perm)-1)
                            if nextState(replayState_perm(i),replayAction_perm(i)) == replayState_perm(i+1)
                                replayDir_perm{i} = 'F';
                            end
                            if nextState(replayState_perm(i+1),replayAction_perm(i+1)) == replayState_perm(i)
                                replayDir_perm{i} = 'R';
                            end
                        end
                        fracFor = nanmean(strcmp(replayDir_perm,'F'));
                        fracRev = nanmean(strcmp(replayDir_perm,'R'));
                        dirScore_perm(p) = fracFor-fracRev;
                    end
                    dirScore_perm = sort(dirScore_perm);
                    lThresh_score = dirScore_perm(floor(nPerm*0.025));
                    hThresh_score = dirScore_perm(ceil(nPerm*0.975));
                    if (disScore<lThresh_score) || (disScore>hThresh_score)
                        sigBool = true;
                    else
                        sigBool = false;
                    end
                end
                
                % Add significant events to 'bucket'
                if sigBool
                    agentPos = simData(k).expList(candidateEvents(e),1);
                    start2now = simData(k).expList(1:candidateEvents(e),1);
                    now2end = simData(k).expList((candidateEvents(e)+1):end,1);
                    
                    try
                        prevSt = simData(k).expList(find(start2now~=agentPos,1,'last'),1);
                        nextSt = now2end(1);
                        if replayDir{1}=='F'
                            prev2For_count = [prev2For_count ; (eventState(1)==prevSt)];
                            next2For_count = [next2For_count ; (eventState(1)==nextSt)];
                            numForwardEvents = numForwardEvents+1;
                        elseif replayDir{1}=='R'
                            prev2Rev_count = [prev2Rev_count ; (eventState(1)==prevSt)];
                            next2Rev_count = [next2Rev_count ; (eventState(1)==nextSt)];
                            numReverseEvents = numReverseEvents+1;
                        end
                    end
                    
                    stepsBack2For = zeros(1,numStepsToCheck);
                    stepsFor2For = zeros(1,numStepsToCheck);
                    stepsBack2Rev = zeros(1,numStepsToCheck);
                    stepsFor2Rev = zeros(1,numStepsToCheck);
                    now2end = [agentPos;now2end];
                    for t=1:numStepsToCheck
                        try
                            start2now = start2now(1:find(start2now~=start2now(end),1,'last'));
                            now2end = now2end(find(now2end~=now2end(1),1,'first'):end);
                            if replayDir{1}=='F'
                                stepsBack2For(1,t) = sum(ismember(eventState,start2now(end)));
                                stepsFor2For(1,t) = sum(ismember(eventState,now2end(1)));
                            elseif replayDir{1}=='R'
                                stepsBack2Rev(1,t) = sum(ismember(eventState,start2now(end)));
                                stepsFor2Rev(1,t) = sum(ismember(eventState,now2end(1)));
                            end
                        end
                    end
                    if replayDir{1}=='F'
                        stepsBack2For_count = [stepsBack2For_count;stepsBack2For];
                        stepsFor2For_count = [stepsFor2For_count;stepsFor2For];
                        forwardNull = [forwardNull; numel(unique(eventState))./numel(validStates)];
                    elseif replayDir{1}=='R'
                        stepsBack2Rev_count = [stepsBack2Rev_count;stepsBack2Rev];
                        stepsFor2Rev_count = [stepsFor2Rev_count;stepsFor2Rev];
                        reverseNull = [reverseNull; numel(unique(eventState))./numel(validStates)];
                    end
                end
            end
        end
    end
    prev2For_prob(k) = nanmean(prev2For_count);
    next2For_prob(k) = nanmean(next2For_count);
    prev2Rev_prob(k) = nanmean(prev2Rev_count);
    next2Rev_prob(k) = nanmean(next2Rev_count);
    stepsBack2For_prob(k,:) = sum(stepsBack2For_count>0) ./ numForwardEvents;
    stepsFor2For_prob(k,:) = sum(stepsFor2For_count>0) ./ numForwardEvents;
    stepsBack2Rev_prob(k,:) = sum(stepsBack2Rev_count>0) ./ numReverseEvents;
    stepsFor2Rev_prob(k,:) = sum(stepsFor2Rev_count>0) ./ numReverseEvents;
    forwardNull_prob(k) = nanmean(forwardNull);
    reverseNull_prob(k) = nanmean(reverseNull);
end



%% PLOT

figure(1); clf;
f1 = bar([nanmean(next2For_prob) nanmean(prev2For_prob) ; nanmean(next2Rev_prob) nanmean(prev2Rev_prob)]);
f1(1).FaceColor=[1 1 1]; % Replay bar color
f1(1).LineWidth=1;
f1(2).FaceColor=[0 0 0]; % Replay bar color
f1(2).LineWidth=1;
set(f1(1).Parent,'XTickLabel',{'Forward','Reverse'});
legend({'Next state','Previous state'},'Location','NortheastOutside');
hold on;
grid on;
l1 = line(xlim,[chanceLevel chanceLevel]);
ylabel('Probability');
title('Predicatibility of past/future paths');


figure(2); clf;
subplot(1,2,1); hold on;
plot(nanmean(stepsBack2For_prob));
plot(nanmean(stepsFor2For_prob));
xlabel('Number of steps');
ylabel('Probability');
title('Forward replay');
legend({'Previous steps','Next steps'});
l1=line(xlim,[nanmean(forwardNull_prob) nanmean(forwardNull_prob)]);
set(l1,'Color',[0.3 0.3 0.3],'LineWidth',1)
ylim([0 1]); grid on

subplot(1,2,2); hold on;
plot(nanmean(stepsBack2Rev_prob));
plot(nanmean(stepsFor2Rev_prob));
xlabel('Number of steps');
ylabel('Probability');
title('Reverse replay');
legend({'Previous steps','Next steps'});
l2=line(xlim,[nanmean(reverseNull_prob) nanmean(reverseNull_prob)]);
set(l2,'Color',[0.3 0.3 0.3],'LineWidth',1)
ylim([0 1]); grid on


%% EXPORT FIGURE
if saveBool
    save genFig_replay2behavior_openMaze_randStart.mat
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    set(gcf, 'renderer', 'painters');
    export_fig(['../Parts/' mfilename], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
end
