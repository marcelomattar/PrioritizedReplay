load_existing_data = true;
addpath('../../../');

if load_existing_data
    load('../../Figure3_FvsR_balance/Code/genFig_FvsR_openMaze.mat','simData','params')
else
    %% STATE-SPACE PARAMETERS
    setParams;
    params.maze             = zeros(6,9); % zeros correspond to 'visitable' states
    params.maze(2:4,3)      = 1; % wall
    params.maze(1:3,8)      = 1; % wall
    params.maze(5,6)        = 1; % wall
    params.s_start          = [3,1]; % beginning state (in matrix notation)
    params.s_start_rand     = true; % Start at random locations after reaching goal
    
    params.s_end            = [1,9]; % goal state (in matrix notation)
    params.rewMag           = 1; % reward magnitude (rows: locations; columns: values)
    params.rewSTD           = 0.1; % reward Gaussian noise (rows: locations; columns: values)
    params.rewProb          = 1; % probability of receiving each reward (columns: values)
    
    %% OVERWRITE PARAMETERS
    params.N_SIMULATIONS    = 100; % number of times to run the simulation
    params.MAX_N_STEPS      = 1e5; % maximum number of steps to simulate
    params.MAX_N_EPISODES   = 50; % maximum number of episodes to simulate (use Inf if no max)
    params.nPlan            = 20; % number of steps to do in planning (set to zero if no planning or to Inf to plan for as long as it is worth it)
    params.onVSoffPolicy    = 'off-policy'; % Choose 'off-policy' (default, learns Q*) or 'on-policy' (learns Qpi) learning for updating Q-values and computing gain
    params.alpha            = 1.0; % learning rate
    params.gamma            = 0.9; % discount factor
    params.softmaxInvT      = 5; % soft-max inverse temperature temperature
    
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
numStepsToCheck = 10;
only_unique_states = false;
replayLength_to_consider = 5;


%% RUN ANALYSES

% Get action consequences from stNac2stp1Nr()
nextState = nan(numel(params.maze),4);
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
chanceLevel = 1/numel(validStates); % overall chance level

stepsBack2For_prob = nan(numStepsToCheck,params.nPlan,params.N_SIMULATIONS); % Fraction of events in which the state visited N steps in the past appears at each position of the replayed forward sequence
stepsFor2For_prob = nan(numStepsToCheck,params.nPlan,params.N_SIMULATIONS); % Fraction of events in which the state visited N steps in the future appears at each position of the replayed forward sequence
stepsBack2Rev_prob = nan(numStepsToCheck,params.nPlan,params.N_SIMULATIONS); % Fraction of events in which the state visited N steps in the past appears at each position of the replayed reverse sequence
stepsFor2Rev_prob = nan(numStepsToCheck,params.nPlan,params.N_SIMULATIONS); % Fraction of events in which the state visited N steps in the future appears at each position of the replayed reverse sequence
forwardNull_prob = nan(params.N_SIMULATIONS,1);
reverseNull_prob = nan(params.N_SIMULATIONS,1);
stepsBack2For2_prob = nan(params.N_SIMULATIONS,numStepsToCheck); % Fraction of events in which the state visited N steps in the past is part of the current chunk replayed (forward)
stepsFor2For2_prob = nan(params.N_SIMULATIONS,numStepsToCheck); % Fraction of events in which the state visited N steps in the future is part of the current chunk replayed (forward)
stepsBack2Rev2_prob = nan(params.N_SIMULATIONS,numStepsToCheck); % Fraction of events in which the state visited N steps in the past is part of the current chunk replayed (reverse)
stepsFor2Rev2_prob = nan(params.N_SIMULATIONS,numStepsToCheck); % Fraction of events in which the state visited N steps in the future is part of the current chunk replayed (reverse)
for k=1:length(simData)
    stepsBack2For_count = zeros(numStepsToCheck,params.nPlan);
    stepsFor2For_count = zeros(numStepsToCheck,params.nPlan);
    stepsBack2Rev_count = zeros(numStepsToCheck,params.nPlan);
    stepsFor2Rev_count = zeros(numStepsToCheck,params.nPlan);
    stepsBack2For2_count = nan(0,numStepsToCheck);
    stepsFor2For2_count = nan(0,numStepsToCheck);
    stepsBack2Rev2_count = nan(0,numStepsToCheck);
    stepsFor2Rev2_count = nan(0,numStepsToCheck);
    numForward = 0;
    numReverse = 0;
    forwardNull = nan(0,1);
    reverseNull = nan(0,1);
    
    fprintf('Simulation #%d\n',k);
    % Identify candidate replay events: timepoints in which the number of replayed states is greater than minFracCells,minNumCells
    candidateEvents = find(cellfun('length',simData(k).replay.state)>=max(sum(params.maze(:)==0)*minFracCells,minNumCells));
    lapNum = [0;simData(k).numEpisodes(1:end-1)] + 1; % episode number for each time point
    lapNum_events = lapNum(candidateEvents); % episode number for each candidate event
    agentPos = simData(k).expList(candidateEvents,1); % agent position during each candidate event
    
    for e=1:length(candidateEvents)
        eventState = simData(k).replay.state{candidateEvents(e)}; % In a multi-step sequence, simData.replay.state has 1->2 in one row, 2->3 in another row, etc
        eventAction = simData(k).replay.action{candidateEvents(e)}; % In a multi-step sequence, simData.replay.action has the action taken at each step of the trajectory
        
        % Identify break points in this event, separating event into sequences
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
            if isempty(eventDir{i}) % If this transition was neither forward nor backward
                breakPts = [breakPts (i-1)]; % Then, call this a breakpoint
            elseif i>1
                if ~strcmp(eventDir{i},eventDir{i-1}) % If this transition was forward and the previous was backwards (or vice-versa)
                    breakPts = [breakPts (i-1)]; % Then, call this a breakpoint
                end
            end
            if i==(length(eventState)-1)
                breakPts = [breakPts i]; % Add a breakpoint after the last transition
            end
        end
        
        % Break this event into segments of sequential activity
        for j=1:(numel(breakPts)-1)
            thisChunk = (breakPts(j)+1):(breakPts(j+1));
            if (length(thisChunk)+1) >= minNumCells
                % Extract information from this sequential event
                replayDir = eventDir(thisChunk); % Direction of transition
                replayState = eventState([thisChunk (thisChunk(end)+1)]); % Start state
                replayAction = eventAction([thisChunk (thisChunk(end)+1)]); % Action
                
                % Assess the significance of this event
                %allPerms = cell2mat(arrayfun(@(x)randperm(length(replayState)),(1:nPerm)','UniformOutput',0));
                sigBool = true; 
                if runPermAnalysis
                    fracFor = nanmean(strcmp(replayDir,'F')); % Fraction of transitions in this chunk whose direction was forward
                    fracRev = nanmean(strcmp(replayDir,'R')); % Fraction of transitions in this chunk whose direction was reverse
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
                    % As a reminder, candidateEvents(e) is a timepoint.
                    agentPos = simData(k).expList(candidateEvents(e),1); % Agent's position when this (significant) chunk was replayed
                    if replayDir{1}=='F'
                        stepsBack2For = zeros(numStepsToCheck,params.nPlan); % Where in this forward sequence does the state visited N steps in the past shows up?
                        stepsBack2For2 = nan(1,numStepsToCheck); % Does the state visited N steps in the past shows up anywhere in this forward sequence?
                        stepsFor2For = zeros(numStepsToCheck,params.nPlan); % Where in this forward sequence does the state visited N steps in the future shows up?
                        stepsFor2For2 = nan(1,numStepsToCheck); % Does the state visited N steps in the future shows up anywhere in this forward sequence?
                        
                        % Look t steps back
                        start2now = simData(k).expList(1:candidateEvents(e),1); % List of states that were visited before the current (from first until current)
                        for t1=1:numStepsToCheck
                            if ~isempty(start2now)
                                if only_unique_states
                                    start2now = start2now(1:find(start2now~=start2now(end),1,'last')); % Remove the state evaluated previously
                                else
                                    start2now = start2now(1:(end-1)); % Remove the state evaluated previously
                                end
                            end
                            if ~isempty(start2now)
                                stepsBack2For(t1,find(replayState==start2now(end))) = 1; %#ok<FNDSB>
                                stepsBack2For2(t1) = any(ismember(replayState(1:min(numel(replayState),replayLength_to_consider)),start2now(end))); % Does the state visited N steps in the past shows up anywhere in this forward sequence?
                            end
                        end
                        stepsBack2For_count = stepsBack2For_count + stepsBack2For;
                        stepsBack2For2_count = [stepsBack2For2_count; stepsBack2For2];
                        
                        % Look t steps ahead
                        now2end = simData(k).expList(candidateEvents(e):end,1); % List of states that will be visited after the current (from current until last)
                        for t2=1:numStepsToCheck
                            if ~isempty(now2end)
                                if only_unique_states
                                    now2end = now2end(find(now2end~=now2end(1),1,'first'):end); % Remove the state evaluated previously
                                else
                                    now2end = now2end(2:end); % Remove the state evaluated previously
                                end
                            end
                            if ~isempty(now2end)
                                stepsFor2For(t2,find(replayState==now2end(1))) = 1; %#ok<FNDSB>
                                stepsFor2For2(t2) = any(ismember(replayState(1:min(numel(replayState),replayLength_to_consider)),now2end(1))); % Does the state visited N steps in the future shows up anywhere in this forward sequence?
                            end
                        end
                        stepsFor2For_count = stepsFor2For_count + stepsFor2For;
                        stepsFor2For2_count = [stepsFor2For2_count; stepsFor2For2];
                        numForward = numForward + 1;
                        
                        % Compute chance level
                        chanceLevel = numel(replayState)/numel(validStates);
                        forwardNull = [forwardNull; chanceLevel];
                        
                    elseif replayDir{1}=='R'
                        stepsBack2Rev = zeros(numStepsToCheck,params.nPlan); % Where in this reverse sequence does the state visited N steps in the past shows up?
                        stepsBack2Rev2 = nan(1,numStepsToCheck); % Does the state visited N steps in the past shows up anywhere in this reverse sequence?
                        stepsFor2Rev = zeros(numStepsToCheck,params.nPlan); % Where in this reverse sequence does the state visited N steps in the future shows up?
                        stepsFor2Rev2 = nan(1,numStepsToCheck); % Does the state visited N steps in the future shows up anywhere in this reverse sequence?
                        % Look t steps back
                        start2now = simData(k).expList(1:candidateEvents(e),1); % List of states that were visited before the current (from first until current)
                        for t1=1:numStepsToCheck
                            if ~isempty(start2now)
                                if only_unique_states
                                    start2now = start2now(1:find(start2now~=start2now(end),1,'last')); % Remove the state evaluated previously
                                else
                                    start2now = start2now(1:(end-1)); % Remove the state evaluated previously
                                end
                            end
                            if ~isempty(start2now)
                                stepsBack2Rev(t1,find(replayState==start2now(end))) = 1; %#ok<FNDSB>
                                stepsBack2Rev2(t1) = any(ismember(replayState(1:min(numel(replayState),replayLength_to_consider)),start2now(end))); % Does the state visited N steps in the past shows up anywhere in this reverse sequence?
                            end
                        end
                        stepsBack2Rev_count = stepsBack2Rev_count + stepsBack2Rev;
                        stepsBack2Rev2_count = [stepsBack2Rev2_count; stepsBack2Rev2];
                        
                        % Look t steps ahead
                        now2end = simData(k).expList(candidateEvents(e):end,1); % List of states that will be visited after the current (from current until last)
                        for t2=1:numStepsToCheck
                            if ~isempty(now2end)
                                if only_unique_states
                                    now2end = now2end(find(now2end~=now2end(1),1,'first'):end); % Remove the state evaluated previously
                                else
                                    now2end = now2end(2:end); % Remove the state evaluated previously
                                end
                            end
                            if ~isempty(now2end)
                                stepsFor2Rev(t2,find(replayState==now2end(1))) = 1; %#ok<FNDSB>
                                stepsFor2Rev2(t2) = any(ismember(replayState(1:min(numel(replayState),replayLength_to_consider)),now2end(1))); % Does the state visited N steps in the future shows up anywhere in this reverse sequence?
                            end
                        end
                        stepsFor2Rev_count = stepsFor2Rev_count + stepsFor2Rev;
                        stepsFor2Rev2_count = [stepsFor2Rev2_count; stepsFor2Rev2];
                        numReverse = numReverse + 1;
                        
                        % Compute chance level
                        chanceLevel = numel(replayState)/numel(validStates);
                        reverseNull = [reverseNull; chanceLevel];
                        
                    end
                end
            end
        end
    end
    stepsBack2For_prob(:,:,k) = stepsBack2For_count/numForward; % Fraction of events in which the state visited N steps in the past appears at each position of the replayed forward sequence
    stepsFor2For_prob(:,:,k) = stepsFor2For_count/numForward; % Fraction of events in which the state visited N steps in the future appears at each position of the replayed forward sequence
    stepsBack2Rev_prob(:,:,k) = stepsBack2Rev_count/numReverse; % Fraction of events in which the state visited N steps in the past appears at each position of the replayed reverse sequence
    stepsFor2Rev_prob(:,:,k) = stepsFor2Rev_count/numReverse; % Fraction of events in which the state visited N steps in the future appears at each position of the replayed reverse sequence
    forwardNull_prob(k) = nanmean(forwardNull);
    reverseNull_prob(k) = nanmean(reverseNull);
    stepsBack2For2_prob(k,:) = nanmean(stepsBack2For2_count,1); % Fraction of events in which the state visited N steps in the past is part of the current chunk replayed (forward)
    stepsFor2For2_prob(k,:) = nanmean(stepsFor2For2_count,1); % Fraction of events in which the state visited N steps in the future is part of the current chunk replayed (forward)
    stepsBack2Rev2_prob(k,:) = nanmean(stepsBack2Rev2_count,1); % Fraction of events in which the state visited N steps in the past is part of the current chunk replayed (reverse)
    stepsFor2Rev2_prob(k,:) = nanmean(stepsFor2Rev2_count,1); % Fraction of events in which the state visited N steps in the future is part of the current chunk replayed (reverse)
end



%% PLOT

figure(1); clf;
subplot(1,2,1); hold on;
plot(nanmean(stepsBack2For2_prob));
plot(nanmean(stepsFor2For2_prob));
xlabel('Number of steps');
ylabel('Probability');
title('Forward replay');
legend({'Previous steps','Next steps'});
l1=line(xlim,[nanmean(forwardNull_prob) nanmean(forwardNull_prob)]);
set(l1,'Color',[0.3 0.3 0.3],'LineWidth',1)
ylim([0 1]); grid on

subplot(1,2,2); hold on;
plot(nanmean(stepsBack2Rev2_prob));
plot(nanmean(stepsFor2Rev2_prob));
xlabel('Number of steps');
ylabel('Probability');
title('Reverse replay');
legend({'Previous steps','Next steps'});
l2=line(xlim,[nanmean(reverseNull_prob) nanmean(reverseNull_prob)]);
set(l2,'Color',[0.3 0.3 0.3],'LineWidth',1)
ylim([0 1]); grid on



figure(2); clf
subplot(2,1,1);
p1=plot(-numStepsToCheck:1:numStepsToCheck,[fliplr(nanmean(stepsBack2For2_prob)) NaN nanmean(stepsFor2For2_prob)]);
set(p1,'LineWidth',2);
title('Forward replay'); xlabel('Number of steps back/forward'); ylabel('Fraction of sequences with that state'); ylim([0 1]); grid on;
l1=line(xlim,[nanmean(forwardNull_prob) nanmean(forwardNull_prob)]);
set(l1,'Color',[0.3 0.3 0.3],'LineWidth',1)

subplot(2,1,2); title('Reverse replay');
p2=plot(-numStepsToCheck:1:numStepsToCheck,[fliplr(nanmean(stepsBack2Rev2_prob)) NaN nanmean(stepsFor2Rev2_prob)]);
set(p2,'LineWidth',2);
title('Reverse replay'); xlabel('Number of steps back/forward'); ylabel('Fraction of sequences with that state'); ylim([0 1]); grid on;
l2=line(xlim,[nanmean(reverseNull_prob) nanmean(reverseNull_prob)]);
set(l2,'Color',[0.3 0.3 0.3],'LineWidth',1)

set(gcf,'Position',[133   281   297   545])
if only_unique_states
    display(sprintf('PS: These analysis ignores consecutive visits to the same state'));
else
    display(sprintf('PS: These analysis DOES NOT ignore consecutive visits to the same state'));
end
display(sprintf('PS2: Importantly, only the first %d elements of a replay sequence are included in the analysis!',replayLength_to_consider));





figure(3); clf
subplot(2,2,1);
imagesc(nanmean(stepsBack2For_prob,3)); caxis([0 0.5]); colorbar; %axis square
title('Forward replay'); ylabel('Steps in the past'); xlabel('Position in replay');
subplot(2,2,2);
imagesc(nanmean(stepsFor2For_prob,3)); caxis([0 0.5]); colorbar; %axis square
title('Forward replay'); ylabel('Steps in the future'); xlabel('Position in replay');
subplot(2,2,3);
imagesc(nanmean(stepsBack2Rev_prob,3)); caxis([0 0.5]); colorbar; %axis square
title('Reverse replay'); ylabel('Steps in the past'); xlabel('Position in replay');
subplot(2,2,4);
imagesc(nanmean(stepsFor2Rev_prob,3)); caxis([0 0.5]); colorbar; %axis square
title('Reverse replay'); ylabel('Steps in the future'); xlabel('Position in replay');


%% EXPORT FIGURE
if saveBool
    save genFig_replay2behavior_openMaze_randStart.mat
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    set(gcf, 'renderer', 'painters');
    export_fig(['../Parts/genFig_replay2behavior_openMaze_randStart_1'], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
end
