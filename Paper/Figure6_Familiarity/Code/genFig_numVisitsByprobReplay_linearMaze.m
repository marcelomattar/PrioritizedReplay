load_existing_data = true;
addpath('../../../');

if load_existing_data
    %load('../../Figure3_FvsR_balance/Code/genFig_FvsR_linearTrack.mat','simData','params')
else
    %% STATE-SPACE PARAMETERS
    setParams;
    params.maze             = zeros(3,10); % zeros correspond to 'visitable' states
    params.maze(2,:)        = 1; % wall
    params.s_start          = [1,1;3,size(params.maze,2)]; % beginning state (in matrix notation)
    params.s_start_rand     = false; % Start at random locations after reaching goal
    
    params.s_end            = [1,size(params.maze,2);3,1]; % goal state (in matrix notation)
    params.rewMag           = 1; % reward magnitude (rows: locations; columns: values)
    params.rewSTD           = 0.1; % reward Gaussian noise (rows: locations; columns: values)
    params.rewProb          = 1; % probability of receiving each reward (columns: values)
    
    %% OVERWRITE PARAMETERS
    params.N_SIMULATIONS    = 10; % number of times to run the simulation
    params.MAX_N_STEPS      = 1e5; % maximum number of steps to simulate
    params.MAX_N_EPISODES   = 50; % maximum number of episodes to simulate (use Inf if no max)
    params.nPlan            = 20; % number of steps to do in planning (set to zero if no planning or to Inf to plan for as long as it is worth it)
    params.onVSoffPolicy    = 'off-policy'; % Choose 'off-policy' (default, learns Q*) or 'on-policy' (learns Qpi) learning for updating Q-values and computing gain
    
    params.alpha            = 1.0; % learning rate
    params.gamma            = 0.9; % discount factor
    params.softmaxInvT      = 5; % soft-max inverse temperature temperature
    params.tieBreak         = 'min'; % How to break ties on EVM (choose which sequence length is prioritized: 'min', 'max', or 'rand')
    params.setAllGainToOne  = false; % Set the gain term of all items to one (for debugging purposes)
    params.setAllNeedToOne  = false; % Set the need term of all items to one (for debugging purposes)
    params.setAllNeedToZero = false; % Set the need term of all items to zero, except for the current state (for debugging purposes)
    
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


%% INITIALIZE VARIABLES
numVisits = zeros(numel(params.maze),params.MAX_N_EPISODES,params.N_SIMULATIONS);
numTimesAct = zeros(numel(params.maze)*4,params.MAX_N_EPISODES,params.N_SIMULATIONS);
numReplay = zeros(numel(params.maze),params.MAX_N_EPISODES,params.N_SIMULATIONS);
numReplayF = zeros(numel(params.maze),params.MAX_N_EPISODES,params.N_SIMULATIONS);
numReplayR = zeros(numel(params.maze),params.MAX_N_EPISODES,params.N_SIMULATIONS);
numReplayAct = zeros(numel(params.maze)*4,params.MAX_N_EPISODES,params.N_SIMULATIONS);
numReplayActF = zeros(numel(params.maze)*4,params.MAX_N_EPISODES,params.N_SIMULATIONS);
numReplayActR = zeros(numel(params.maze)*4,params.MAX_N_EPISODES,params.N_SIMULATIONS);

nextState = nan(numel(params.maze),4);


%% RUN ANALYSIS

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

agent2replayStart_prob = nan(params.N_SIMULATIONS,21);
agent2replayEnd_prob = nan(params.N_SIMULATIONS,21);
for k=1:length(simData)
    fprintf('Simulation #%d\n',k);
    % Identify candidate replay events
    candidateEvents = find(cellfun('length',simData(k).replay.state)>=max(sum(params.maze(:)==0)*minFracCells,minNumCells));
    lapNum = [0;simData(k).numEpisodes(1:end-1)] + 1;
    lapNum_events = lapNum(candidateEvents);
    agentPos = simData(k).expList(candidateEvents,1);
    
    agent2replayStart = nan(0,21);
    agent2replayEnd = nan(0,21);
    
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
                    % Count number of replayed states
                    N = histc(replayState,0.5:1:(numel(params.maze)+0.5));
                    numReplay(:,lapNum_events(e),k) = numReplay(:,lapNum_events(e),k) + N(1:end-1)';
                    if replayDir{1}=='F'
                        numReplayF(:,lapNum_events(e),k) = numReplayF(:,lapNum_events(e),k) + N(1:end-1)';
                    elseif replayDir{1}=='R'
                        numReplayR(:,lapNum_events(e),k) = numReplayR(:,lapNum_events(e),k) + N(1:end-1)';
                    end
                    
                    % Count number of replayed actions
                    actIdx = sub2ind([numel(params.maze) 4],replayState,replayAction);
                    N = histc(actIdx,0.5:1:((numel(params.maze)*4)+0.5));
                    numReplayAct(:,lapNum_events(e),k) = numReplayAct(:,lapNum_events(e),k) + N(1:end-1)';
                    if replayDir{1}=='F'
                        numReplayActF(:,lapNum_events(e),k) = numReplayActF(:,lapNum_events(e),k) + N(1:end-1)';
                    elseif replayDir{1}=='R'
                        numReplayActR(:,lapNum_events(e),k) = numReplayActR(:,lapNum_events(e),k) + N(1:end-1)';
                    end
                end
            end
        end
    end
    
    % Count number of visits to each state
    for l=1:params.MAX_N_EPISODES
        N = histc(simData(k).expList(lapNum==l,1),0.5:1:(numel(params.maze)+0.5));
        numVisits(:,l,k) = N(1:end-1);
    end
    
    % Count number of specific state-action combinations
    for l=1:params.MAX_N_EPISODES
        actIdx = sub2ind([numel(params.maze) 4],simData(k).expList(lapNum==l,1),simData(k).expList(lapNum==l,2));
        N = histc(actIdx,0.5:1:((numel(params.maze)*4)+0.5));
        numTimesAct(:,l,k) = N(1:end-1);
    end
end


%% ANALYZE DATA PER LAP

% Analyze replay by state
nVisits2test = 0:20;
probReplay_by_numVisits = nan(numel(nVisits2test),params.MAX_N_EPISODES,params.N_SIMULATIONS);
probReplayF_by_numVisits = nan(numel(nVisits2test),params.MAX_N_EPISODES,params.N_SIMULATIONS);
probReplayR_by_numVisits = nan(numel(nVisits2test),params.MAX_N_EPISODES,params.N_SIMULATIONS);
probReplayAct_by_numVisits = nan(numel(nVisits2test),params.MAX_N_EPISODES,params.N_SIMULATIONS);
probReplayActF_by_numVisits = nan(numel(nVisits2test),params.MAX_N_EPISODES,params.N_SIMULATIONS);
probReplayActR_by_numVisits = nan(numel(nVisits2test),params.MAX_N_EPISODES,params.N_SIMULATIONS);
for k=1:params.N_SIMULATIONS
    for l=1:params.MAX_N_EPISODES
        % Remove walls and goal states
        visitByState = numVisits(:,l,k);
        validStates = find(params.maze==0); % can start at any non-wall state
        validStates = validStates(~ismember(validStates,sub2ind(size(params.maze),params.s_end(:,1),params.s_end(:,2)))); %... but remove the goal states from the list
        visitByState = visitByState(validStates);
        thisReplayCount_all = numReplay(validStates,l,k);
        thisReplayCount_forward = numReplayF(validStates,l,k);
        thisReplayCount_reverse = numReplayR(validStates,l,k);
        for v=1:numel(nVisits2test)
            probReplay_by_numVisits(v,l,k) = nanmean(thisReplayCount_all(visitByState==v)>0);
            probReplayF_by_numVisits(v,l,k) = nanmean(thisReplayCount_forward(visitByState==v)>0);
            probReplayR_by_numVisits(v,l,k) = nanmean(thisReplayCount_reverse(visitByState==v)>0);
        end
        
        vv = zeros(numel(params.maze),1);
        vv(validStates)=1;
        vv = repmat(vv,1,4);
        visitByAction = numTimesAct(:,l,k);
        visitByAction = visitByAction(logical(vv(:)));
        thisReplayActCount_all = numReplayAct(logical(vv(:)),l,k);
        thisReplayActCount_forward = numReplayActF(logical(vv(:)),l,k);
        thisReplayActCount_reverse = numReplayActR(logical(vv(:)),l,k);
        for v=1:numel(nVisits2test)
            probReplayAct_by_numVisits(v,l,k) = nanmean(thisReplayActCount_all(visitByAction==v)>0);
            probReplayActF_by_numVisits(v,l,k) = nanmean(thisReplayActCount_forward(visitByAction==v)>0);
            probReplayActR_by_numVisits(v,l,k) = nanmean(thisReplayActCount_reverse(visitByAction==v)>0);
        end
    end
end

figure(1); clf;
subplot(2,3,1);
plot(nVisits2test,nanmean(nanmean(probReplay_by_numVisits,2),3))
xlabel('Number of visits to a state within a lap');
ylabel('Probability that that state is replayed in that lap');
title('All significant replay events'); 
grid on;
subplot(2,3,2);
plot(nVisits2test,nanmean(nanmean(probReplayF_by_numVisits,2),3))
xlabel('Number of visits to a state within a lap');
ylabel('Probability that that state is replayed in that lap');
title('Considering only FORWARD events'); 
grid on;
subplot(2,3,3);
plot(nVisits2test,nanmean(nanmean(probReplayR_by_numVisits,2),3))
xlabel('Number of visits to a state within a lap');
ylabel('Probability that that state is replayed in that lap');
title('Considering only REVERSE events'); 
grid on;

subplot(2,3,4);
plot(nVisits2test,nanmean(nanmean(probReplayAct_by_numVisits,2),3))
xlabel('Number of times a given action is executed within a lap');
ylabel('Probability that that action is replayed in that lap');
title('All significant replay events'); 
grid on;
subplot(2,3,5);
plot(nVisits2test,nanmean(nanmean(probReplayActF_by_numVisits,2),3))
xlabel('Number of times a given action is executed within a lap');
ylabel('Probability that that action is replayed in that lap');
title('Considering only FORWARD events'); 
grid on;
subplot(2,3,6);
plot(nVisits2test,nanmean(nanmean(probReplayActR_by_numVisits,2),3))
xlabel('Number of times a given action is executed within a lap');
ylabel('Probability that that action is replayed in that lap');
title('Considering only REVERSE events'); 
grid on;

set(gcf,'Position',[221         383        1270         439])



%% EXPORT FIGURE
if saveBool
    save genFig_numVisitsByprobReplay_linearMaze.mat

    fh=findall(0,'type','figure');
    for i=1:numel(fh)
        figure(fh(i).Number);
        set(gca, 'Clipping', 'off');
        set(gcf, 'Clipping', 'off');
        set(gcf, 'renderer', 'painters');
        export_fig(['../Parts/' mfilename '_' num2str(fh(i).Number)], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    end
end
