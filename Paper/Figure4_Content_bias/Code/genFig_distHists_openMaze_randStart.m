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

% Build adjacency matrix of graph
A = zeros(numel(params.maze),numel(params.maze));
for s=1:size(nextState,1)
    for a=1:size(nextState,2)
        A(s,nextState(s,a)) = 1;
    end
end

% Find shortest path between all pairs of nodes
dist = nan(numel(params.maze),numel(params.maze));
for s1=1:size(nextState,1)
    for s2=1:size(nextState,1)
        dist(s1,s2) = graphshortestpath(sparse(A),s1,s2);
    end
end

dist2agentProb_byEpisode = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2agentProb_byEpisode_atReward = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2agentProb_byEpisode_elsewhere = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2rewProb_byEpisode = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2rewProb_byEpisode_atReward = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2rewProb_byEpisode_elsewhere = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);

dist2agentProb_byReplay = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2agentProb_byReplay_atReward = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2agentProb_byReplay_elsewhere = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2agentProb_byReplay_forward = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2agentProb_byReplay_reverse = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);

dist2rewProb_byReplay = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2rewProb_byReplay_atReward = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2rewProb_byReplay_elsewhere = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2rewProb_byReplay_forward = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2rewProb_byReplay_reverse = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);

dist2agentProb_start = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2agentProb_end = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2agentProb_start_forward = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2agentProb_end_forward = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2agentProb_start_reverse = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2agentProb_end_reverse = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);

dist2rewProb_start = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2rewProb_end = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2rewProb_start_forward = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2rewProb_end_forward = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2rewProb_start_reverse = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);
dist2rewProb_end_reverse = zeros(params.N_SIMULATIONS,max(dist(~isinf(dist)))+1);

for k=1:length(simData)
    fprintf('Simulation #%d\n',k);
    
    % Calculate the activation probability per episode
    lapNum = [0;simData(k).numEpisodes(1:end-1)] + 1;
    nEpi = max(lapNum);
    dist2agentCount_byEpisode = zeros(nEpi,max(dist(~isinf(dist)))+1);
    dist2agentCount_byEpisode_atReward = zeros(nEpi,max(dist(~isinf(dist)))+1);
    dist2agentCount_byEpisode_elsewhere = zeros(nEpi,max(dist(~isinf(dist)))+1);
    dist2rewCount_byEpisode = zeros(nEpi,max(dist(~isinf(dist)))+1);
    dist2rewCount_byEpisode_atReward = zeros(nEpi,max(dist(~isinf(dist)))+1);
    dist2rewCount_byEpisode_elsewhere = zeros(nEpi,max(dist(~isinf(dist)))+1);
    for e=1:nEpi
        episode_tsis = find(lapNum==e);
        for t=1:length(episode_tsis)
            if ~isempty(simData(k).replay.state{episode_tsis(t)})
                agentPos = simData(k).expList(episode_tsis(t),1);
                dist2agent = dist(agentPos,simData(k).replay.state{episode_tsis(t)});
                N = histc(dist2agent,-0.5:1:max(dist(~isinf(dist)))+0.5);
                dist2agentCount_byEpisode(e,:) = dist2agentCount_byEpisode(e,:) + N(1:(end-1));
                if agentPos==50
                    dist2agentCount_byEpisode_atReward(e,:) = dist2agentCount_byEpisode_atReward(e,:) + N(1:(end-1));
                else
                    dist2agentCount_byEpisode_elsewhere(e,:) = dist2agentCount_byEpisode_elsewhere(e,:) + N(1:(end-1));
                end
                
                rewPos = sub2ind(size(params.maze),params.s_end(1),params.s_end(2));
                dist2rew = dist(rewPos,simData(k).replay.state{episode_tsis(t)});
                N = histc(dist2rew,-0.5:1:max(dist(~isinf(dist)))+0.5);
                dist2rewCount_byEpisode(e,:) = dist2rewCount_byEpisode(e,:) + N(1:(end-1));
                if agentPos==50
                    dist2rewCount_byEpisode_atReward(e,:) = dist2rewCount_byEpisode_atReward(e,:) + N(1:(end-1));
                else
                    dist2rewCount_byEpisode_elsewhere(e,:) = dist2rewCount_byEpisode_elsewhere(e,:) + N(1:(end-1));
                end
            end
        end
    end
    % Average across episodes
    dist2agentProb_byEpisode(k,:) = nanmean(dist2agentCount_byEpisode ./ repmat(nansum(dist2agentCount_byEpisode,2),1,length(N)-1),1);
    dist2agentProb_byEpisode_atReward(k,:) = nanmean(dist2agentCount_byEpisode_atReward ./ repmat(nansum(dist2agentCount_byEpisode_atReward,2),1,length(N)-1),1);
    dist2agentProb_byEpisode_elsewhere(k,:) = nanmean(dist2agentCount_byEpisode_elsewhere ./ repmat(nansum(dist2agentCount_byEpisode_elsewhere,2),1,length(N)-1),1);
    dist2rewProb_byEpisode(k,:) = nanmean(dist2rewCount_byEpisode ./ repmat(nansum(dist2rewCount_byEpisode,2),1,length(N)-1),1);
    dist2rewProb_byEpisode_atReward(k,:) = nanmean(dist2rewCount_byEpisode_atReward ./ repmat(nansum(dist2rewCount_byEpisode_atReward,2),1,length(N)-1),1);
    dist2rewProb_byEpisode_elsewhere(k,:) = nanmean(dist2rewCount_byEpisode_elsewhere ./ repmat(nansum(dist2rewCount_byEpisode_elsewhere,2),1,length(N)-1),1);
    
    % Calculate the activation probability per replay event
    % Identify candidate replay events
    candidateEvents = find(cellfun('length',simData(k).replay.state)>=max(sum(params.maze(:)==0)*minFracCells,minNumCells));
    lapNum = [0;simData(k).numEpisodes(1:end-1)] + 1;
    lapNum_events = lapNum(candidateEvents);
    agentPos = simData(k).expList(candidateEvents,1);
    
    dist2agentCount_byReplay = nan(0,max(dist(~isinf(dist)))+1);
    dist2agentCount_byReplay_atReward = nan(0,max(dist(~isinf(dist)))+1);
    dist2agentCount_byReplay_elsewhere = nan(0,max(dist(~isinf(dist)))+1);
    dist2agentCount_byReplay_forward = nan(0,max(dist(~isinf(dist)))+1);
    dist2agentCount_byReplay_reverse = nan(0,max(dist(~isinf(dist)))+1);
    
    dist2rewCount_byReplay = nan(0,max(dist(~isinf(dist)))+1);
    dist2rewCount_byReplay_atReward = nan(0,max(dist(~isinf(dist)))+1);
    dist2rewCount_byReplay_elsewhere = nan(0,max(dist(~isinf(dist)))+1);
    dist2rewCount_byReplay_forward = nan(0,max(dist(~isinf(dist)))+1);
    dist2rewCount_byReplay_reverse = nan(0,max(dist(~isinf(dist)))+1);
    
    dist2agentCount_start = zeros(1,max(dist(~isinf(dist)))+1);
    dist2agentCount_end = zeros(1,max(dist(~isinf(dist)))+1);
    dist2agentCount_start_forward = zeros(1,max(dist(~isinf(dist)))+1);
    dist2agentCount_end_forward = zeros(1,max(dist(~isinf(dist)))+1);
    dist2agentCount_start_reverse = zeros(1,max(dist(~isinf(dist)))+1);
    dist2agentCount_end_reverse = zeros(1,max(dist(~isinf(dist)))+1);
    
    dist2rewCount_start = zeros(1,max(dist(~isinf(dist)))+1);
    dist2rewCount_end = zeros(1,max(dist(~isinf(dist)))+1);
    dist2rewCount_start_forward = zeros(1,max(dist(~isinf(dist)))+1);
    dist2rewCount_end_forward = zeros(1,max(dist(~isinf(dist)))+1);
    dist2rewCount_start_reverse = zeros(1,max(dist(~isinf(dist)))+1);
    dist2rewCount_end_reverse = zeros(1,max(dist(~isinf(dist)))+1);
    
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
                    dist2agent = dist(agentPos(e),replayState);
                    N = histc(dist2agent,-0.5:1:max(dist(~isinf(dist)))+0.5);
                    dist2agentCount_byReplay = [dist2agentCount_byReplay ; (N(1:(end-1))>0)];
                    dist2agentCount_start(dist(agentPos(e),replayState(1))+1) = dist2agentCount_start(dist(agentPos(e),replayState(1))+1) + 1;
                    dist2agentCount_end(dist(agentPos(e),replayState(end))+1) = dist2agentCount_end(dist(agentPos(e),replayState(end))+1) + 1;
                    if agentPos(e)==50
                        dist2agentCount_byReplay_atReward = [dist2agentCount_byReplay_atReward ; (N(1:(end-1))>0)];
                    else
                        dist2agentCount_byReplay_elsewhere = [dist2agentCount_byReplay_elsewhere ; (N(1:(end-1))>0)];
                    end
                    if replayDir{1}=='F'
                        dist2agentCount_byReplay_forward = [dist2agentCount_byReplay_forward ; (N(1:(end-1))>0)];
                        dist2agentCount_start_forward(dist(agentPos(e),replayState(1))+1) = dist2agentCount_start_forward(dist(agentPos(e),replayState(1))+1) + 1;
                        dist2agentCount_end_forward(dist(agentPos(e),replayState(end))+1) = dist2agentCount_end_forward(dist(agentPos(e),replayState(end))+1) + 1;
                    elseif replayDir{1}=='R'
                        dist2agentCount_byReplay_reverse = [dist2agentCount_byReplay_reverse ; (N(1:(end-1))>0)];
                        dist2agentCount_start_reverse(dist(agentPos(e),replayState(1))+1) = dist2agentCount_start_reverse(dist(agentPos(e),replayState(1))+1) + 1;
                        dist2agentCount_end_reverse(dist(agentPos(e),replayState(end))+1) = dist2agentCount_end_reverse(dist(agentPos(e),replayState(end))+1) + 1;
                    end
                    
                    rewPos = sub2ind(size(params.maze),params.s_end(1),params.s_end(2));
                    dist2rew = dist(rewPos,replayState);
                    N = histc(dist2rew,-0.5:1:max(dist(~isinf(dist)))+0.5);
                    dist2rewCount_byReplay = [dist2rewCount_byReplay ; (N(1:(end-1))>0)];
                    dist2rewCount_start(dist(rewPos,replayState(1))+1) = dist2rewCount_start(dist(rewPos,replayState(1))+1) + 1;
                    dist2rewCount_end(dist(rewPos,replayState(end))+1) = dist2rewCount_end(dist(rewPos,replayState(end))+1) + 1;
                    if agentPos(e)==50
                        dist2rewCount_byReplay_atReward = [dist2rewCount_byReplay_atReward ; (N(1:(end-1))>0)];
                    else
                        dist2rewCount_byReplay_elsewhere = [dist2rewCount_byReplay_elsewhere ; (N(1:(end-1))>0)];
                    end
                    if replayDir{1}=='F'
                        dist2rewCount_byReplay_forward = [dist2rewCount_byReplay_forward ; (N(1:(end-1))>0)];
                        dist2rewCount_start_forward(dist(rewPos,replayState(1))+1) = dist2rewCount_start_forward(dist(rewPos,replayState(1))+1) + 1;
                        dist2rewCount_end_forward(dist(rewPos,replayState(end))+1) = dist2rewCount_end_forward(dist(rewPos,replayState(end))+1) + 1;
                    elseif replayDir{1}=='R'
                        dist2rewCount_byReplay_reverse = [dist2rewCount_byReplay_reverse ; (N(1:(end-1))>0)];
                        dist2rewCount_start_reverse(dist(rewPos,replayState(1))+1) = dist2rewCount_start_reverse(dist(rewPos,replayState(1))+1) + 1;
                        dist2rewCount_end_reverse(dist(rewPos,replayState(end))+1) = dist2rewCount_end_reverse(dist(rewPos,replayState(end))+1) + 1;
                    end
                end
            end
        end
    end
    
    dist2agentProb_byReplay(k,:) = nanmean(dist2agentCount_byReplay ./ repmat(nansum(dist2agentCount_byReplay,2),1,length(N)-1));
    dist2agentProb_byReplay_atReward(k,:) = nanmean(dist2agentCount_byReplay_atReward ./ repmat(nansum(dist2agentCount_byReplay_atReward,2),1,length(N)-1));
    dist2agentProb_byReplay_elsewhere(k,:) = nanmean(dist2agentCount_byReplay_elsewhere ./ repmat(nansum(dist2agentCount_byReplay_elsewhere,2),1,length(N)-1));
    dist2agentProb_byReplay_forward(k,:) = nanmean(dist2agentCount_byReplay_forward ./ repmat(nansum(dist2agentCount_byReplay_forward,2),1,length(N)-1));
    dist2agentProb_byReplay_reverse(k,:) = nanmean(dist2agentCount_byReplay_reverse ./ repmat(nansum(dist2agentCount_byReplay_reverse,2),1,length(N)-1));
    
    dist2rewProb_byReplay(k,:) = nanmean(dist2rewCount_byReplay ./ repmat(nansum(dist2rewCount_byReplay,2),1,length(N)-1));
    dist2rewProb_byReplay_atReward(k,:) = nanmean(dist2rewCount_byReplay_atReward ./ repmat(nansum(dist2rewCount_byReplay_atReward,2),1,length(N)-1));
    dist2rewProb_byReplay_elsewhere(k,:) = nanmean(dist2rewCount_byReplay_elsewhere ./ repmat(nansum(dist2rewCount_byReplay_elsewhere,2),1,length(N)-1));
    dist2rewProb_byReplay_forward(k,:) = nanmean(dist2rewCount_byReplay_forward ./ repmat(nansum(dist2rewCount_byReplay_forward,2),1,length(N)-1));
    dist2rewProb_byReplay_reverse(k,:) = nanmean(dist2rewCount_byReplay_reverse ./ repmat(nansum(dist2rewCount_byReplay_reverse,2),1,length(N)-1));
    
    dist2agentProb_start(k,:) = dist2agentCount_start./nansum(dist2agentCount_start);
    dist2agentProb_end(k,:) = dist2agentCount_end./nansum(dist2agentCount_end);
    dist2agentProb_start_forward(k,:) = dist2agentCount_start_forward./nansum(dist2agentCount_start_forward);
    dist2agentProb_end_forward(k,:) = dist2agentCount_end_forward./nansum(dist2agentCount_end_forward);
    dist2agentProb_start_reverse(k,:) = dist2agentCount_start_reverse./nansum(dist2agentCount_start_reverse);
    dist2agentProb_end_reverse(k,:) = dist2agentCount_end_reverse./nansum(dist2agentCount_end_reverse);
    
    dist2rewProb_start(k,:) = dist2rewCount_start./nansum(dist2rewCount_start);
    dist2rewProb_end(k,:) = dist2rewCount_end./nansum(dist2rewCount_end);
    dist2rewProb_start_forward(k,:) = dist2rewCount_start_forward./nansum(dist2rewCount_start_forward);
    dist2rewProb_end_forward(k,:) = dist2rewCount_end_forward./nansum(dist2rewCount_end_forward);
    dist2rewProb_start_reverse(k,:) = dist2rewCount_start_reverse./nansum(dist2rewCount_start_reverse);
    dist2rewProb_end_reverse(k,:) = dist2rewCount_end_reverse./nansum(dist2rewCount_end_reverse);
end

% Build null models
wallsIdx = find(params.maze(:));
rewIdx = sub2ind(size(params.maze),params.s_end(1),params.s_end(2));
allStates = 1:numel(params.maze);
validStates = and(~ismember(allStates,wallsIdx),~ismember(allStates,rewIdx));

% Make list of agent's position during replay
dist2agentCount_null = zeros(17,1);
for k=1:length(simData)
    backupTime = find(cellfun('length',simData(k).replay.state)>0);
    distCount = nan(numel(backupTime),sum(validStates));
    for i=1:numel(backupTime)
        sti = simData(k).expList(backupTime(i),1); % Where the agent was when this backup happened
        distCount(i,:) = dist(sti,validStates);
    end
    dist2agentCount_null = dist2agentCount_null + histc(distCount(:),-0.5:1:15.5);
end
% Build null model of distances to agent
dist2agentProb_null = dist2agentCount_null(1:end-1)./sum(dist2agentCount_null);

% Build null model of distances to reward
dist2rewProb_null = histc(dist(rewIdx,validStates),-0.5:1:15.5);
dist2rewProb_null = dist2rewProb_null(1:end-1)./sum(dist2rewProb_null);



%% PLOT

%{
May plot the following variables:

dist2agentProb_byEpisode
dist2agentProb_byEpisode_atReward
dist2agentProb_byEpisode_elsewhere
dist2rewProb_byEpisode
dist2rewProb_byEpisode_atReward
dist2rewProb_byEpisode_elsewhere


dist2agentProb_byReplay
dist2agentProb_byReplay_atReward
dist2agentProb_byReplay_elsewhere
dist2agentProb_byReplay_forward
dist2agentProb_byReplay_reverse

dist2rewProb_byReplay
dist2rewProb_byReplay_atReward
dist2rewProb_byReplay_elsewhere
dist2rewProb_byReplay_forward
dist2rewProb_byReplay_reverse

dist2agentProb_start
dist2agentProb_end
dist2agentProb_start_forward
dist2agentProb_end_forward
dist2agentProb_start_reverse
dist2agentProb_end_reverse

dist2rewProb_start
dist2rewProb_end
dist2rewProb_start_forward
dist2rewProb_end_forward
dist2rewProb_start_reverse
dist2rewProb_end_reverse
%}

figure(1); clf
yMax = 0.6;

subplot(3,2,1);
h=bar(nanmean(dist2agentProb_byEpisode,1));
h.XData = 0:15; h.Parent.YTick = 0:0.1:yMax;
axis([-0.5 15.5 0 yMax]); grid on;
ylabel('Probability');
xlabel('Distance');
title('Agent to backup location (by Episode)')
hold on; plot(0:15,dist2agentProb_null)

subplot(3,2,2);
h=bar(nanmean(dist2rewProb_byEpisode,1));
h.XData = 0:15; h.Parent.YTick = 0:0.1:yMax;
axis([-0.5 15.5 0 yMax]); grid on;
ylabel('Probability');
xlabel('Distance');
title('Reward to backup location (by Episode)')
hold on; plot(0:15,dist2rewProb_null)

subplot(3,2,3);
h=bar(nanmean(dist2agentProb_start,1));
h.XData = 0:15; h.Parent.YTick = 0:0.1:yMax;
axis([-0.5 15.5 0 yMax]); grid on;
ylabel('Probability');
xlabel('Distance');
title('Agent to replay START (by Replay)')
hold on; plot(0:15,dist2agentProb_null)

subplot(3,2,4);
h=bar(nanmean(dist2rewProb_start,1));
h.XData = 0:15; h.Parent.YTick = 0:0.1:yMax;
axis([-0.5 15.5 0 yMax]); grid on;
ylabel('Probability');
xlabel('Distance');
title('Reward to replay START (by Replay)')
hold on; plot(0:15,dist2rewProb_null)

subplot(3,2,5);
h=bar(nanmean(dist2agentProb_end,1));
h.XData = 0:15; h.Parent.YTick = 0:0.1:yMax;
axis([-0.5 15.5 0 yMax]); grid on;
ylabel('Probability');
xlabel('Distance');
title('Agent to replay END (by Replay)')
hold on; plot(0:15,dist2agentProb_null)

subplot(3,2,6);
h=bar(nanmean(dist2rewProb_end,1));
h.XData = 0:15; h.Parent.YTick = 0:0.1:yMax;
axis([-0.5 15.5 0 yMax]); grid on;
ylabel('Probability');
xlabel('Distance');
title('Reward to replay END (by Replay)')
hold on; plot(0:15,dist2rewProb_null)


%% EXPORT FIGURE
if saveBool
    save genFig_distHists_openMaze_randStart.mat
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    set(gcf, 'renderer', 'painters');
    export_fig(['../Parts/' mfilename], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
end
