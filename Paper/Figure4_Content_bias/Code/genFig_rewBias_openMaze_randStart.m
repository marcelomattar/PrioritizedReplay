load_existing_data = true;
addpath('../../../');

if load_existing_data
    %load('../../Figure3_FvsR_balance/Code/genFig_FvsR_openMaze.mat','simData','params')
    %params.N_SIMULATIONS = 100;
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

actProb_byEpisode = zeros(numel(params.maze),params.N_SIMULATIONS);
actProb_byEpisode_atReward = zeros(numel(params.maze),params.N_SIMULATIONS);
actProb_byEpisode_elsewhere = zeros(numel(params.maze),params.N_SIMULATIONS);

actProb_byReplay = zeros(numel(params.maze),params.N_SIMULATIONS);
actProb_byReplay_atReward = zeros(numel(params.maze),params.N_SIMULATIONS);
actProb_byReplay_elsewhere = zeros(numel(params.maze),params.N_SIMULATIONS);
actProb_byReplay_forward = zeros(numel(params.maze),params.N_SIMULATIONS);
actProb_byReplay_reverse = zeros(numel(params.maze),params.N_SIMULATIONS);

actProb_start = zeros(numel(params.maze),params.N_SIMULATIONS);
actProb_end = zeros(numel(params.maze),params.N_SIMULATIONS);
actProb_start_forward = zeros(numel(params.maze),params.N_SIMULATIONS);
actProb_end_forward = zeros(numel(params.maze),params.N_SIMULATIONS);
actProb_start_reverse = zeros(numel(params.maze),params.N_SIMULATIONS);
actProb_end_reverse = zeros(numel(params.maze),params.N_SIMULATIONS);
for k=1:params.N_SIMULATIONS
    fprintf('Simulation #%d\n',k);
    
    % Calculate the activation probability per episode
    lapNum = [0;simData(k).numEpisodes(1:end-1)] + 1;
    nEpi = max(lapNum);
    actCount_byEpisode = zeros(nEpi,numel(params.maze));
    actCount_byEpisode_atReward = zeros(nEpi,numel(params.maze));
    actCount_byEpisode_elsewhere = zeros(nEpi,numel(params.maze));
    for e=1:nEpi
        episode_tsis = find(lapNum==e);
        for t=1:length(episode_tsis)
            if ~isempty(simData(k).replay.state{episode_tsis(t)})
                N = histc(simData(k).replay.state{episode_tsis(t)},0.5:1:(numel(params.maze)+0.5));
                actCount_byEpisode(e,:) = actCount_byEpisode(e,:) + N(1:(end-1)); % cumulative sum across timepoints in this episode
                
                agentPos = simData(k).expList(episode_tsis(t),1);
                if agentPos==50
                    actCount_byEpisode_atReward(e,:) = actCount_byEpisode_atReward(e,:) + N(1:(end-1));
                else
                    actCount_byEpisode_elsewhere(e,:) = actCount_byEpisode_elsewhere(e,:) + N(1:(end-1));
                end
            end
        end
    end
    actProb_byEpisode(:,k) = nanmean(actCount_byEpisode>0,1)'; % Average across episodes
    actProb_byEpisode_atReward(:,k) = nanmean(actCount_byEpisode_atReward>0,1)'; % Average across episodes
    actProb_byEpisode_elsewhere(:,k) = nanmean(actCount_byEpisode_elsewhere>0,1)'; % Average across episodes
    
    
    % Calculate the activation probability per replay event
    % Identify candidate replay events
    candidateEvents = find(cellfun('length',simData(k).replay.state)>=max(sum(params.maze(:)==0)*minFracCells,minNumCells));
    lapNum = [0;simData(k).numEpisodes(1:end-1)] + 1;
    lapNum_events = lapNum(candidateEvents);
    agentPos = simData(k).expList(candidateEvents,1);
    actCount_byReplay = nan(0,numel(params.maze));
    actCount_byReplay_atReward = nan(0,numel(params.maze));
    actCount_byReplay_elsewhere = nan(0,numel(params.maze));
    actCount_byReplay_forward = nan(0,numel(params.maze));
    actCount_byReplay_reverse = nan(0,numel(params.maze));
    actCount_start = zeros(numel(params.maze),1);
    actCount_end = zeros(numel(params.maze),1);
    actCount_start_forward = zeros(numel(params.maze),1);
    actCount_end_forward = zeros(numel(params.maze),1);
    actCount_start_reverse = zeros(numel(params.maze),1);
    actCount_end_reverse = zeros(numel(params.maze),1);
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
                    N = histc(replayState,0.5:1:(numel(params.maze)+0.5));
                    actCount_byReplay = [actCount_byReplay ; (N(1:(end-1))>0)];
                    if agentPos(e)==50
                        actCount_byReplay_atReward = [actCount_byReplay_atReward ; (N(1:(end-1))>0)];
                    else
                        actCount_byReplay_elsewhere = [actCount_byReplay_elsewhere ; (N(1:(end-1))>0)];
                    end
                    if replayDir{1}=='F'
                        %if simData(k).numEpisodes(candidateEvents(e))<5
                        actCount_byReplay_forward = [actCount_byReplay_forward ; (N(1:(end-1))>0)];
                        actCount_start_forward(replayState(1)) = actCount_start_forward(replayState(1)) + 1;
                        actCount_end_forward(replayState(end)) = actCount_end_forward(replayState(end)) + 1;
                        %end
                    elseif replayDir{1}=='R'
                        %if simData(k).numEpisodes(candidateEvents(e))<5
                        actCount_byReplay_reverse = [actCount_byReplay_reverse ; (N(1:(end-1))>0)];
                        actCount_start_reverse(replayState(1)) = actCount_start_reverse(replayState(1)) + 1;
                        actCount_end_reverse(replayState(end)) = actCount_end_reverse(replayState(end)) + 1;
                        %end
                    end
                    actCount_start(replayState(1)) = actCount_start(replayState(1)) + 1;
                    actCount_end(replayState(end)) = actCount_end(replayState(end)) + 1;
                end
            end
        end
    end
    actProb_byReplay(:,k) = nanmean(actCount_byReplay>0,1)'; % Average across episodes
    actProb_byReplay_atReward(:,k) = nanmean(actCount_byReplay_atReward>0,1)'; % Average across episodes
    actProb_byReplay_elsewhere(:,k) = nanmean(actCount_byReplay_elsewhere>0,1)'; % Average across episodes
    actProb_byReplay_forward(:,k) = nanmean(actCount_byReplay_forward>0,1)'; % Average across episodes
    actProb_byReplay_reverse(:,k) = nanmean(actCount_byReplay_reverse>0,1)'; % Average across episodes
    
    actProb_start(:,k) = actCount_start ./ sum(actCount_start);
    actProb_end(:,k) = actCount_end ./ sum(actCount_end);
    actProb_start_forward(:,k) = actCount_start_forward ./ sum(actCount_start_forward);
    actProb_end_forward(:,k) = actCount_end_forward ./ sum(actCount_end_forward);
    actProb_start_reverse(:,k) = actCount_start_reverse ./ sum(actCount_start_reverse);
    actProb_end_reverse(:,k) = actCount_end_reverse ./ sum(actCount_end_reverse);
end


%% PLOT

% Define colormap: black-red-yellow
map1 = [(0:0.01:1)' zeros(size((0:0.01:1)')) zeros(size((0:0.01:1)'))];
map2 = [ones(size((0:0.01:1)')) (0:0.01:1)' zeros(size((0:0.01:1)'))];
map = [map1;map2];


figure(1); clf;
%caxlim = roundn(max([nanmean(actProb_byEpisode,2);nanmean(actProb_byReplay,2)]),-1);
caxlim = 0.5;

subplot(3,2,1);
imagesc(reshape(nanmean(actProb_byEpisode,2),6,9))
axis equal
colormap(map);
hcb=colorbar;
ylabel(hcb, 'Probability');
title('Activation probability (per Episode)')
ylabel('Everywhere')
caxis([0 caxlim])

subplot(3,2,2);
imagesc(reshape(nanmean(actProb_byReplay,2),6,9))
axis equal
colormap(map);
hcb=colorbar;
ylabel(hcb, 'Probability');
title('Activation probability (per Replay event)')
ylabel('Everywhere')
caxis([0 caxlim])

subplot(3,2,3);
imagesc(reshape(nanmean(actProb_byEpisode_atReward,2),6,9))
axis equal
colormap(map);
hcb=colorbar;
ylabel(hcb, 'Probability');
title('Activation probability (per Episode)')
ylabel('at Reward site (i.e. 2,9)')
caxis([0 caxlim])

subplot(3,2,4);
imagesc(reshape(nanmean(actProb_byReplay_atReward,2),6,9))
axis equal
colormap(map);
hcb=colorbar;
ylabel(hcb, 'Probability');
title('Activation probability (per Replay event)')
ylabel('at Reward site (i.e. 2,9)')
caxis([0 caxlim])

subplot(3,2,5);
imagesc(reshape(nanmean(actProb_byEpisode_elsewhere,2),6,9))
axis equal
colormap(map);
hcb=colorbar;
ylabel(hcb, 'Probability');
title('Activation probability (per Episode)')
ylabel('Elsewhere (all but 2,9)')
caxis([0 caxlim])

subplot(3,2,6);
imagesc(reshape(nanmean(actProb_byReplay_elsewhere,2),6,9))
axis equal
colormap(map);
hcb=colorbar;
ylabel(hcb, 'Probability');
title('Activation probability (per Replay event)')
ylabel('Elsewhere (all but 2,9)')
caxis([0 caxlim])

set(gcf,'Position',[215         199        1493        1115])




figure(2); clf;
set(gcf,'Position',[1154         403        1188         286])
%caxlim = roundn(max([nanmean(actProb_byReplay_forward,2);nanmean(actProb_byReplay_reverse,2)]),-1);
caxlim = 0.5;
subplot(1,2,1);
imagesc(reshape(nanmean(actProb_byReplay_forward,2),6,9))
colormap(map);
hcb=colorbar;
ylabel(hcb, 'Probability');
title('Activation probability (Forward)')
caxis([0 caxlim])
axis equal
subplot(1,2,2);
imagesc(reshape(nanmean(actProb_byReplay_reverse,2),6,9))
colormap(map);
hcb=colorbar;
ylabel(hcb, 'Probability');
title('Activation probability (Reverse)')
caxis([0 caxlim])
axis equal




figure(3); clf;
caxlim = 0.1;
subplot(3,2,1);
imagesc(reshape(nanmean(actProb_start,2),6,9))
axis equal
colormap(map);
hcb=colorbar;
ylabel(hcb, 'Probability');
ylabel('Forward and Reverse');
title('Probability of START (sum over squares = 1)')
caxis([0 caxlim])

subplot(3,2,2);
imagesc(reshape(nanmean(actProb_end,2),6,9))
axis equal
colormap(map);
hcb=colorbar;
ylabel(hcb, 'Probability');
ylabel('Forward and Reverse');
title('Probability of END (sum over squares = 1)')
caxis([0 caxlim])

subplot(3,2,3);
imagesc(reshape(nanmean(actProb_start_forward,2),6,9))
axis equal
colormap(map);
hcb=colorbar;
ylabel(hcb, 'Probability');
ylabel('Forward only');
title('Probability of START (sum over squares = 1)')
caxis([0 caxlim])

subplot(3,2,4);
imagesc(reshape(nanmean(actProb_end_forward,2),6,9))
axis equal
colormap(map);
hcb=colorbar;
ylabel(hcb, 'Probability');
ylabel('Forward only');
title('Probability of END (sum over squares = 1)')
caxis([0 caxlim])

subplot(3,2,5);
imagesc(reshape(nanmean(actProb_start_reverse,2),6,9))
axis equal
colormap(map);
hcb=colorbar;
ylabel(hcb, 'Probability');
ylabel('Reverse only');
title('Probability of START (sum over squares = 1)')
caxis([0 caxlim])

subplot(3,2,6);
imagesc(reshape(nanmean(actProb_end_reverse,2),6,9))
axis equal
colormap(map);
hcb=colorbar;
ylabel(hcb, 'Probability');
ylabel('Reverse only');
title('Probability of END (sum over squares = 1)')
caxis([0 caxlim])

set(gcf,'Position',[215         199        1493        1115])


%% EXPORT FIGURE
if saveBool
    save genFig_rewBias_openMaze_randStart.mat
    for f=1:3
        fh = figure(f);
        set(gca, 'Clipping', 'off');
        set(gcf, 'Clipping', 'off');
        set(gcf, 'renderer', 'painters');
        %export_fig(['../Parts/' mfilename '_' num2str(f)], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
        print(['../Parts/' mfilename '_' num2str(f)],'-dpdf','-fillpage')
    end
end
