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
    %params.s_end            = [1,9;6,9]; % goal state (in matrix notation)
    params.s_end            = [1,9]; % goal state (in matrix notation)
    params.s_start          = [3,1]; % beginning state (in matrix notation)
    params.s_start_rand     = true; % Start at random locations after reaching goal
    
    %% OVERWRITE PARAMETERS
    params.N_SIMULATIONS    = 1000; % number of times to run the simulation
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
forwardCount = zeros(length(simData),params.MAX_N_EPISODES); % Number of forward events per episode
reverseCount = zeros(length(simData),params.MAX_N_EPISODES); % Number of forward events per episode
dirScoreForward = zeros(params.N_SIMULATIONS,params.MAX_N_EPISODES); % Direction score for forward events per episode
dirScoreReverse = zeros(params.N_SIMULATIONS,params.MAX_N_EPISODES); % Direction score for reverse events per episode
actProb_byEpisode = zeros(numel(params.maze),params.MAX_N_EPISODES,params.N_SIMULATIONS); % Activation probability per episode
avGain_byEpisode = zeros(params.MAX_N_EPISODES,params.N_SIMULATIONS); % Average gain per episode
avNeed_byEpisode = zeros(params.MAX_N_EPISODES,params.N_SIMULATIONS); % Average need per episode

actCount_byReplay = cell(params.MAX_N_EPISODES,params.N_SIMULATIONS); % Activation probability per replay
actCount_byReplay_forward = cell(params.MAX_N_EPISODES,params.N_SIMULATIONS); % Activation probability per forward replay
actCount_byReplay_reverse = cell(params.MAX_N_EPISODES,params.N_SIMULATIONS); % Activation probability per reverse replay
actProb_byReplay = zeros(numel(params.maze),params.MAX_N_EPISODES,params.N_SIMULATIONS); % Activation probability per replay
actProb_byReplay_forward = zeros(numel(params.maze),params.MAX_N_EPISODES,params.N_SIMULATIONS); % Activation probability per forward replay
actProb_byReplay_reverse = zeros(numel(params.maze),params.MAX_N_EPISODES,params.N_SIMULATIONS); % Activation probability per reverse replay

gain_byReplay = cell(params.MAX_N_EPISODES,params.N_SIMULATIONS); % List of gain by Replay event
gain_byReplay_forward = cell(params.MAX_N_EPISODES,params.N_SIMULATIONS); % List of gain by forward event
gain_byReplay_reverse = cell(params.MAX_N_EPISODES,params.N_SIMULATIONS); % List of gain by reverse event
need_byReplay = cell(params.MAX_N_EPISODES,params.N_SIMULATIONS); % List of need by Replay event
need_byReplay_forward = cell(params.MAX_N_EPISODES,params.N_SIMULATIONS); % List of need by forward event
need_byReplay_reverse = cell(params.MAX_N_EPISODES,params.N_SIMULATIONS); % List of need by reverse event

avGain_byReplay = nan(params.MAX_N_EPISODES,params.N_SIMULATIONS); % Average gain by Replay event
avGain_byReplay_forward = nan(params.MAX_N_EPISODES,params.N_SIMULATIONS); % Average gain by forward event
avGain_byReplay_reverse = nan(params.MAX_N_EPISODES,params.N_SIMULATIONS); % Average gain by reverse event
avNeed_byReplay = nan(params.MAX_N_EPISODES,params.N_SIMULATIONS); % Average need by Replay event
avNeed_byReplay_forward = nan(params.MAX_N_EPISODES,params.N_SIMULATIONS); % Average need by forward event
avNeed_byReplay_reverse = nan(params.MAX_N_EPISODES,params.N_SIMULATIONS); % Average need by reverse event

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

for k=1:length(simData)
    fprintf('Simulation #%d\n',k);
    
    % Calculate the activation probability per episode
    lapNum = [0;simData(k).numEpisodes(1:end-1)] + 1;
    nEpi = max(lapNum);
    actCount_byEpisode = zeros(nEpi,numel(params.maze));
    for e=1:nEpi
        episode_tsis = find(lapNum==e);
        gainList_byEpisode = zeros(0,1);
        needList_byEpisode = zeros(0,1);
        for t=1:length(episode_tsis)
            if ~isempty(simData(k).replay.state{episode_tsis(t)})
                N = histc(simData(k).replay.state{episode_tsis(t)},0.5:1:(numel(params.maze)+0.5));
                actCount_byEpisode(e,:) = actCount_byEpisode(e,:) + N(1:(end-1)); % cumulative sum across timepoints in this episode
                gainList_byEpisode = [gainList_byEpisode;simData(k).replay.gain{episode_tsis(t)}];
                needList_byEpisode = [needList_byEpisode;simData(k).replay.need{episode_tsis(t)}];
            end
        end
        %avGain_byEpisode(e,k) = nanmean(gainList_byEpisode);
        avGain_byEpisode(e,k) = nanmean(cellfun(@sum,gainList_byEpisode));
        %avNeed_byEpisode(e,k) = nanmean(needList_byEpisode);
        avNeed_byEpisode(e,k) = nanmean(cellfun(@(v) v(end), needList_byEpisode));
    end
    actProb_byEpisode(:,:,k) = (actCount_byEpisode>0)';
    
    
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
                replayDir = eventDir(thisChunk);
                replayState = eventState([thisChunk (thisChunk(end)+1)]);
                replayAction = eventAction([thisChunk (thisChunk(end)+1)]);
                replayGain = simData(k).replay.gain{candidateEvents(e)}; replayGain = replayGain([thisChunk (thisChunk(end)+1)]);
                replayNeed = simData(k).replay.need{candidateEvents(e)}; replayNeed = replayNeed([thisChunk (thisChunk(end)+1)]);
                
                % Assess the significance of this event
                %allPerms = cell2mat(arrayfun(@(x)randperm(length(replayState)),(1:nPerm)','UniformOutput',0));
                sigBool = true; %#ok<NASGU>
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
                    N = histc(replayState,0.5:1:(numel(params.maze)+0.5));
                    actCount_byReplay{lapNum_events(e),k} = [actCount_byReplay{lapNum_events(e),k} ; N(1:(end-1))];
                    gain_byReplay{lapNum_events(e),k} = [gain_byReplay{lapNum_events(e),k} ; cellfun(@sum,replayGain)];
                    need_byReplay{lapNum_events(e),k} = [need_byReplay{lapNum_events(e),k} ; cellfun(@(v) v(end), replayNeed)];
                    
                    if replayDir{1}=='F'
                        forwardCount(k,lapNum_events(e)) = forwardCount(k,lapNum_events(e)) + 1;
                        dirScoreForward(k,lapNum_events(e)) = dirScoreForward(k,lapNum_events(e)) + disScore;
                        actCount_byReplay_forward{lapNum_events(e),k} = [actCount_byReplay_forward{lapNum_events(e),k} ; N(1:(end-1))];
                        gain_byReplay_forward{lapNum_events(e),k} = [gain_byReplay_forward{lapNum_events(e),k} ; cellfun(@sum,replayGain)];
                        need_byReplay_forward{lapNum_events(e),k} = [need_byReplay_forward{lapNum_events(e),k} ; cellfun(@(v) v(end), replayNeed)];
                        
                    elseif replayDir{1}=='R'
                        reverseCount(k,lapNum_events(e)) = reverseCount(k,lapNum_events(e)) + 1;
                        dirScoreReverse(k,lapNum_events(e)) = dirScoreReverse(k,lapNum_events(e)) + disScore;
                        actCount_byReplay_reverse{lapNum_events(e),k} = [actCount_byReplay_reverse{lapNum_events(e),k} ; N(1:(end-1))];
                        gain_byReplay_reverse{lapNum_events(e),k} = [gain_byReplay_reverse{lapNum_events(e),k} ; cellfun(@sum,replayGain)];
                        need_byReplay_reverse{lapNum_events(e),k} = [need_byReplay_reverse{lapNum_events(e),k} ; cellfun(@(v) v(end), replayNeed)];
                    end
                end
            end
        end
    end
    
    for ep=1:params.MAX_N_EPISODES
        if size(actCount_byReplay{ep,k},1)>0
            actProb_byReplay(:,ep,k) = nanmean(actCount_byReplay{ep,k}>0,1);
            avGain_byReplay(ep,k) = nanmean(gain_byReplay{ep,k});
            avNeed_byReplay(ep,k) = nanmean(need_byReplay{ep,k});
        end
        if size(actCount_byReplay_forward{ep,k},1)>0
            actProb_byReplay_forward(:,ep,k) = nanmean(actCount_byReplay_forward{ep,k}>0,1);
            avGain_byReplay_forward(ep,k) = nanmean(gain_byReplay_forward{ep,k});
            avNeed_byReplay_forward(ep,k) = nanmean(need_byReplay_forward{ep,k});
        end
        if size(actCount_byReplay_reverse{ep,k},1)>0
            actProb_byReplay_reverse(:,ep,k) = nanmean(actCount_byReplay_reverse{ep,k}>0,1);
            avGain_byReplay_reverse(ep,k) = nanmean(gain_byReplay_reverse{ep,k});
            avNeed_byReplay_reverse(ep,k) = nanmean(need_byReplay_reverse{ep,k});
        end
    end
end


%% PLOT RESULTS

% 1. Probability of replay
figure(1); clf;
subplot(2,3,1);
h=plot(1:50,nanmean(forwardCount+reverseCount,1));
set(h,'LineWidth',1)
ylim([0 3]); xlim([0 50]); grid on;
title('Number of replay events')
ylabel('Count (within episode)');
xlabel('Episode');
subplot(2,3,2);
h=plot(1:50,nanmean(forwardCount,1));
set(h,'LineWidth',1)
ylim([0 3]); xlim([0 50]); grid on;
title('Number of FORWARD events')
ylabel('Count (within episode)');
xlabel('Episode');
subplot(2,3,3);
h=plot(1:50,nanmean(reverseCount,1));
set(h,'LineWidth',1)
ylim([0 1]); xlim([0 50]); grid on;
title('Number of REVERSE events')
ylabel('Count (within episode)');
xlabel('Episode');
subplot(2,3,4);
h=plot(1:50,nanmean((forwardCount+reverseCount)>0,1));
set(h,'LineWidth',1)
ylim([0 1]); xlim([0 50]); grid on;
title('Probability of replay events')
ylabel('Probability (within episode)');
xlabel('Episode');
subplot(2,3,5);
h=plot(1:50,nanmean(forwardCount>0,1));
set(h,'LineWidth',1)
ylim([0 1]); xlim([0 50]); grid on;
title('Probability of FORWARD events')
ylabel('Probability (within episode)');
xlabel('Episode');
subplot(2,3,6);
h=plot(1:50,nanmean(reverseCount>0,1));
set(h,'LineWidth',1)
ylim([0 1]); xlim([0 50]); grid on;
title('Probability of REVERSE events')
ylabel('Probability (within episode)');
xlabel('Episode');
set(gcf,'Position',[1000         791        1182         547])


% 2. Activation probability

% Remove states that could not be replayed
validStates = find(params.maze==0); % can start at any non-wall state
validStates = validStates(~ismember(validStates,sub2ind(size(params.maze),params.s_end(:,1),params.s_end(:,2)))); %... but remove the goal states from the list
actProb_byReplay = actProb_byReplay(validStates,:,:);
actProb_byReplay_forward = actProb_byReplay_forward(validStates,:,:);
actProb_byReplay_reverse = actProb_byReplay_reverse(validStates,:,:);
actProb_byEpisode = actProb_byEpisode(validStates,:,:);

figure(2); clf;
subplot(2,2,1);
h=plot(1:50,nanmean(nanmean(actProb_byEpisode,3),1));
set(h,'LineWidth',1)
ylim([0 1]); xlim([0 50]); grid on;
title('Activation probability (per Episode)')
ylabel('Probability');
xlabel('Episode');
subplot(2,2,2);
h=plot(1:50,nanmean(nanmean(actProb_byReplay,3),1));
set(h,'LineWidth',1)
ylim([0 0.5]); xlim([0 50]); grid on;
title('Activation probability (per Replay event)')
ylabel('Probability');
xlabel('Episode');
subplot(2,2,3);
h=plot(1:50,nanmean(nanmean(actProb_byReplay_forward,3),1));
set(h,'LineWidth',1)
ylim([0 0.5]); xlim([0 50]); grid on;
title('Activation probability (per Forward event)')
ylabel('Probability');
xlabel('Episode');
subplot(2,2,4);
h=plot(1:50,nanmean(nanmean(actProb_byReplay_reverse,3),1));
set(h,'LineWidth',1)
ylim([0 0.5]); xlim([0 50]); grid on;
title('Activation probability (per Reverse event)')
ylabel('Probability');
xlabel('Episode');
set(gcf,'Position',[964   244   748   503])


% 3. Gain and Need
episodes2plot = 1:50;
figure(3); clf;
%avGain_byEpisode(avGain_byEpisode<(10*params.minGain))=nan;

subplot(4,2,1);
h=plot(episodes2plot,nanmean(avGain_byEpisode(episodes2plot,:),2));
set(h,'LineWidth',1)
xlim([0 50]); grid on;
title('Average Gain (per backup in general)')
ylabel('Gain');
xlabel('Episode');
subplot(4,2,2);
h=plot(episodes2plot,nanmean(avNeed_byEpisode(episodes2plot,:),2));
set(h,'LineWidth',1)
xlim([0 50]); grid on;
title('Average Need (per backup in general)')
ylabel('Need');
xlabel('Episode');
subplot(4,2,3);
h=plot(episodes2plot,nanmean(avGain_byReplay(episodes2plot,:),2));
set(h,'LineWidth',1)
xlim([0 50]); grid on;
title('Average Gain (per Replay backup)')
ylabel('Gain');
xlabel('Episode');
subplot(4,2,4);
h=plot(episodes2plot,nanmean(avNeed_byReplay(episodes2plot,:),2));
set(h,'LineWidth',1)
xlim([0 50]); grid on;
title('Average Need (per Replay backup)')
ylabel('Need');
xlabel('Episode');
subplot(4,2,5);
h=plot(episodes2plot,nanmean(avGain_byReplay_forward(episodes2plot,:),2));
set(h,'LineWidth',1)
xlim([0 50]); grid on;
title('Average Gain (per Forward Replay backup)')
ylabel('Gain');
xlabel('Episode');
subplot(4,2,6);
h=plot(episodes2plot,nanmean(avNeed_byReplay_forward(episodes2plot,:),2));
set(h,'LineWidth',1)
xlim([0 50]); grid on;
title('Average Need (per Forward Replay backup)')
ylabel('Need');
xlabel('Episode');
subplot(4,2,7);
h=plot(episodes2plot,nanmean(avGain_byReplay_reverse(episodes2plot,:),2));
set(h,'LineWidth',1)
xlim([0 50]); grid on;
title('Average Gain (per Reverse Replay backup)')
ylabel('Gain');
xlabel('Episode');
subplot(4,2,8);
h=plot(episodes2plot,nanmean(avNeed_byReplay_reverse(episodes2plot,:),2));
set(h,'LineWidth',1)
xlim([0 50]); grid on;
title('Average Need (per Reverse Replay backup)')
ylabel('Need');
xlabel('Episode');
set(gcf,'Position',[968         118         748        1022])



%% EXPORT FIGURE
if saveBool
    save genFig_replayByLaps_openMaze.mat

    fh=findall(0,'type','figure');
    for i=1:numel(fh)
        figure(fh(i).Number);
        set(gca, 'Clipping', 'off');
        set(gcf, 'Clipping', 'off');
        set(gcf, 'renderer', 'painters');
        export_fig(['../Parts/' mfilename '_' num2str(fh(i).Number)], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    end
end
