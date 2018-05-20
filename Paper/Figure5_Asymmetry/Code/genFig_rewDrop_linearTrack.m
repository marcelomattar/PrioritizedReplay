%% STATE-SPACE PARAMETERS
addpath('../../../');
clear;
setParams;
params.maze             = zeros(3,10); % zeros correspond to 'visitable' states
params.maze(2,:)        = 1; % wall
params.s_start          = [1,1;3,size(params.maze,2)]; % beginning state (in matrix notation)
params.s_start_rand     = false; % Start at random locations after reaching goal

params.s_end            = [1,size(params.maze,2);3,1]; % goal state (in matrix notation)
params.rewMag           = [1 0]; % reward magnitude (rows: locations; columns: values)
params.rewSTD           = [0.1 0]; % reward Gaussian noise (rows: locations; columns: values)
params.rewProb          = [0.5 0.5]; % probability of receiving each reward (columns: values)

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

saveStr = input('Do you want to produce figures (y/n)? ','s');
if strcmp(saveStr,'y')
    saveBool = true;
else
    saveBool = false;
end


%% RUN SIMULATION
rng(mean('replay'));
for k=1:params.N_SIMULATIONS
    simData(k) = replaySim(params);
end


%% ANALYSIS PARAMETERS
minNumCells = 5;
minFracCells = 0;
runPermAnalysis = true; % Run permutation analysis (true or false)
nPerm = 500; % Number of permutations for assessing significance of an event


%% INITIALIZE VARIABLES
forwardCount_baseline = zeros(length(simData),numel(params.maze));
reverseCount_baseline = zeros(length(simData),numel(params.maze));
forwardCount_rewShift = zeros(length(simData),numel(params.maze));
reverseCount_rewShift = zeros(length(simData),numel(params.maze));
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
                    reward_tsi = ismember(simData(k).expList(1:candidateEvents(e),4),sub2ind(size(params.maze),params.s_end(:,1),params.s_end(:,2)));
                    lastReward_tsi = find(reward_tsi,1,'last'); % reward_tsi identifies the timepoint corresponding to the last reward received (at or prior to the current chunk)
                    lastReward_mag = simData(k).expList(lastReward_tsi,3); % lastReward_mag is the magnitude of the last reward received, prior to this chunk
                    if replayDir{1}=='F'
                        if abs(lastReward_mag-params.rewMag(1))<abs(lastReward_mag-params.rewMag(2)) % If this reward is closer to 1 than it is to 0
                            forwardCount_baseline(k,agentPos(e)) = forwardCount_baseline(k,agentPos(e)) + 1;
                        else % If this reward is closer to 0 than it is to 1
                            forwardCount_rewShift(k,agentPos(e)) = forwardCount_rewShift(k,agentPos(e)) + 1;
                        end
                    elseif replayDir{1}=='R'
                        if abs(lastReward_mag-params.rewMag(1))<abs(lastReward_mag-params.rewMag(2)) % If this reward is closer to 1 than it is to 0
                            reverseCount_baseline(k,agentPos(e)) = reverseCount_baseline(k,agentPos(e)) + 1;
                        else % If this reward is closer to 0 than it is to 1
                            reverseCount_rewShift(k,agentPos(e)) = reverseCount_rewShift(k,agentPos(e)) + 1;
                        end
                    end
                end
            end
        end
    end
end

numEpisodes_baseline = params.MAX_N_EPISODES*params.rewProb(1);
numEpisodes_rewShift = params.MAX_N_EPISODES*params.rewProb(2);
preplayF_baseline = nansum([forwardCount_baseline(:,1),forwardCount_baseline(:,30)],2) ./ numEpisodes_baseline;
replayF_baseline = nansum([forwardCount_baseline(:,6),forwardCount_baseline(:,25)],2) ./ numEpisodes_baseline;
preplayR_baseline = nansum([reverseCount_baseline(:,1),reverseCount_baseline(:,30)],2) ./ numEpisodes_baseline;
replayR_baseline = nansum([reverseCount_baseline(:,6),reverseCount_baseline(:,25)],2) ./ numEpisodes_baseline;

preplayF_rewShift = nansum([forwardCount_rewShift(:,1),forwardCount_rewShift(:,30)],2) ./ numEpisodes_rewShift;
replayF_rewShift = nansum([forwardCount_rewShift(:,6),forwardCount_rewShift(:,25)],2) ./ numEpisodes_rewShift;
preplayR_rewShift = nansum([reverseCount_rewShift(:,1),reverseCount_rewShift(:,30)],2) ./ numEpisodes_rewShift;
replayR_rewShift = nansum([reverseCount_rewShift(:,6),reverseCount_rewShift(:,25)],2) ./ numEpisodes_rewShift;


%% PLOT

% Forward-vs-Reverse
figure(1); clf;
subplot(1,3,1);
f1 = bar([nanmean(preplayF_baseline) nanmean(replayF_baseline) ; nanmean(preplayR_baseline) nanmean(replayR_baseline)]);
legend({'Preplay','Replay'},'Location','NortheastOutside');
f1(1).FaceColor=[1 1 1]; % Replay bar color
f1(1).LineWidth=1;
f1(2).FaceColor=[0 0 0]; % Replay bar color
f1(2).LineWidth=1;
set(f1(1).Parent,'XTickLabel',{'Forward correlated','Reverse correlated'});
ylim([0 1]);
ylabel('Events/Lap');
grid on
title('Baseline (1x)');

subplot(1,3,2);
f1 = bar([nanmean(preplayF_rewShift) nanmean(replayF_rewShift) ; nanmean(preplayR_rewShift) nanmean(replayR_rewShift)]);
legend({'Preplay','Replay'},'Location','NortheastOutside');
f1(1).FaceColor=[1 1 1]; % Replay bar color
f1(1).LineWidth=1;
f1(2).FaceColor=[0 0 0]; % Replay bar color
f1(2).LineWidth=1;
set(f1(1).Parent,'XTickLabel',{'Forward correlated','Reverse correlated'});
ylim([0 1]);
ylabel('Events/Lap');
grid on
title('Reward drop (0x)');

subplot(1,3,3);
F_baseline = nanmean(replayF_baseline+preplayF_baseline);
F_rewShift = nanmean(replayF_rewShift+preplayF_rewShift);
R_baseline = nanmean(replayR_baseline+preplayR_baseline);
R_rewShift = nanmean(replayR_rewShift+preplayR_rewShift);
load('genFig_FvsR_linearTrack.mat','preplayF','replayF','preplayR','replayR')
F_orig = nanmean(replayF+preplayF);
R_orig = nanmean(replayR+preplayR);
f1 = bar([100*((F_baseline/F_orig)-1) 100*((F_rewShift/F_orig)-1) ; 100*((R_baseline/R_orig)-1) 100*((R_rewShift/R_orig)-1)]);
legend({'Unchanged','0x reward'},'Location','NortheastOutside');
set(f1(1).Parent,'XTickLabel',{'Forward correlated','Reverse correlated'});
ylim([-200 200]);
grid on
title('Changes from baseline');

set(gcf,'Position',[234         908        1651         341])


%% Replication of Ambrose et al (2016)
% Equal reward
load('genFig_FvsR_linearTrack.mat','forwardCount','reverseCount')
eqRew_F1 = nansum([forwardCount(:,1),forwardCount(:,6)],2);
eqRew_F2 = nansum([forwardCount(:,25),forwardCount(:,30)],2);
eqRew_R1 = nansum([reverseCount(:,1),reverseCount(:,6)],2);
eqRew_R2 = nansum([reverseCount(:,25),reverseCount(:,30)],2);
% Reward shift
diffRew_F1 = preplayF_baseline+replayF_baseline; % Number of forward sequences on 1x trials
diffRew_F2 = preplayF_rewShift+replayF_rewShift; % Number of forward sequences on 0x trials
diffRew_R1 = preplayR_baseline+replayR_baseline; % Number of reverse sequences on 1x trials
diffRew_R2 = preplayR_rewShift+replayR_rewShift; % Number of reverse sequences on 0x trials
% Calculate percent differences
F_1x1x = 100 * ((nanmean(eqRew_F2)-nanmean(eqRew_F1)) / nanmean(eqRew_F1));
F_1x0x = 100 * ((nanmean(diffRew_F2)-nanmean(diffRew_F1)) / nanmean(diffRew_F1));
R_1x1x = 100 * ((nanmean(eqRew_R2)-nanmean(eqRew_R1)) / nanmean(eqRew_R1));
R_1x0x = 100 * ((nanmean(diffRew_R2)-nanmean(diffRew_R1)) / nanmean(diffRew_R1));

figure(2); clf;
subplot(2,2,1);
f1 = bar([F_1x1x F_1x0x]);
ylim([-100 200]);
ylabel('Simulation')
grid on
title('Forward replay');
subplot(2,2,2);
f2 = bar([R_1x1x R_1x0x]);
ylim([-100 200]);
grid on
title('Reverse replay');

subplot(2,2,3);
f1 = bar([13 -83]);
ylim([-100 200]);
ylabel('Ambrose et al (2016)')
grid on
title('Forward replay');
subplot(2,2,4);
f2 = bar([-1 -88]);
ylim([-100 200]);
grid on
title('Reverse replay');
set(gcf,'Position',[463   342   918   504])


%% Changes in forward and reverse replay
figure(3); clf;
deltaF_baseline = [F_orig F_baseline];
deltaF_rewShift = [F_orig F_rewShift];
deltaR_baseline = [R_orig R_baseline];
deltaR_rewShift = [R_orig R_rewShift];

f1=bar([R_orig R_baseline R_rewShift]);
f1.FaceColor=[1 1 1];
f1.LineWidth=1;
set(f1(1).Parent,'XTickLabel',{'1x/1x','1x','0x'});
ylim([0 1]);
ylabel('Events/Lap');
grid on


%% EXPORT FIGURE
if saveBool
    save genFig_rewDrop_linearTrack.mat
    
    % Set clipping off
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    
    set(gcf, 'renderer', 'painters');
    export_fig(['../Parts/' mfilename], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    %print(filename,'-dpdf','-fillpage')
end


