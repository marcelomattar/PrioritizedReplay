%% STATE-SPACE PARAMETERS
addpath('../../../');
clear;
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
params.N_SIMULATIONS    = 10000; % number of times to run the simulation
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

params.PLOT_STEPS       = false; % Plot each step of real experience
params.PLOT_Qvals       = false; % Plot Q-values
params.PLOT_PLANS       = false; % Plot each planning step
params.PLOT_EVM         = false; % Plot need and gain
params.PLOT_wait        = 11 ; % Number of full episodes completed before plotting

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
forwardCount = zeros(length(simData),numel(params.maze));
reverseCount = zeros(length(simData),numel(params.maze));
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
                    if replayDir{1}=='F'
                        forwardCount(k,agentPos(e)) = forwardCount(k,agentPos(e)) + 1;
                    elseif replayDir{1}=='R'
                        reverseCount(k,agentPos(e)) = reverseCount(k,agentPos(e)) + 1;
                    end
                end
            end
        end
    end
end

% Compute the number of significant events BEFORE (preplay) and AFTER (replay) an event (which could be larger than 1)
% PS: Notice that this is not a measure of the percent of episodes with a significant event (which would produce a smaller numbers)
preplayF = nansum([forwardCount(:,1),forwardCount(:,30)],2)./params.MAX_N_EPISODES;
replayF = nansum([forwardCount(:,6),forwardCount(:,25)],2)./params.MAX_N_EPISODES;
preplayR = nansum([reverseCount(:,1),reverseCount(:,30)],2)./params.MAX_N_EPISODES;
replayR = nansum([reverseCount(:,6),reverseCount(:,25)],2)./params.MAX_N_EPISODES;


%% PLOT

% Forward-vs-Reverse
figure(1); clf;
f1 = bar([nanmean(preplayF) nanmean(replayF) ; nanmean(preplayR) nanmean(replayR)]);
legend({'Preplay','Replay'},'Location','NortheastOutside');
f1(1).FaceColor=[1 1 1]; % Replay bar color
f1(1).LineWidth=1;
f1(2).FaceColor=[0 0 0]; % Replay bar color
f1(2).LineWidth=1;
set(f1(1).Parent,'XTickLabel',{'Forward correlated','Reverse correlated'});
ymax=ceil(max(reshape([nanmean(preplayF) nanmean(replayF) ; nanmean(preplayR) nanmean(replayR)],[],1)));
ylim([0 ymax]);
ylabel('Events/Lap');
grid on


%% EXPORT FIGURE
if saveBool
    save genFig_FvsR_linearTrack.mat

    % Set clipping off
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    
    set(gcf, 'renderer', 'painters');
    export_fig(['../Parts/' mfilename], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    %print(filename,'-dpdf','-fillpage')
end


