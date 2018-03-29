%% STATE-SPACE PARAMETERS
addpath('../../../');
clear;
setParams;
params.maze             = zeros(6,9); % zeros correspond to 'visitable' states
params.maze(2:4,3)      = 1; % wall
params.maze(1:3,8)      = 1; % wall
params.maze(5,6)        = 1; % wall
%params.s_end            = [1,9;6,9]; % goal state (in matrix notation)
params.s_end            = [1,9]; % goal state (in matrix notation)
params.s_start          = [3,1]; % beginning state (in matrix notation)
params.s_start_rand     = false; % Start at random locations after reaching goal

%% OVERWRITE PARAMETERS
params.N_SIMULATIONS    = 1000; % number of times to run the simulation
params.MAX_N_STEPS      = 1e5; % maximum number of steps to simulate
params.MAX_N_EPISODES   = 2; % maximum number of episodes to simulate (use Inf if no max) -> Choose between 20 and 100
params.nPlan            = 20; % number of steps to do in planning (set to zero if no planning or to Inf to plan for as long as it is worth it)

params.setAllGainToOne  = false; % Set the gain term of all items to one (for illustration purposes)
params.setAllNeedToOne  = false; % Set the need term of all items to one (for illustration purposes)
params.rewSTD           = 0.1; % reward standard deviation (can be a vector -- e.g. [1 0.1])
params.softmaxT         = 0.2; % soft-max temperature -> higher means more exploration and, therefore, more reverse replay
params.gamma            = 0.90; % discount factor

params.updIntermStates  = true; % Update intermediate states when performing n-step backup
params.baselineGain     = 1e-10; % Gain is set to at least this value (interpreted as "information gain") -> Use 1e-3 if LR=0.8

saveStr = input('Do you want to produce figures (y/n)? ','s');
if strcmp(saveStr,'y')
    saveBool = true;
else
    saveBool = false;
end


%% RUN SIMULATION
rng(mean('replay'));


%% ANALYSIS PARAMETERS
minNumCells = 3;
minFracCells = 0;
runPermAnalysis = true; % Run permutation analysis (true or false)
nPerm = 500; % Number of permutations for assessing significance of an event


%% INITIALIZE VARIABLES
stateNumbers = reshape(1:numel(params.maze(:)),size(params.maze,1),size(params.maze,2));
startStates = sub2ind(size(params.maze),params.s_start(:,1),params.s_start(:,2));
goalStates = sub2ind(size(params.maze),params.s_end(:,1),params.s_end(:,2));
forwardCount = zeros(params.N_SIMULATIONS,numel(params.maze));
reverseCount = zeros(params.N_SIMULATIONS,numel(params.maze));
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


for k=1:params.N_SIMULATIONS
    
    simData(k) = replaySim(params);
    
    forwardEvents = [];
    reverseEvents = [];
    %start_after_goal = find(and(simData(k).expList(:,1)==startStates(1),[simData(k).expList(end,4); simData(k).expList(1:end-1,4)]==goalStates(1)));
    start_after_goal = find([simData(k).expList(end,4); simData(k).expList(1:end-1,4)]==goalStates(1));
    % Identify candidate replay events
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
                    if replayDir{1}=='F'
                        forwardCount(k,agentPos(e)) = forwardCount(k,agentPos(e)) + 1;
                        forwardEvents = [forwardEvents; candidateEvents(e)];
                        %if and(candidateEvents(e)==start_after_goal(2) , sum(ismember([1 2 3], thisChunk(1)))>0)
                        %if sum(ismember(1:5, thisChunk(1)))>0
                        %    break
                        %end
                        %if (agentPos(e)~=50) && (ismember(replayState(1),[2,3,4,9]))
                        if (agentPos(e)~=50)
                            display(thisChunk);
                            flag=1; 
                            break
                        end
                    elseif replayDir{1}=='R'
                        reverseCount(k,agentPos(e)) = reverseCount(k,agentPos(e)) + 1;
                        reverseEvents = [reverseEvents; candidateEvents(e)];
                    end
                end
            end
        end
        if(flag==1)
          break
        end
    end
    if(flag==1)
        break
    end
end


%% FORWARD REPLAY

thisForward = forwardEvents(1);
sti = simData(k).expList(thisForward,1);
at = simData(k).expList(thisForward,2);
rt = simData(k).expList(thisForward,3);
stp1i = simData(k).expList(thisForward,4);
st = nan(1,2); stp1 = nan(1,2);
[I,J] = ind2sub(size(params.maze),sti);
st(1) = I; st(2) = J;
[I,J] = ind2sub(size(params.maze),stp1i);
stp1(1) = I; stp1(2) = J;
replayState = simData(k).replay.state{thisForward};
replayAction = simData(k).replay.action{thisForward};
SR = simData(k).replay.SR{thisForward}(sti,:);
saGain = simData(k).replay.saGain{thisForward};
QVals = simData(k).replay.backupsQvals{thisForward};

% Plot 4 steps of Gain-term
j=1;
for i=[thisChunk (thisChunk(end)+1)]
    figure(1); clf;
    if i==1
        stepQvals = simData(k).Q{thisForward};
    else
        stepQvals = reshape(QVals(:,i-1),size(simData(k).Q{thisForward}));
    end
    squareValues = max(stepQvals,[],2);
    %arrowValues = zeros(size(simData(k).Q{thisForward}));
    %arrowValues(replayState(i),replayAction(i)) = simData(k).replay.gain{thisForward}(i);
    arrowValues = reshape(simData(k).replay.backupsSAGain{thisForward}(:,i),size(simData(k).Q{thisForward},1),size(simData(k).Q{thisForward},2));
    
    colscale_squares = 'b2r(-1,1)';
    colscale_arrows = 'b2r(-1,1)';
    highlight = zeros(size(simData(k).Q{thisForward}));
    highlight(replayState(i),replayAction(i)) = 1;
    
    if saveBool
        plotMazeWithArrows(st,squareValues,arrowValues,params,colscale_arrows,colscale_squares,highlight(:));
        set(gcf,'Position',[562,535,560,420])
        set(gca, 'Clipping', 'off');
        set(gcf, 'Clipping', 'off');
        set(gcf, 'renderer', 'painters');
        print(['../Parts/openMaze_forwardGain' num2str(j)],'-dpdf');
        pause(1);
    end
    j=j+1;
end

% Plot Need-term
squareValues = SR';
arrowValues = zeros(size(simData(k).Q{thisForward}));
colscale_squares = 'gray';
colscale_arrows = 'b2r(-1,1)';
if saveBool
    plotMazeWithArrows(st,squareValues,arrowValues,params,colscale_arrows,colscale_squares);
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    set(gcf, 'renderer', 'painters');
    %export_fig(['../Parts/reverse' num2str(i)], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    print(['../Parts/openMaze_forwardNeed'],'-dpdf');
end

% Plot Replay trace
planning_backups = simData(k).replay.backups{thisForward};
cyan = [0 1 1];
purple = [0.5 0 0.5];
scal = linspace(0, 1, size(planning_backups,1)+10);
map = cyan + scal' * (purple-cyan);
if saveBool
    plotReplayTrace(st,planning_backups,params,map);
    set(gcf,'Position',[562,535,560,420])
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    set(gcf, 'renderer', 'painters');
    %export_fig(['../Parts/reverse' num2str(i)], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    print(['../Parts/openMaze_forwardTrace'],'-dpdf');
end


%% REVERSE REPLAY

firstGoal = forwardEvents(1)-1;
sti = simData(k).expList(firstGoal,1);
at = simData(k).expList(firstGoal,2);
rt = simData(k).expList(firstGoal,3);
stp1i = simData(k).expList(firstGoal,4);
st = nan(1,2); stp1 = nan(1,2);
[I,J] = ind2sub(size(params.maze),sti);
st(1) = I; st(2) = J;
[I,J] = ind2sub(size(params.maze),stp1i);
stp1(1) = I; stp1(2) = J;
replayState = simData(k).replay.state{firstGoal};
replayAction = simData(k).replay.action{firstGoal};
SR = simData(k).replay.SR{firstGoal}(sti,:);
saGain = simData(k).replay.saGain{firstGoal};
QVals = simData(k).replay.backupsQvals{firstGoal};

% Plot 4 steps of Gain-term
for i=1:4
    figure(1); clf;
    if i==1
        stepQvals = simData(k).Q{firstGoal};
    else
        stepQvals = reshape(QVals(:,i-1),size(simData(k).Q{firstGoal}));
    end
    squareValues = max(stepQvals,[],2);
    %arrowValues = zeros(size(simData(k).Q{firstGoal}));
    %arrowValues(replayState(i),replayAction(i)) = simData(k).replay.gain{firstGoal}(i);
    arrowValues = reshape(simData(k).replay.backupsSAGain{firstGoal}(:,i),size(simData(k).Q{firstGoal},1),size(simData(k).Q{firstGoal},2));
    colscale_squares = 'b2r(-1,1)';
    colscale_arrows = 'b2r(-1,1)';
    highlight = zeros(size(simData(k).Q{firstGoal}));
    highlight(replayState(i),replayAction(i)) = 1;
    
    if saveBool
        plotMazeWithArrows(st,squareValues,arrowValues,params,colscale_arrows,colscale_squares,highlight(:));
        set(gcf,'Position',[562,535,560,420])
        set(gca, 'Clipping', 'off');
        set(gcf, 'Clipping', 'off');
        set(gcf, 'renderer', 'painters');
        print(['../Parts/openMaze_reverseGain' num2str(i)],'-dpdf');
        pause(1);
    end
end

% Plot Need-term
squareValues = SR';
arrowValues = zeros(size(simData(k).Q{firstGoal}));
colscale_squares = 'gray';
colscale_arrows = 'b2r(-1,1)';
if saveBool
    plotMazeWithArrows(st,squareValues,arrowValues,params,colscale_arrows,colscale_squares);
    set(gcf,'Position',[562,535,560,420])
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    set(gcf, 'renderer', 'painters');
    %export_fig(['../Parts/reverse' num2str(i)], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    print(['../Parts/openMaze_reverseNeed'],'-dpdf');
end

% Plot Replay trace
planning_backups = simData(k).replay.backups{firstGoal};
cyan = [0 1 1];
purple = [0.5 0 0.5];
scal = linspace(0, 1, size(planning_backups,1)+10);
map = cyan + scal' * (purple-cyan);
if saveBool
    plotReplayTrace(st,planning_backups,params,map);
    set(gcf,'Position',[562,535,560,420])
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    set(gcf, 'renderer', 'painters');
    %export_fig(['../Parts/reverse' num2str(i)], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    print(['../Parts/openMaze_reverseTrace'],'-dpdf');
end

