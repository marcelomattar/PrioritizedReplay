%% STATE-SPACE PARAMETERS
addpath('../../../');
clear;
setParams;
params.maze             = zeros(3,10); % zeros correspond to 'visitable' states
params.maze(2,:)        = 1; % wall
params.s_end            = [1,size(params.maze,2);3,1]; % goal state (in matrix notation)
params.s_start          = [1,1;3,size(params.maze,2)]; % beginning state (in matrix notation)
params.s_start_rand     = false; % Start at random locations after reaching goal

%% OVERWRITE PARAMETERS
params.N_SIMULATIONS    = 1; % number of times to run the simulation
params.MAX_N_STEPS      = 1e5; % maximum number of steps to simulate
params.MAX_N_EPISODES   = 5; % maximum number of episodes to simulate (use Inf if no max) -> Choose between 20 and 100
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


%% INITIALIZE VARIABLES
stateNumbers = reshape(1:numel(params.maze(:)),size(params.maze,1),size(params.maze,2));
startStates = sub2ind(size(params.maze),params.s_start(:,1),params.s_start(:,2));
goalStates = sub2ind(size(params.maze),params.s_end(:,1),params.s_end(:,2));


%% REVERSE REPLAY

firstGoal = find(ismember(simData.expList(:,4),goalStates),1,'first');
sti = simData.expList(firstGoal,1);
at = simData.expList(firstGoal,2);
rt = simData.expList(firstGoal,3);
stp1i = simData.expList(firstGoal,4);
st = nan(1,2); stp1 = nan(1,2);
[I,J] = ind2sub(size(params.maze),sti);
st(1) = I; st(2) = J; 
[I,J] = ind2sub(size(params.maze),stp1i);
stp1(1) = I; stp1(2) = J; 
replayState = simData.replay.state{firstGoal};
replayAction = simData.replay.action{firstGoal};
SR = simData.replay.SR{firstGoal}(sti,:);
saGain = simData.replay.saGain{firstGoal};
QVals = simData.replay.backupsQvals{firstGoal};

% Plot 4 steps of Gain-term
for i=1:4
    figure(1); clf;
    if i==1
        stepQvals = simData.Q{firstGoal};
    else
        stepQvals = reshape(QVals(:,i-1),size(simData.Q{firstGoal}));
    end
    squareValues = max(stepQvals,[],2);
    arrowValues = zeros(size(simData.Q{firstGoal}));
    arrowValues(replayState(i),replayAction(i)) = simData.replay.gain{firstGoal}(i);
    colscale_squares = 'b2r(-1,1)';
    colscale_arrows = 'b2r(-1,1)';
    %plotMazeWithArrows(st,squareValues,arrowValues,params,colscale_arrows,colscale_squares);
    highlight = zeros(size(simData.Q{firstGoal}));
    highlight(replayState(i),replayAction(i)) = 1;
    
    if saveBool
        plotMazeWithArrows(st,squareValues,arrowValues,params,colscale_arrows,colscale_squares,highlight(:));
        set(gcf,'Position',[562,535,560,420])
        set(gca, 'Clipping', 'off');
        set(gcf, 'Clipping', 'off');
        set(gcf, 'renderer', 'painters');
        %export_fig(['../Parts/reverse' num2str(i)], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
        print(['../Parts/linearTrack_reverseGain' num2str(i)],'-dpdf');
        pause(1);
    end
end

% Plot Need-term
squareValues = SR';
arrowValues = zeros(size(simData.Q{firstGoal}));
colscale_squares = 'gray';
colscale_arrows = 'b2r(-1,1)';
if saveBool
    plotMazeWithArrows(st,squareValues,arrowValues,params,colscale_arrows,colscale_squares);
    set(gcf,'Position',[562,535,560,420])
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    set(gcf, 'renderer', 'painters');
    %export_fig(['../Parts/reverse' num2str(i)], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    print(['../Parts/linearTrack_reverseNeed'],'-dpdf');
end

% Plot Replay trace
planning_backups = simData.replay.backups{firstGoal};
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
    print(['../Parts/linearTrack_reverseTrace'],'-dpdf');
end

if saveBool
    figure(4);
    colormap(map);
    colorbar;
    print(['../Parts/linearTrack_colorbar'],'-dpdf');
end


%% FORWARD REPLAY

secondStart = find(simData.expList(:,1)==startStates(1),2,'first');
secondStart = secondStart(2);
sti = simData.expList(secondStart,1);
at = simData.expList(secondStart,2);
rt = simData.expList(secondStart,3);
stp1i = simData.expList(secondStart,4);
st = nan(1,2); stp1 = nan(1,2);
[I,J] = ind2sub(size(params.maze),sti);
st(1) = I; st(2) = J; 
[I,J] = ind2sub(size(params.maze),stp1i);
stp1(1) = I; stp1(2) = J; 
replayState = simData.replay.state{secondStart};
replayAction = simData.replay.action{secondStart};
SR = simData.replay.SR{secondStart}(sti,:);
saGain = simData.replay.saGain{secondStart};
QVals = simData.replay.backupsQvals{secondStart};

% Plot 4 steps of Gain-term
for i=1:4
    figure(1); clf;
    if i==1
        stepQvals = simData.Q{secondStart};
    else
        stepQvals = reshape(QVals(:,i-1),size(simData.Q{secondStart}));
    end
    squareValues = max(stepQvals,[],2);
    arrowValues = zeros(size(simData.Q{secondStart}));
    arrowValues(replayState(i),replayAction(i)) = simData.replay.gain{secondStart}(i);
    colscale_squares = 'b2r(-1,1)';
    colscale_arrows = 'b2r(-1,1)';
    highlight = zeros(size(simData.Q{secondStart}));
    highlight(replayState(i),replayAction(i)) = 1;
    
    if saveBool
        plotMazeWithArrows(st,squareValues,arrowValues,params,colscale_arrows,colscale_squares,highlight(:));
        set(gcf,'Position',[562,535,560,420])
        set(gca, 'Clipping', 'off');
        set(gcf, 'Clipping', 'off');
        set(gcf, 'renderer', 'painters');
        %export_fig(['../Parts/reverse' num2str(i)], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
        print(['../Parts/linearTrack_forwardGain' num2str(i)],'-dpdf');
        pause(1);
    end
end

% Plot Need-term
squareValues = SR';
arrowValues = zeros(size(simData.Q{secondStart}));
colscale_squares = 'gray';
colscale_arrows = 'b2r(-1,1)';
if saveBool
    plotMazeWithArrows(st,squareValues,arrowValues,params,colscale_arrows,colscale_squares);
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    set(gcf, 'renderer', 'painters');
    %export_fig(['../Parts/reverse' num2str(i)], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    print(['../Parts/linearTrack_forwardNeed'],'-dpdf');
end

% Plot Replay trace
planning_backups = simData.replay.backups{secondStart};
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
    print(['../Parts/linearTrack_forwardTrace'],'-dpdf');
end
