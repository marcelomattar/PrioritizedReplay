%% STATE-SPACE PARAMETERS
addpath('../../../');
clear;
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

% Using greedy policy!
params.actPolicy        = 'e_greedy'; % Choose 'thompson_sampling' or 'e_greedy' or 'softmax'
params.epsilon          = 0; % probability of a random action (epsilon-greedy)

saveStr = input('Do you want to produce figures (y/n)? ','s');
if strcmp(saveStr,'y')
    saveBool = true;
else
    saveBool = false;
end


%% RUN SIMULATION (prioritized replay)
rng(mean('replay'));
params.nPlan            = 20;
params.setAllGainToOne  = false; % Set the gain term of all items to one (for illustration purposes)
params.setAllNeedToOne  = false; % Set the need term of all items to one (for illustration purposes)
for k=1:params.N_SIMULATIONS
    rng('shuffle');
    simData(k) = replaySim(params);
end
stepsPerEpisode_PrioReplay = nan(params.MAX_N_EPISODES,params.N_SIMULATIONS);
for k=1:params.N_SIMULATIONS
    stepsPerEpisode_PrioReplay(:,k) = simData(k).stepsPerEpisode;
end
clear simData;
stepsBestPolicy_PrioReplay = nanmean(cummin(stepsPerEpisode_PrioReplay),2);


%% RUN SIMULATION (no replay)
rng(mean('replay'));
params.nPlan            = 0;
params.setAllGainToOne  = false; % Set the gain term of all items to one (for illustration purposes)
params.setAllNeedToOne  = false; % Set the need term of all items to one (for illustration purposes)
for k=1:params.N_SIMULATIONS
    rng('shuffle');
    simData(k) = replaySim(params);
end
stepsPerEpisode_NoReplay = nan(params.MAX_N_EPISODES,params.N_SIMULATIONS);
for k=1:params.N_SIMULATIONS
    stepsPerEpisode_NoReplay(:,k) = simData(k).stepsPerEpisode;
end
clear simData;
stepsBestPolicy_NoReplay = nanmean(cummin(stepsPerEpisode_NoReplay),2);


%% RUN SIMULATION (DYNA)
rng(mean('replay'));
params.nPlan            = 20;
params.setAllGainToOne  = true; % Set the gain term of all items to one (for illustration purposes)
params.setAllNeedToOne  = true; % Set the need term of all items to one (for illustration purposes)
for k=1:params.N_SIMULATIONS
    rng('shuffle');
    simData(k) = replaySim(params);
end
stepsPerEpisode_DYNA = nan(params.MAX_N_EPISODES,params.N_SIMULATIONS);
for k=1:params.N_SIMULATIONS
    stepsPerEpisode_DYNA(:,k) = simData(k).stepsPerEpisode;
end
clear simData;
stepsBestPolicy_DYNA = nanmean(cummin(stepsPerEpisode_DYNA),2);


%% RUN SIMULATION (PRIORITIZED SWEEPING)
%{
rng(mean('replay'));
params.nPlan            = 20;
for k=1:params.N_SIMULATIONS
    simData(k) = prioritized_sweeping(params);
end
stepsPerEpisode_PrioSweep = nan(params.MAX_N_EPISODES,params.N_SIMULATIONS);
for k=1:params.N_SIMULATIONS
    stepsPerEpisode_PrioSweep(:,k) = simData(k).stepsPerEpisode;
end
clear simData;
stepsBestPolicy_PrioSweep = nanmean(cummin(stepsPerEpisode_PrioSweep),2);
%}


%% COMPUTE CHANCE LEVEL

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

% Pick next start state at random
validStates = find(params.maze==0); % can start at any non-wall state
goalState = sub2ind(size(params.maze),params.s_end(:,1),params.s_end(:,2));
validStates = validStates(~ismember(validStates,goalState)); %... but remove the goal states from the list
chanceLevel = nanmean(dist(validStates,goalState));


%% PLOT RESULTS

figure(1); clf;

subplot(1,2,1)
f1=plot(nanmean(stepsPerEpisode_NoReplay,2));
hold on;
f2=plot(nanmean(stepsPerEpisode_DYNA,2));
%f3=plot(nanmean(stepsPerEpisode_PrioSweep,2));
f4=plot(nanmean(stepsPerEpisode_PrioReplay,2));
grid on;
%legend({'No replay', 'DYNA', 'Prioritized sweeping', 'Prioritized replay'})
legend({'No replay', 'DYNA', 'Prioritized replay'})
h=gcf;
h.Children(2).XTick = 0:5:20; xlim([0 20]);
h.Children(2).YTick = 0:50:200; ylim([0 200]);
l1 = line(xlim,[chanceLevel chanceLevel]);
f1.LineWidth=1;
f2.LineWidth=1;
%f3.LineWidth=1;
f4.LineWidth=1;
l1.LineWidth=1; l1.LineStyle=':'; l1.Color=[0 0 0];
ylabel('Number of steps to reward');
xlabel('# Episodes');
title('Learning performance');

subplot(1,2,2)
f1=plot(stepsBestPolicy_NoReplay);
hold on;
f2=plot(stepsBestPolicy_DYNA);
%f3=plot(stepsBestPolicy_PrioSweep);
f4=plot(stepsBestPolicy_PrioReplay);
grid on;
%legend({'No replay', 'DYNA', 'Prioritized sweeping', 'Prioritized replay'})
legend({'No replay', 'DYNA', 'Prioritized replay'})
h=gcf;
h.Children(2).XTick = 0:5:20; xlim([0 20]);
h.Children(2).YTick = 0:50:200; ylim([0 200]);
%l1 = line(xlim,[chanceLevel chanceLevel]);
f1.LineWidth=1;
f2.LineWidth=1;
%f3.LineWidth=1;
f4.LineWidth=1;
%l1.LineWidth=1; l1.LineStyle=':'; l1.Color=[0 0 0];
ylabel('Number of steps for best policy');
xlabel('# Episodes');
title('Learning performance (best so far)');

set(gcf,'Position',[1    81   983   281]);


%% EXPORT FIGURE
if saveBool
    save genFig_behavior_openMaze.mat

    % Set clipping off
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    
    set(gcf, 'renderer', 'painters');
    export_fig(['../Parts/' mfilename], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    %print(filename,'-dpdf','-fillpage')
end


