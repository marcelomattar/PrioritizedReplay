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
params.MAX_N_EPISODES   = 50; % maximum number of episodes to simulate (use Inf if no max) -> Choose between 20 and 100

params.setAllGainToOne  = false; % Set the gain term of all items to one (for illustration purposes)
params.setAllNeedToOne  = false; % Set the need term of all items to one (for illustration purposes)
params.rewSTD           = 0.1; % reward standard deviation (can be a vector -- e.g. [1 0.1])
params.planPolicy       = 'softmax'; % Choose 'thompson_sampling' or 'e_greedy' or 'softmax'
params.softmaxT         = 0.2; % soft-max temperature -> higher means more exploration and, therefore, more reverse replay
params.actPolicy        = 'e_greedy'; % Choose 'thompson_sampling' or 'e_greedy' or 'softmax'
params.epsilon          = 0; % probability of a random action (epsilon-greedy)
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


%% RUN SIMULATION (prioritized replay)
rng(mean('replay'));
params.nPlan            = 20;
for k=1:params.N_SIMULATIONS
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
for k=1:params.N_SIMULATIONS
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
    simData(k) = replaySim(params);
end
stepsPerEpisode_DYNA = nan(params.MAX_N_EPISODES,params.N_SIMULATIONS);
for k=1:params.N_SIMULATIONS
    stepsPerEpisode_DYNA(:,k) = simData(k).stepsPerEpisode;
end
clear simData;
stepsBestPolicy_DYNA = nanmean(cummin(stepsPerEpisode_DYNA),2);


%% RUN SIMULATION (PRIORITIZED SWEEPING)
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


%% PLOT RESULTS

figure(1); clf;

subplot(1,2,1)
f1=plot(nanmean(stepsPerEpisode_NoReplay,2));
hold on;
f2=plot(nanmean(stepsPerEpisode_DYNA,2));
f3=plot(nanmean(stepsPerEpisode_PrioSweep,2));
f4=plot(nanmean(stepsPerEpisode_PrioReplay,2));
grid on;
legend({'No replay', 'DYNA', 'Prioritized sweeping', 'Prioritized replay'})
h=gcf;
h.Children(2).XTick = 0:5:20; xlim([0 20]);
h.Children(2).YTick = 0:50:200; ylim([0 200]);
l1 = line(xlim,[14 14]);
f1.LineWidth=1;
f2.LineWidth=1;
f3.LineWidth=1;
f4.LineWidth=1;
l1.LineWidth=1; l1.LineStyle=':'; l1.Color=[0 0 0];
ylabel('Number of steps to reward');
xlabel('# Episodes');
title('Learning performance');

subplot(1,2,2)
f1=plot(stepsBestPolicy_NoReplay);
hold on;
f2=plot(stepsBestPolicy_DYNA);
f3=plot(stepsBestPolicy_PrioSweep);
f4=plot(stepsBestPolicy_PrioReplay);
grid on;
legend({'No replay', 'DYNA', 'Prioritized sweeping', 'Prioritized replay'})
h=gcf;
h.Children(2).XTick = 0:5:20; xlim([0 20]);
h.Children(2).YTick = 0:50:200; ylim([0 200]);
l1 = line(xlim,[14 14]);
f1.LineWidth=1;
f2.LineWidth=1;
f3.LineWidth=1;
f4.LineWidth=1;
l1.LineWidth=1; l1.LineStyle=':'; l1.Color=[0 0 0];
ylabel('Number of steps for best policy');
xlabel('# Episodes');
title('Learning performance');

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


