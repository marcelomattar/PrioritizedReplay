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
l1 = line(xlim,[9 9]);
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
l1 = line(xlim,[9 9]);
f1.LineWidth=1;
f2.LineWidth=1;
%f3.LineWidth=1;
f4.LineWidth=1;
l1.LineWidth=1; l1.LineStyle=':'; l1.Color=[0 0 0];
ylabel('Number of steps for best policy');
xlabel('# Episodes');
title('Learning performance (best so far)');

set(gcf,'Position',[1    81   983   281]);


%% EXPORT FIGURE
if saveBool
    save genFig_behavior_linearTrack.mat

    % Set clipping off
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    
    set(gcf, 'renderer', 'painters');
    export_fig(['../Parts/' mfilename], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    %print(filename,'-dpdf','-fillpage')
end


