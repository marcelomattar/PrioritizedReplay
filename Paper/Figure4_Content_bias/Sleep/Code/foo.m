%% STATE-SPACE PARAMETERS

% PS: Need to change replaySim.m to only allow replay at the start
% location!
params.onlineVSoffline  = 'offline'; % Choose 'online' or 'offline' to determine what to use as the need-term

addpath('../../../');
clear;
setParams;
params.maze             = zeros(2,5); % zeros correspond to 'visitable' states
params.maze(2,[1,2,4,5])= 1; % wall
params.s_end            = [1,1;1,5]; % goal state (in matrix notation)
params.s_start          = [2,3]; % beginning state (in matrix notation)
params.s_start_rand     = false; % Start at random locations after reaching goal

params.rewMag           = [0 1]; % reward magnitude (can be a vector -- e.g. [1 0.1])
params.rewSTD           = [0 0.1]; % reward standard deviation (can be a vector -- e.g. [1 0.1])

%% OVERWRITE PARAMETERS
params.N_SIMULATIONS    = 100; % number of times to run the simulation
params.MAX_N_STEPS      = 1e5; % maximum number of steps to simulate
params.MAX_N_EPISODES   = 50; % maximum number of episodes to simulate (use Inf if no max) -> Choose between 20 and 100
params.nPlan            = 1; % number of steps to do in planning (set to zero if no planning or to Inf to plan for as long as it is worth it)

params.setAllGainToOne  = false; % Set the gain term of all items to one (for illustration purposes)
params.setAllNeedToOne  = false; % Set the need term of all items to one (for illustration purposes)
params.softmaxT         = 0.2; % soft-max temperature -> higher means more exploration and, therefore, more reverse replay
params.gamma            = 0.90; % discount factor

params.updIntermStates  = true; % Update intermediate states when performing n-step backup
params.baselineGain     = 1e-10; % Gain is set to at least this value (interpreted as "information gain") -> Use 1e-3 if LR=0.8

params.alpha            = 1.0; % learning rate for real experience (non-bayesian)
params.copyQinPlanBkps  = false; % Copy the Q-value (mean and variance) on planning backups (i.e., LR=1.0)
params.copyQinGainCalc  = true; % Copy the Q-value (mean and variance) on gain calculation (i.e., LR=1.0)

params.PLOT_STEPS       = false; % Plot each step of real experience
params.PLOT_Qvals       = false; % Plot Q-values
params.PLOT_PLANS       = false; % Plot each planning step
params.PLOT_EVM         = false; % Plot need and gain
params.PLOT_wait        = 0 ; % Number of full episodes completed before plotting

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


%% RUN ANALYSIS
replayCount = zeros(numel(params.maze),4);
for k=1:params.N_SIMULATIONS
    saReplay = [cell2mat(simData(k).replay.state) cell2mat(simData(k).replay.action)];
    for s=1:numel(params.maze)
        for a=1:4
            replayCount(s,a) = sum(and(saReplay(:,1)==s,saReplay(:,2)==a));
        end
    end
end
figure(1);clf;
replayRight = replayCount(5,3)+replayCount(7,3);
replayLeft = replayCount(5,4)+replayCount(3,4);
h1=bar(100*[replayLeft/sum(replayCount(:)) replayRight/sum(replayCount(:))]);
h1.Parent.XTickLabel={'LEFT','RIGHT'};
ylim([0 100]);

figure(2);clf;
subplot(1,2,1);
replayStem = replayCount(6,1);
replayRight = replayCount(5,3)+replayCount(7,3);
replayLeft = replayCount(5,4)+replayCount(3,4);
h1=bar([replayLeft/sum(replayCount(:)) replayStem/sum(replayCount(:)) replayRight/sum(replayCount(:))]);

subplot(1,2,2);
replayStem = sum(sum(replayCount([5,6],:)));
replayRight = sum(sum(replayCount([7,9],:)));
replayLeft = sum(sum(replayCount([1,3],:)));
h1=bar([replayLeft/sum(replayCount(:)) replayStem/sum(replayCount(:)) replayRight/sum(replayCount(:))]);


%% EXPORT FIGURE
%{
if saveBool
    save genFig_FvsR_linearTrack.mat

    % Set clipping off
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    
    set(gcf, 'renderer', 'painters');
    export_fig(['../Parts/' mfilename], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    %print(filename,'-dpdf','-fillpage')
end
%}

