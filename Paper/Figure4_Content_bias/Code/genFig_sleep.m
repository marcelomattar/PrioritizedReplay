%% STATE-SPACE PARAMETERS

addpath('../../../');
clear;
setParams;
params.maze             = zeros(2,5); % zeros correspond to 'visitable' states
params.maze(2,[1,2,4,5])= 1; % wall
params.s_start          = [2,3]; % beginning state (in matrix notation)
params.s_start_rand     = false; % Start at random locations after reaching goal

params.s_end            = [1,1;1,5]; % goal state (in matrix notation)
params.rewMag           = [0;1]; % reward magnitude (rows: locations; columns: values)
params.rewSTD           = [0;0.1]; % reward Gaussian noise (rows: locations; columns: values)
params.rewProb          = 1; % probability of receiving each reward (columns: values)


%% OVERWRITE PARAMETERS
params.N_SIMULATIONS    = 1000; % number of times to run the simulation
params.MAX_N_STEPS      = 1e5; % maximum number of steps to simulate
params.MAX_N_EPISODES   = 50; % maximum number of episodes to simulate (use Inf if no max) -> Choose between 20 and 100
params.nPlan            = 20; % number of steps to do in planning (set to zero if no planning or to Inf to plan for as long as it is worth it)
params.onVSoffPolicy    = 'off-policy'; % Choose 'off-policy' (default, learns Q*) or 'on-policy' (learns Qpi) learning for updating Q-values and computing gain
params.alpha            = 1.0; % learning rate for real experience (non-bayesian)
params.gamma            = 0.90; % discount factor
params.softmaxInvT      = 5; % soft-max inverse temperature temperature

params.onlineVSoffline  = 'offline'; % Choose 'online' or 'offline' to determine what to use as the need-term

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
replayCount = zeros(numel(params.maze),4,params.N_SIMULATIONS);
for k=1:params.N_SIMULATIONS
    saReplay = [reshape(cell2mat(simData(k).replay.state),[],1) reshape(cell2mat(simData(k).replay.action),[],1)];
    for s=1:numel(params.maze)
        for a=1:4
            replayCount(s,a,k) = sum(and(saReplay(:,1)==s,saReplay(:,2)==a));
        end
    end
end
replayCount = nanmean(replayCount,3);

figure(1);clf;
replayRight = replayCount(5,3)+replayCount(7,3);
replayLeft = replayCount(5,4)+replayCount(3,4);
h1=bar(100*[replayLeft/sum(replayCount(:)) replayRight/sum(replayCount(:))]);
h1.Parent.XTickLabel={'UNCUED','CUED'};
ylim([0 100]);
%{
figure(2);clf;
subplot(1,2,1);
replayStem = replayCount(6,1);
replayRight = replayCount(5,3)+replayCount(7,3);
replayLeft = replayCount(5,4)+replayCount(3,4);
h2=bar([replayLeft/sum(replayCount(:)) replayStem/sum(replayCount(:)) replayRight/sum(replayCount(:))]);
h2.Parent.XTickLabel={'LEFT','STEM','RIGHT'};
title(sprintf('PS: Considering only R/L actions\n in R/L arms (and UP in STEM)\nPS2: Fractions do not sum to 1'))

subplot(1,2,2);
replayStem = sum(sum(replayCount([5,6],:)));
replayRight = sum(sum(replayCount([7,9],:)));
replayLeft = sum(sum(replayCount([1,3],:)));
h3=bar([replayLeft/sum(replayCount(:)) replayStem/sum(replayCount(:)) replayRight/sum(replayCount(:))]);
h3.Parent.XTickLabel={'LEFT','STEM','RIGHT'};
title(sprintf('PS: Considering the actual state location\nPS2: Fractions do not sum to 1'))
%}

%% EXPORT FIGURE
if saveBool
    save genFig_sleep.mat

    % Set clipping off
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    
    set(gcf, 'renderer', 'painters');
    export_fig(['../Parts/' mfilename], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    %print(filename,'-dpdf','-fillpage')
end

