%% STATE-SPACE PARAMETERS
addpath('../../../');
setParams;
params.maze             = zeros(3,10); % zeros correspond to 'visitable' states
params.maze(2,:)        = 1; % wall
params.s_end            = [1,size(params.maze,2);3,1]; % goal state (in matrix notation)
params.s_start          = [1,1;3,size(params.maze,2)]; % beginning state (in matrix notation)
params.s_start_rand     = false; % Start at random locations after reaching goal

%% OVERWRITE PARAMETERS
params.N_SIMULATIONS    = 1; % number of times to run the simulation
params.MAX_N_STEPS      = 1e5; % maximum number of steps to simulate
params.MAX_N_EPISODES   = 50; % maximum number of episodes to simulate (use Inf if no max)


%% RUN SIMULATION
rng(mean('replay'));
simData = replaySim(params);


%% PLOT RESULTS

% Plot reverse event
reverseEvent = and(~isnan(simData.replay.nStep),simData.numEpisodes==1);
[I,J] = ind2sub(size(params.maze),simData.expList(reverseEvent,1));
st = [I,J];
planning_backups = [simData.replay.state{reverseEvent}' simData.replay.action{reverseEvent}'];
plotReplayTrace(st,planning_backups,params,hot(size(planning_backups,1)+10));
set(gca, 'Clipping', 'off'); set(gcf, 'Clipping', 'off');
set(gcf, 'renderer', 'painters');
print('../Parts/reverseTrace_linearTrack','-dpdf')

% Plot forward event
forwardEvent = simData.replay.nStep>7;
[I,J] = ind2sub(size(params.maze),simData.expList(forwardEvent,1));
st = [I,J];
planning_backups = [simData.replay.state{forwardEvent}' simData.replay.action{forwardEvent}'];
plotReplayTrace(st,planning_backups,params,hot(size(planning_backups,1)+10));
set(gca, 'Clipping', 'off'); set(gcf, 'Clipping', 'off');
set(gcf, 'renderer', 'painters');
print('../Parts/forwardTrace_linearTrack','-dpdf')