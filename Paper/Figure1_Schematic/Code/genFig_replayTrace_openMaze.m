%% STATE-SPACE PARAMETERS
addpath('../../../');
setParams;
params.maze             = zeros(6,9); % zeros correspond to 'visitable' states
params.maze(2:4,3)      = 1; % wall
params.maze(1:3,8)      = 1; % wall
params.maze(5,6)        = 1; % wall
params.s_end            = [1,9]; % goal state (in matrix notation)
params.s_start          = [3,1]; % beginning state (in matrix notation)
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
reverseEvent = false(size(simData.replay.nStep));
reverseEvent(find(~isnan(simData.replay.nStep),1,'first')) = true;
[I,J] = ind2sub(size(params.maze),simData.expList(reverseEvent,1));
st = [I,J];
planning_backups = [simData.replay.state{reverseEvent}' simData.replay.action{reverseEvent}'];
plotReplayTrace(st,planning_backups,params,hot(size(planning_backups,1)+10));
set(gca, 'Clipping', 'off'); set(gcf, 'Clipping', 'off');
set(gcf, 'renderer', 'painters');
set(gcf, 'renderer', 'painters');
%export_fig('../Parts/reverseTrace_openMaze', '-dpdf', '-eps', '-q101', '-nocrop', '-painters');
print('../Parts/reverseTrace_openMaze','-dpdf')

% Plot forward event
forwardEvent = false(size(simData.replay.nStep));
forwardEvent(find(simData.replay.nStep == max(simData.replay.nStep),1,'first')) = true;
assert(sum(forwardEvent)==1,'The number of events selected is not equal to one');
[I,J] = ind2sub(size(params.maze),simData.expList(forwardEvent,1));
st = [I,J];
planning_backups = [simData.replay.state{forwardEvent}' simData.replay.action{forwardEvent}'];
plotReplayTrace(st,planning_backups,params,hot(size(planning_backups,1)+10));
set(gca, 'Clipping', 'off'); set(gcf, 'Clipping', 'off');
set(gcf, 'renderer', 'painters');
%export_fig('../Parts/forwardTrace_openMaze', '-dpdf', '-eps', '-q101', '-nocrop', '-painters');
print('../Parts/forwardTrace_openMaze','-dpdf')
