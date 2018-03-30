%% STATE-SPACE PARAMETERS
setParams;
params.maze             = zeros(6,9); % zeros correspond to 'visitable' states
params.maze(2:4,3)      = 1; % wall
params.maze(1:3,8)      = 1; % wall
params.maze(5,6)        = 1; % wall
params.s_start          = [3,1]; % beginning state (in matrix notation)
params.s_start_rand     = false; % Start at random locations after reaching goal

%params.s_end            = [1,9;6,9]; % goal state (in matrix notation)
%params.rewMag           = [1 0.5; 0.1 0.05]; % reward magnitude (rows: locations; columns: values)
%params.rewSTD           = [1 0.5; 0.1 0.05]; % reward Gaussian noise (rows: locations; columns: values)
params.s_end            = [1,9]; % goal state (in matrix notation)
params.rewMag           = 1; % reward magnitude (rows: locations; columns: values)
params.rewSTD           = 0.1; % reward Gaussian noise (rows: locations; columns: values)
params.rewProb          = 1; % probability of receiving each reward (columns: values)


%% PLOTTING SETTINGS
params.PLOT_STEPS       = true; % Plot each step of real experience
params.PLOT_Qvals       = true; % Plot Q-values
params.PLOT_PLANS       = true; % Plot each planning step
params.PLOT_EVM         = true; % Plot need and gain
params.PLOT_TRACE       = false; % Plot all planning traces
params.PLOT_wait        = 1 ; % Number of full episodes completed before plotting


%% RUN SIMULATION
rng(mean('replay'));
simData = replaySim(params);
