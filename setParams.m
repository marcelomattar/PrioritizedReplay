%% SIMULATION PARAMETERS
params.MAX_N_STEPS      = 1e5; % maximum number of steps to simulate
params.MAX_N_EPISODES   = 50; % maximum number of episodes to simulate (use Inf if no max)


%% MDP PARAMETERS
params.gamma            = 0.9; % discount factor
params.alpha            = 1.0; % learning rate for real experience (non-bayesian)
params.lambda           = 0; % eligibility trace parameter
params.TLearnRate       = 0.9; % learning rate for the transition matrix (0=uniform; 1=only last)
params.actPolicy        = 'softmax'; % Choose 'e_greedy' or 'softmax'
params.softmaxInvT      = 5; % soft-max inverse temperature temperature
params.epsilon          = 0.05; % probability of a random action (epsilon-greedy)
params.preExplore       = true; % Let the agent explore the maze (without rewards) to learn transition model
params.add_goal2start   = true; % Include a transition from goal to start in transition matrix -- this allows Need-term to wrap around
params.rewOnlyPositive  = true; % When drawing reward samples, use only 


%% PLANNING PARAMETERS
params.nPlan            = 20; % number of steps to do in planning (set to zero if no planning or to Inf to plan for as long as planning beats the opportunity cost)
params.EVMthresh        = 0; % minimum EVM so that planning is performed (use Inf if wish to use opportunity cost)

% Parameters for n-step backups
params.expandFurther    = true; % Expand the last backup further
params.planPolicy       = 'softmax'; % Choose 'thompson_sampling' or 'e_greedy' or 'softmax'

% Other planning parameters
params.planOnlyAtGorS   = true; % boolean variable indicating if planning should happen only if the agent is at the start or goal state
params.baselineGain     = 1e-8; % Gain is set to at least this value (interpreted as "information gain")
params.tieBreak         = 'rand'; % How to break ties on EVM (choose between 'min', 'max', or 'rand');
params.onlineVSoffline  = 'online'; % Choose 'online' or 'offline' (e.g. sleep) to determine what to use as the need-term
params.remove_samestate = true; % Remove actions whose consequences lead to same state (e.g. hitting the wall)
params.allowLoops       = false; % Allow planning trajectories that return to a location appearing previously in the plan
params.copyQinGainCalc  = false; % Copy Q-value on gain calculation (i.e., LR=1.0)
params.copyQinPlanBkps  = false; % Copy Q-value on planning backups (i.e., LR=1.0)
params.setAllGainToOne  = false; % Set the gain term of all items to one (for debugging purposes)
params.setAllNeedToOne  = false; % Set the need term of all items to one (for debugging purposes)


%% PLOTTING SETTINGS
params.PLOT_STEPS       = false; % Plot each step of real experience
params.PLOT_Qvals       = false; % Plot Q-values
params.PLOT_PLANS       = false; % Plot each planning step
params.PLOT_EVM         = false; % Plot need and gain
params.PLOT_TRACE       = false; % Plot all planning traces
params.PLOT_wait        = 1 ; % Number of full episodes completed before plotting
params.delay            = 0.2; % How long to wait between successive plots

params.arrowOffset      = 0.25;
params.arrowSize        = 100;
params.agentSize        = 15;
params.agentColor       = [0.5 0.5 0.5];
params.startendSymbSize = 24;

%params.colormap         = b2r(-1,1); close all;
%params.debugMode        = false;