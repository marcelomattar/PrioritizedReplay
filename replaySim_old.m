function simData = replaySim_old(params)
%REPLAYSIM      Replay simulation on a grid-world.
%   params = Structure with all simulation parameters
%
%   Marcelo G Mattar (mmattar@princeton.edu)    Jan 2017


%% SET PARAMETERS (using the old setParams.m FOR COMPATIBILITY)

% MDP PARAMETERS
params.gamma            = 0.9; % discount factor
params.rewMag           = 1; % reward magnitude (can be a vector -- e.g. [1 0.1])
params.rewSTD           = 0.1; % reward standard deviation (can be a vector -- e.g. [1 0.1])
params.probNoReward     = 0; % probability of receiving no reward
params.TLearnRate       = 0.9; % learning rate for the transition matrix (0=uniform; 1=only last)
params.actPolicy        = 'softmax'; % Choose 'thompson_sampling' or 'e_greedy' or 'softmax'
params.epsilon          = 0.05; % probability of a random action (epsilon-greedy)
params.softmaxT         = 0.2; % soft-max temperature
params.preExplore       = true; % freely explore the maze without rewards to learn transition model
params.add_goal2start   = true; % Include a transition from goal to start in transition matrix -- this allows Need-term to wrap around

% LEARNING PARAMETERS - DETERMINISTIC
params.alpha            = 1.0; % learning rate for real experience (non-bayesian)
params.lambda           = 0; % eligibility trace parameter

% LEARNING PARAMETERS - BAYESIAN
params.bayesVersion     = false; % Specify whether to use Kalman-TD for learning Q-values
params.priorQvar        = 0;%.01; % Prior variance of the Q-values
params.theta_noise      = 0;%.0001; % Variance of the evolution noise (i.e. Q-values follow a random walk, which means that they get more and more uncertain with time). The assumption of a process noise for the evolution of Q-values is necessary because we utilize this framework for the learning of Q-values in a non-stationary MDP, i.e., the reward function of the environment might change over time
params.reward_noise     = 0;%.01; % Variance of the observation noise (how much should you trust in an observed reward). Rewards are modeled as a Gaussian variance with this variance.

% DEFINE PLANNING PARAMETERS
params.nPlan            = 20; % number of steps to do in planning (set to zero if no planning or to Inf to plan for as long as it is worth it)
params.planOnlyAtGorS   = true; % boolean variable indicating if planning should happen only if the agent is at the start or goal state

params.expandFurther    = true; % Expand the last backup further
params.nSteps           = 1; % Number of planning steps to evaluate in forward replay
params.nStepForAll      = false; % boolean variable indicating whether n-step planning is to be done for all states or only for current
params.EVMdivideByN     = true; % Divide the n-step EVM by the number of steps necessary to complete
params.planPolicy       = 'softmax'; % Choose 'thompson_sampling' or 'e_greedy' or 'softmax'
params.updIntermStates  = true; % Update intermediate states when performing n-step backup

params.copyQinPlanBkps  = false; % Copy the Q-value (mean and variance) on planning backups (i.e., LR=1.0)
params.copyQinGainCalc  = true; % Copy the Q-value (mean and variance) on gain calculation (i.e., LR=1.0)
params.allowLoops       = false; % Allow planning trajectories that return to a location appearing previously in the plan
params.remove_samestate = true; % Remove actions whose consequences lead to same state (e.g. hitting the wall)
params.EVMthresh        = 0; % minimum EVM so that planning is performed (use Inf if wish to use opportunity cost)
params.baselineGain     = 1e-10; % Gain is set to at least this value (interpreted as "information gain")

params.minGain          = 1e-10; % replay must have at least this Gain (to avoid precision errors); if gain is lower than this value, it is set to zero
params.onlineVSoffline  = 'online'; % Choose 'online' or 'offline' to determine what to use as the need-term
params.setAllGainToOne  = false; % Set the gain term of all items to one (for illustration purposes)
params.setAllNeedToOne  = false; % Set the need term of all items to one (for illustration purposes)

% PLOTTING SETTINGS
params.PLOT_STEPS       = false; % Plot each step of real experience
params.PLOT_Qvals       = false; % Plot Q-values
params.PLOT_PLANS       = false; % Plot each planning step
params.PLOT_EVM         = false; % Plot need and gain
params.PLOT_TRACE       = false; % Plot all planning traces
params.PLOT_wait        = 2 ; % Number of full episodes completed before plotting
params.delay            = 0.5; % How long to wait between successive plots
params.colormap         = b2r(-1,1); close all;

params.debugMode        = false;

params.arrowOffset      = 0.25;
params.arrowSize        = 100;
params.agentSize        = 15;
params.agentColor       = [0.5 0.5 0.5];
params.startendSymbSize = 24;


%% INITIALIZE VARIABLES

% get the initial maze dimensions:
[sideII,sideJJ] = size(params.maze);

% maximal number of states:
nStates = sideII*sideJJ;

% Transition count
T = zeros(nStates,nStates);

% on each grid we can choose from among at most this many actions:
nActions = 4; % 1=UP; 2=DOWN; 3=RIGHT; 4=LEFT

% Matrices with all states and actions with right index
allStates = repmat((1:nStates)',1,nActions);
allActions = repmat(1:nActions,nStates,1);

% Other parameters
Q = zeros(nStates,nActions); % State-action value function
if params.bayesVersion
    P0 = params.priorQvar * eye(nStates*nActions); % Prior covariance matrix
else
    P0 = zeros(nStates*nActions); % Set covariance matrix to zero
end
P = reshape(P0,nStates,nActions,nStates,nActions); % Covariance matrix
Qvar = reshape(diag(reshape(P,numel(Q),numel(Q))),nStates,nActions);

% Eligibility matrix
eTr = zeros(nStates,nActions);

% Last time each action has been taken
lastTimeActTaken = zeros(nStates,nActions);

% Arrays to hold the model of the environment (assuming it is deterministic):
Model_ns = nan(nStates,nActions); % <- next state
Model_nr = nan(nStates,nActions); % <- next reward

% Create an array to hold individual experiences
expList = nan(0, 4);

nBackups = 0; % <- keep track of how many backups were performed
numEpisodes = 0; % <- keep track of how many times we reach the end of our maze

% keep track of how many timestep we take per episode
ets = []; ts=0;

% keep track of how many times we have solved our problem in this number of timesteps:
cr = zeros(1,params.MAX_N_STEPS+1); cr(1) = 0;

% clear debug file
if params.debugMode
    fileID = fopen('debug.txt','wt');
    fclose(fileID);
end

% Preallocate output variables
simData.Q = cell(params.MAX_N_STEPS,1);
simData.T = cell(params.MAX_N_STEPS,1);
simData.replay.gain = cell(params.MAX_N_STEPS,1);
simData.replay.need = cell(params.MAX_N_STEPS,1);
simData.replay.EVM = cell(params.MAX_N_STEPS,1);
simData.replay.backupsQvals = cell(params.MAX_N_STEPS,1);
simData.replay.state = cell(params.MAX_N_STEPS,1);
simData.replay.action = cell(params.MAX_N_STEPS,1);
simData.replay.backups = cell(params.MAX_N_STEPS,1);
simData.replay.SR = cell(params.MAX_N_STEPS,1);
simData.replay.saGain = cell(params.MAX_N_STEPS,1);
simData.numEpisodes = nan(params.MAX_N_STEPS,1);
simData.backupDeltas = [];


%% PRE-EXPLORE MAZE (have the animal freely explore the maze without rewards to learn action consequences)
if params.preExplore
    for sti=1:nStates
        for at=1:nActions
            % Sample action consequences (minus reward, as experiment didn't 'start' yet)
            [st(1),st(2)] = ind2sub( [sideII,sideJJ], sti );
            if (params.maze(st(1),st(2)) == 0) && ~ismember([st(1) st(2)],params.s_end,'rows') % Don't explore walls or goal state (if goal state is included, the agent will be able to replay the experience of performing the various actions at the goal state)
                [~,~,stp1i] = stNac2stp1Nr(st,at,params); % state and action to state plus one and reward
                Model_ns(sti,at) = stp1i; % Update state-action-state model
                Model_nr(sti,at) = 0; % Update state-action-reward model
                expList(size(expList,1)+1,:) = [sti,at,0,stp1i]; % Update list of experiences
                T(sti,stp1i) = T(sti,stp1i) + 1; % Update transition matrix
            end
        end
    end
end
% Normalize so that rows sum to one
T = T./repmat(nansum(T,2),1,size(T,1));
T(isnan(T)) = 0; % Dividing by zero (i.e., no transitions to wall states are made) causes NaNs
% Add transitions from goal states to start states
if params.add_goal2start
    for i=1:size(params.s_end,1)
        if ~params.s_start_rand
            startnum = i+1;
            if startnum>size(params.s_start,1); startnum=1; end;
            gi = sub2ind(size(params.maze),params.s_end(i,1),params.s_end(i,2));
            bi = sub2ind(size(params.maze),params.s_start(startnum,1),params.s_start(startnum,2));
            T(gi,:) = 0; % Transitions from goal to anywhere else: 0
            T(gi,bi) = 1; % Transitions from goal to start else: 1
        else
            validStates = find(params.maze==0); % can start at any non-wall state
            validStates = validStates(~ismember(validStates,sub2ind(size(params.maze),params.s_end(:,1),params.s_end(:,2)))); %... but remove the goal states from the list
            goalState = sub2ind(size(params.maze),params.s_end(i,1),params.s_end(i,2));
            T(goalState,validStates) = 1/numel(T(goalState,validStates));
        end
    end
end

% move the agent to the (first) starting state
if ~params.s_start_rand
    st = params.s_start(1,:);
    sti = sub2ind( [sideII,sideJJ], st(1), st(2) );
else % Start at a random state
    validStates = find(params.maze==0); % can start at any non-wall state
    validStates = validStates(~ismember(validStates,sub2ind(size(params.maze),params.s_end(:,1),params.s_end(:,2)))); %... but remove the goal states from the list
    sti = validStates(randi(numel(validStates)));
    st = nan(1,2);
    [st(1),st(2)] = ind2sub( [sideII,sideJJ], sti );
end


%% EXPLORE MAZE
for tsi=1:params.MAX_N_STEPS
    
    %if mod(tsi,max(1000,params.MAX_N_STEPS/100))==0; fprintf('%d steps\n',tsi); end;
    if mod(tsi,100)==0; fprintf('%d steps\n',tsi); end;
    
    %% PLOT AGENT LOCATION
    if params.PLOT_STEPS && (numEpisodes>=params.PLOT_wait)
        figure(1); clf;
        if params.PLOT_Qvals
            plotMazeWithArrows(st,max(Q,[],2),min(Q(:),params.rewMag),params,'b2r(-1,1)','b2r(-1,1)')
        else
            plotMazeWithArrows(st,max(Q,[],2),nan(size(Q(:))),params,'b2r(-1,1)','b2r(-1,1)')
        end
        set(gcf,'Position',[1,535,560,420])
    end
    
    
    %% ACTION SELECTION
    probs = pAct(Q(sti,:),Qvar(sti,:),params.actPolicy,params);
    at = find(rand > [0 cumsum(probs)],1,'last');
    
    
    %% PERFORM ACTION
    % Move to state stp1 and collect reward
    [rew,stp1,stp1i] = stNac2stp1Nr(st,at,params); % state and action to state plus one and reward
    %{
    %if or(stp1i==3,stp1i==28)
    if and(stp1i==28,numEpisodes>=2)
        rew = -1;
    end
    if and(sti==1,numEpisodes>2)
        display('a');
    end
    %}
    lastTimeActTaken(sti,at) = tsi;
    
    
    %% UPDATE MODEL, TRANSITION MATRIX, AND EXPERIENCE LIST
    % Update model of the environment (with only latest information)
    Model_ns(sti,at) = stp1i; % next state
    Model_nr(sti,at) = rew;  % next reward
    % Update transition matrix
    targVec = zeros(1,nStates); targVec(stp1i) = 1;
    T(sti,:) = T(sti,:) + params.TLearnRate*(targVec-T(sti,:));
    % Update list of experiences
    expList(size(expList,1)+1,:) = [sti,at,rew,stp1i];
    simData.expList(tsi,:) = [sti,at,rew,stp1i];
    
    
    %% UPDATE Q-VALUES (LEARNING)
    if ~params.bayesVersion
        delta = ( rew + params.gamma*max(Q(stp1i,:)) - Q(sti,at) );
        eTr(sti,at) = eTr(sti,at) + 1; % Update eligibility trace
        Q = Q + (params.alpha * eTr) * delta; % TD-learning
        eTr = eTr * params.lambda * params.gamma; % Decay eligibility trace
    else
        [Q,P,K,Qsigma] = KTDSARSA(sti,at,rew,stp1i,bestAction(Q(stp1i,:),squeeze(P(stp1i,:,stp1i,:))),Q,P,params);
    end
    QpreReplay = Q;
    nBackups = nBackups+1;
    
    
    %% PLANNING PREP
    p=1;
    if params.planOnlyAtGorS % Only do replay if either current or last trial was a goal state
        if tsi>1
            [prevII,prevJJ] = ind2sub(size(params.maze),[expList(end-1,4);expList(end-1,1)]); % previous state (notice that we include both start and end state of the previous timestep, because the last row of expList might be a transition from goal to start)
        else
            prevII = 0; prevJJ = 0;
        end
        [currII,currJJ] = ind2sub(size(params.maze),expList(end,4)); % current state
        if ~or(ismember([currII,currJJ],params.s_end,'rows'),any(ismember([prevII,prevJJ],params.s_end,'rows')))
        %if ~ismember([currII,currJJ],params.s_end,'rows')
            p=Inf; % Otherwise, no planning
        end
    end
    
    % Pre-allocate variable to store the last Q-value update realized
    planning_backups = nan(0,5); % List of planning backups (to be used for creating a plot with the full planning trajectory/trace)
    backupsGain = nan(0,1); % List of GAIN for backups executed
    backupsNeed = nan(0,1); % List of NEED for backups executed
    backupsEVM = nan(0,1); % List of EVM for backups executed
    backupsSAGain = nan(0,0); % List of gain terms for all actions at each replay step
    backupsQvals = nan(0,1); % Q-values at completion of each replay step
    backups_gainNstep = nan(0,2); % gain term for n-step backups
    
    
    %% PLANNING STEPS
    while and(p <= params.nPlan , or(rew~=0,numEpisodes>0)) % Skip planning before the first reward is encountered
        
        % Create a list of 1-step backups based on 1-step models
        modelExps = [allStates(:) allActions(:) Model_nr(:) Model_ns(:)];
        
        % Remove actions that have no corresponding entry in the models -- e.g. actions starting from invalid states
        modelExps = modelExps(~any(isnan(modelExps),2),:);
        
        % Remove actions that lead to same state (optional) -- e.g. hitting the wall
        if params.remove_samestate
            modelExps = modelExps(~(modelExps(:,1)==modelExps(:,4)),:);
        end
        
        % Create auxiliar variables
        nStepExps = cell(size(modelExps,1),1);
        nSteps = ones(size(modelExps,1),1); % nSteps is used to discount the value of the resulting state
        steps2complete = ones(size(modelExps,1),1); % steps2complete is the number of planning steps requires to complete this backup
        
        % Expand previous backup further
        if and(params.expandFurther,size(planning_backups,1)>0)
            sn=planning_backups(end,4); % Final state reached in the last planning step
            probs = pAct(Q(sn,:),Qvar(sn,:),params.actPolicy,params);
            an = find(rand > [0 cumsum(probs)],1,'last'); % Select action to append using the same action selection policy used in real experience
            snp1 = Model_ns(sn,an); % Resulting state from taking action an in state sn
            rn = Model_nr(sn,an); % Reward received on this step only
            nStepPrev = planning_backups(end,5);
            n = nStepPrev+1;
            seqStart = find(planning_backups(:,5)==1,1,'last');
            seqSoFar = planning_backups(seqStart:end,:);
            % If a model for (sn,an) is available
            if ~or(isnan(Model_ns(sn,an)),isnan(Model_nr(sn,an)))
                % Check whether a loop is formed
                next_step_is_repeated = ismember(snp1,[seqSoFar(:,1);seqSoFar(:,4)]);
                % PS: Notice that we can't enforce that planning is done
                % only when the next state is not repeated or don't form a
                % loop. The reason is that the next step needs to be
                % derived 'on-policy', otherwise the Q-values may not
                % converge.
                if or(params.allowLoops,~next_step_is_repeated) % If loops are not allowed and next state is repeated, don't do anything
                    % Include this experience for evaluation
                    seqUpdated = [seqSoFar;[sn an rn snp1 n]];
                    s = seqUpdated(1,1); % s: first state of the trajectory
                    a = seqUpdated(1,2); % a: first action of the trajectory
                    r = params.gamma.^(0:(n-1)) * seqUpdated(:,3); % Accumulated discounted reward
                    modelExps = [modelExps; [s,a,r,snp1]];
                    nSteps = [nSteps;n]; % nSteps is used to discount the value of the resulting state
                    steps2complete = [steps2complete;1]; % This backup takes only 1 step to complete (as opposed to an n-step backup)
                    nStepExps{size(modelExps,1)} = seqUpdated; % Notice that rew=rn here (only considers reward from this step)
                end
            end
        end
        
        % Create a list of n-step backups
        if params.nSteps>1
            for s=1:nStates
                if or(params.nStepForAll,sti==s)
                    for a=1:nActions
                        % So, right now we're creating an n-step trajectory starting at (s,a)
                        seqSoFar = [s a Model_nr(s,a) Model_ns(s,a)];
                        if and(s~=Model_ns(s,a),~isnan(Model_ns(s,a)))
                            for n=2:params.nSteps
                                sn = seqSoFar(end,4);
                                probs = pAct(Q(sn,:),Qvar(sn,:),params.actPolicy,params);
                                an = find(rand > [0 cumsum(probs)],1,'last'); % Select action to append using the same action selection policy used in real experience
                                rn = Model_nr(sn,an);
                                snp1 = Model_ns(sn,an);
                                seqSoFar = [seqSoFar; [sn an rn snp1]];
                                % Check whether next step is valid
                                next_step_is_invalid = or(isnan(Model_ns(sn,an)),isnan(Model_nr(sn,an)))';
                                % Check whether a loop is formed
                                next_step_is_repeated = ismember(snp1,seqSoFar(:,1));
                                if next_step_is_invalid || and(~params.allowLoops,next_step_is_repeated)
                                    break
                                else
                                    % Include this experience for evaluation
                                    r = params.gamma.^(0:(n-1)) * seqSoFar(:,3); % Accumulated discounted reward
                                    modelExps = [modelExps; [s,a,r,snp1]];
                                    nSteps = [nSteps;n]; % nSteps is used to discount the value of the resulting state
                                    steps2complete = [steps2complete;n]; % This backup takes only 1 step to complete (as opposed to an n-step backup)
                                    nStepExps{size(modelExps,1)} = seqSoFar; % Notice that rew=rn here (only considers reward from this step)
                                end
                            end
                        end
                    end
                end
            end
        end
        
        % Remove entries that require more steps than are available (essential)
        modelExps = modelExps(steps2complete<=(params.nPlan+1-p),:);
        nStepExps = nStepExps(steps2complete<=(params.nPlan+1-p),:);
        nSteps = nSteps(steps2complete<=(params.nPlan+1-p),:);
        steps2complete = steps2complete(steps2complete<=(params.nPlan+1-p),:);
        
        % Gain term
        [gain,pA_pre,pA_post, Qpre, Qpost] = gainTerm(Q,P,modelExps,nSteps,params); % Considers the gain of updating modelExps(:,[1,2]) with info from modelExps(:,[3,4])
        gain(gain<params.minGain) = 0; % Set tiny gains to zero, as pAct() calculated probabilities only up to some precision
        if params.setAllGainToOne
            gain = ones(size(gain)); % Set the gain term of all items to one (for illustration purposes)
        end
        % Apply baseline gain (any replay have at least this gain)
        gain = max(gain,params.baselineGain);
        saGain = nan(size(Q));
        for s=1:size(saGain,1)
            for a=1:size(saGain,2)
                if sum(and(modelExps(:,1)==s,modelExps(:,2)==a))>0
                    saGain(s,a) = max(gain(and(modelExps(:,1)==s,modelExps(:,2)==a)));
                end
            end
        end
        
        % Need term
        [need,SR] = needTerm(sti,T,modelExps,params);
        if params.setAllNeedToOne
            need = ones(size(need)); % Set the need term of all items to one (for illustration purposes)
        end
        
        % Expected value of memories
        EVM = need.*gain;
        % Divide EVM by the number of steps in the n-step trajectory
        if params.EVMdivideByN
            EVM = EVM ./ steps2complete;
        end
        
        % DEBUGGING
        displayText = false;
        if numel(params.maze)==30
            linearMaze = true;
        else
            linearMaze = false;
        end
        if linearMaze
            if ismember(stp1,params.s_end,'rows') && p==1
                allQs = reshape(max(Q,[],2),3,10);
                if sti==25
                    if displayText
                        fprintf('Received reward: %.4f\n', rew*params.gamma);
                        fprintf('Current Q-value: %.4f\n', Q(22,at));
                        fprintf('EVM: %.8f (G=%.8f x N=%.8f)\n', EVM(and(modelExps(:,1)==22,modelExps(:,2)==at)),gain(and(modelExps(:,1)==22,modelExps(:,2)==at)),need(and(modelExps(:,1)==22,modelExps(:,2)==at)));
                        allQs(22) = -1;
                    end
                    simData.backupDeltas = [simData.backupDeltas; Q(22,at) rew*params.gamma gain(and(modelExps(:,1)==22,modelExps(:,2)==at)) (EVM(and(modelExps(:,1)==22,modelExps(:,2)==at))==max(EVM))];
                elseif sti==6
                    if displayText
                        fprintf('Received reward: %.4f\n', rew*params.gamma);
                        fprintf('Current Q-value: %.4f\n', Q(9,at));
                        fprintf('EVM: %.8f (G=%.8f x N=%.8f)\n', EVM(and(modelExps(:,1)==9,modelExps(:,2)==at)),gain(and(modelExps(:,1)==9,modelExps(:,2)==at)),need(and(modelExps(:,1)==9,modelExps(:,2)==at)));
                        allQs(9) = -1;
                    end
                    simData.backupDeltas = [simData.backupDeltas; Q(9,at) rew*params.gamma gain(and(modelExps(:,1)==9,modelExps(:,2)==at)) (EVM(and(modelExps(:,1)==9,modelExps(:,2)==at))==max(EVM))];
                end
                if displayText
                    display(allQs);
                    display('');
                end
            end
        end
        
        if params.PLOT_EVM && (double(rew~=0)+numEpisodes>=params.PLOT_wait)
            figure(2); clf;
            plotMazeWithArrows(st,SR(sti,:)',4*saGain(:),params,'YlGnBu','gray'); % Notice the scaling factor, causing Gain=0.25 to be colored as yellow
            set(gcf,'Position',[562,535,560,420])
            
            % Make GIF
            if sti==Inf
                frame = getframe(2);
                im = frame2im(frame);
                [imind,cm] = rgb2ind(im,256);
                if ~exist('OpenReverse1b.gif','file')
                    imwrite(imind,cm,'OpenReverse1b.gif','gif', 'Loopcount',inf);
                else
                    imwrite(imind,cm,'OpenReverse1b.gif','gif','WriteMode','append');
                end
            end
        end
        
        %% PERFORM THE UPDATE
        opportCost = cr(tsi)/tsi; % Average expected reward from a random act
        EVMthresh = min(opportCost,params.EVMthresh); % if EVMthresh==Inf, threshold is opportCost
        
        if params.debugMode
            fileID = fopen('debug.txt','a');
            fprintf(fileID,'tsi = %d; ; st = [%d,%d]; p = %d; EVMthresh = %.4f\n',tsi, st(1), st(2), p, EVMthresh);
            fprintf(fileID,'EVM = %.4f; Gain = %.4f; Need = %.4f\n',max(EVM), gain(EVM(:)==max(EVM)),need(EVM(:)==max(EVM)));
            fclose(fileID);
        end
        
        if max(EVM) > EVMthresh
            % Identify state-action pairs with highest priority
            maxEVM_idx = find(EVM(:)==max(EVM));
            if numel(maxEVM_idx)>1 % If there are multiple items with equal gain
                maxEVM_idx = maxEVM_idx(nSteps(maxEVM_idx) == max(nSteps(maxEVM_idx))); % Select the one corresponding to a longer trajectory
                if numel(maxEVM_idx)>1 % If there are still multiple items with equal gain (and equal length)
                    maxEVM_idx = maxEVM_idx(randi(numel(maxEVM_idx))); % ... select one at random
                end
            end
            
            if all(gain<1e-9)
                numEpisodes;
            end
            
            % Plot planning steps
            if params.PLOT_PLANS && (double(rew>0)+numEpisodes>=params.PLOT_wait)
                figure(1); clf;
                
                highlight = zeros(size(Q(:)));
                if nSteps(maxEVM_idx)>1 && steps2complete(maxEVM_idx)==1 % Highlight extended trace
                    highlight(sub2ind(size(Q),nStepExps{maxEVM_idx}(:,1),nStepExps{maxEVM_idx}(:,2))) = 1;
                else % Highlight only the first action from the trace
                    highlight(sub2ind(size(Q),modelExps(maxEVM_idx,1),modelExps(maxEVM_idx,2))) = 1;
                end
                
                % Plot
                if params.PLOT_Qvals
                    plotMazeWithArrows(st,max(Q,[],2),min(Q(:),max(params.rewMag)),params,'b2r(-1,1)','b2r(-1,1)',highlight,rew)
                else
                    plotMazeWithArrows(st,max(Q,[],2),nan(size(Q(:))),params,'b2r(-1,1)','b2r(-1,1)',highlight,rew)
                end
                set(gcf,'Position',[1,535,560,420])
                
                for n=2:steps2complete(maxEVM_idx)
                    highlight(sub2ind(size(Q),nStepExps{maxEVM_idx}(n,1),nStepExps{maxEVM_idx}(n,2))) = 1;
                    if params.PLOT_Qvals
                        plotMazeWithArrows(st,max(Q,[],2),min(Q(:),params.rewMag),params,'b2r(-1,1)','b2r(-1,1)',highlight)
                    else
                        plotMazeWithArrows(st,max(Q,[],2),nan(size(Q(:))),params,'b2r(-1,1)','b2r(-1,1)',highlight)
                    end
                    set(gcf,'Position',[1,535,560,420])
                end
                
                % Make GIF
                if sti==Inf
                    frame = getframe(1);
                    im = frame2im(frame);
                    [imind,cm] = rgb2ind(im,256);
                    if ~exist('OpenReverse1a.gif','file')
                        imwrite(imind,cm,'OpenReverse1a.gif','gif', 'Loopcount',inf);
                    else
                        imwrite(imind,cm,'OpenReverse1a.gif','gif','WriteMode','append');
                    end
                end
            end
            
            % Retrieve information from this experience
            s_plan = modelExps(maxEVM_idx,1);
            a_plan = modelExps(maxEVM_idx,2);
            r_plan = modelExps(maxEVM_idx,3);
            stp1_plan = modelExps(maxEVM_idx,4);
            atp1_plan = bestAction(Q(stp1_plan,:),squeeze(P(stp1_plan,:,stp1_plan,:)));
            n_plan = nSteps(maxEVM_idx);
            
            % Update Q-value of the first action
            if params.copyQinPlanBkps
                Q(s_plan,a_plan) = r_plan + (params.gamma^n_plan)*max(Q(stp1_plan,:));
                P(s_plan,a_plan,s_plan,a_plan) = params.reward_noise + ((params.gamma.^n_plan).^2) .* P(stp1_plan,atp1_plan,stp1_plan,atp1_plan);
            else
                if params.bayesVersion
                    [Q,P,K,Qsigma] = KTDSARSA(s_plan,a_plan,r_plan,stp1_plan,atp1_plan,Q,P,params,0,n_plan);
                else
                    Qtarget = r_plan + (params.gamma^n_plan)*max(Q(stp1_plan,:));
                    Q(s_plan,a_plan) = Q(s_plan,a_plan) + params.alpha * (Qtarget-Q(s_plan,a_plan));
                    % PS: Not necessary to update P, which is all zeros anyways
                end
            end
            
            if params.updIntermStates && (n_plan>1)
                % Update intermediate action values
                for n=2:n_plan
                    % Retrieve experience information
                    s_planN = nStepExps{maxEVM_idx}(n,1);
                    a_planN = nStepExps{maxEVM_idx}(n,2);
                    rewToEnd = nStepExps{maxEVM_idx}(n:end,3);
                    r_planN = (params.gamma.^(0:(length(rewToEnd)-1))) * rewToEnd;
                    n_planN = length(rewToEnd);
                    
                    % Update Q-values of the intermediate actions
                    % Notice that these updates use stp1_plan and atp1_plan, which correspond to the end of the n-step trajectory
                    if params.copyQinPlanBkps
                        Q(s_planN,a_planN) = r_planN + (params.gamma^n_planN)*max(Q(stp1_plan,:));
                        P(s_planN,a_planN,s_planN,a_planN) = params.reward_noise + ((params.gamma^n_planN).^2) .* P(stp1_plan,atp1_plan,stp1_plan,atp1_plan);
                    else
                        if params.bayesVersion
                            [Q,P,K,Qsigma] = KTDSARSA(s_planN,a_planN,r_planN,stp1_plan,atp1_plan,Q,P,params,0,n_planN);
                        else
                            Qtarget = r_planN + (params.gamma^n_planN)*max(Q(stp1_plan,:));
                            Q(s_planN,a_planN) = Q(s_planN,a_planN) + params.alpha * (Qtarget-Q(s_planN,a_planN));
                            % PS: Not necessary to update P, which is all zeros anyways
                        end
                    end
                end
            end
            
            if params.debugMode
                fileID = fopen('debug.txt','a');
                fprintf(fileID,'[%d, %d, %.4f %d]; n=%d\n',s_plan,a_plan,r_plan,stp1_plan,n_plan);
                fclose(fileID);
            end
            
             % List of planning backups (to be used in creating a plot with the full planning trajectory/trace)
            backupsGain = [backupsGain gain(maxEVM_idx)]; % List of GAIN for backups executed
            backupsNeed = [backupsNeed need(maxEVM_idx)]; % List of NEED for backups executed
            backupsEVM = [backupsEVM EVM(maxEVM_idx)]; % List of EVM for backups executed
            backupsQvals = [backupsQvals Q(:)]; % Existing Q-values upon completion of this step
            backupsSAGain = [backupsSAGain saGain(:)]; % List of gain terms for all actions at each replay step
            if nSteps(maxEVM_idx)==1
                planning_backups = [planning_backups; [modelExps(maxEVM_idx,1:4) nSteps(maxEVM_idx,1)]];
            else
                if steps2complete(maxEVM_idx)==1
                    planning_backups = [planning_backups; [nStepExps{maxEVM_idx}(end,1:4) nSteps(maxEVM_idx,1)]];
                else
                    planning_backups = [planning_backups; [nStepExps{maxEVM_idx} (1:steps2complete(maxEVM_idx))']];
                end
                
                % Save gain
                backups_gainNstep = [backups_gainNstep; nSteps(maxEVM_idx,1) gain(maxEVM_idx)];
            end
            p = p + steps2complete(maxEVM_idx);
        else
            break
        end
        
    end
    
    if and(~isinf(p),params.debugMode)
        fileID = fopen('debug.txt','a');
        fprintf(fileID,'\n');
        fclose(fileID);
    end
    
    % Plot planning traces
    %if params.PLOT_TRACE && ~isempty(planning_backups)
    if params.PLOT_TRACE && (double(rew>0)+numEpisodes>=params.PLOT_wait)
        figure(3);  clf; hold on;
        plotReplayTrace(st,planning_backups,params,hot(size(planning_backups,1)+10));
        set(gcf,'Position',[1123,535,560,420])
        %print(['./simulations/' num2str(tsi) '.pdf'],'-dpdf')
        %pause
    end
    
    
    %% MOVE AGENT TO NEXT STATE
    st = stp1; sti = stp1i;
    
    
    %% COMPLETE STEP
    ts=ts+1; % Timesteps to solution (reset to zero at the end of the episode)
    cr(tsi+1) = cr(tsi)+rew; % Cumulative reward
    
    if ismember(st,params.s_end,'rows') % agent is at a terminal state
        if params.s_start_rand
            % Start at a random state
            validStates = find(params.maze==0); % can start at any non-wall state
            validStates = validStates(~ismember(validStates,sub2ind(size(params.maze),params.s_end(:,1),params.s_end(:,2)))); %... but remove the goal states from the list
            stp1i = validStates(randi(numel(validStates)));
            [stp1(1),stp1(2)] = ind2sub( [sideII,sideJJ], stp1i );
        else
            % Return to start
            goalnum = find(all(repmat(st,size(params.s_end,1),1)==params.s_end,2));
            startnum = goalnum+1;
            if startnum>size(params.s_start,1); startnum=1; end;
            stp1 = params.s_start(startnum,:);
            stp1i = sub2ind( [sideII,sideJJ], stp1(1), stp1(2) );
        end
        % Update transition matrix and list of experiences
        if params.add_goal2start
            targVec = zeros(1,nStates); targVec(stp1i) = 1;
            T(sti,:) = T(sti,:) + params.TLearnRate*(targVec-T(sti,:)); % Update transition matrix
            expList(size(expList,1)+1,:) = [sti,nan,nan,stp1i]; % Update list of experiences
        end
        st = stp1; sti = stp1i; % Move the agent to the start location
        
        % record that we took "ts" timesteps to get to the solution (end state)
        ets = [ets; ts]; ts=0;
        % record that we got to the end:
        numEpisodes = numEpisodes+1;
        
        % Reset eligibility matrix
        eTr = zeros(size(eTr));
    end
    
    
    %% SAVE SIMULATION DATA
    simData.numEpisodes(tsi) = numEpisodes;
    simData.Q{tsi} = QpreReplay;
    simData.T{tsi} = T;
    simData.stepsPerEpisode = ets;
    
    if numel(backupsEVM)>0
        simData.replay.gain{tsi} = backupsGain;
        simData.replay.need{tsi} = backupsNeed;
        simData.replay.EVM{tsi} = backupsEVM;
        simData.replay.backupsQvals{tsi} = backupsQvals;
        simData.replay.backupsSAGain{tsi} = backupsSAGain;
        simData.replay.state{tsi} = planning_backups(:,1)';
        simData.replay.action{tsi} = planning_backups(:,2)';
        simData.replay.backups{tsi} = planning_backups;
        simData.replay.SR{tsi} = SR;
        simData.replay.backups_gainNstep{tsi} = backups_gainNstep;
        saGain = nan(size(Q));
        for s=1:size(saGain,1)
            for a=1:size(saGain,2)
                if sum(and(modelExps(:,1)==s,modelExps(:,2)==a))>0
                    saGain(s,a) = max(gain(and(modelExps(:,1)==s,modelExps(:,2)==a)));
                end
            end
        end
        simData.replay.saGain{tsi} = saGain;
    end
    assert(size(simData.expList,1)==tsi,'expList has incorrect size')
    
    % If max number of episodes is reached, trim down simData.replay
    if numEpisodes==params.MAX_N_EPISODES
        simData.replay.gain = simData.replay.gain(1:tsi);
        simData.replay.need = simData.replay.need(1:tsi);
        simData.replay.EVM = simData.replay.EVM(1:tsi);
        simData.replay.backupsQvals = simData.replay.backupsQvals(1:tsi);
        simData.replay.backupsSAGain = simData.replay.backupsSAGain(1:tsi);
        simData.replay.state = simData.replay.state(1:tsi);
        simData.replay.action = simData.replay.action(1:tsi);
        simData.replay.backups = simData.replay.backups(1:tsi);
        simData.replay.SR = simData.replay.SR(1:tsi);
        simData.replay.backups_gainNstep = simData.replay.backups_gainNstep(1:tsi);
        simData.replay.saGain = simData.replay.saGain(1:tsi);
        simData.numEpisodes = simData.numEpisodes(1:tsi);
        break
    end
    
end




function probs = pAct(Qmean,Qvar,actPolicy,params)
%PACT           Calculate the probability of executing each action.
%   Qmean: |A|x1 array with the mean Q-value for each action
%   Qvar: |A|x1 array with the variance of the Q-value for each action
%   actPolicy: policy used to calculate the probabilities
%   params: simulation parameters
%
%   Marcelo G Mattar (mmattar@princeton.edu)    May 2017

%% INITIALIZE VARIABLES
probs = nan(size(Qmean));


%% IMPLEMENT ACTION SELECTION STRATEGY

switch actPolicy
    case 'e_greedy'
        for s=1:size(Qmean,1)
            Qbest = find(Qmean(s,:)==max(Qmean(s,:)));
            %{
            if numel(Qbest)>1 % If there are multiple actions with max Q...
                Qbest = Qbest(randi(numel(Qbest),1)); % pick each with equal probability
            end
            probs(s,:) = repmat(params.epsilon/size(probs,2),1,size(probs,2));
            probs(s,Qbest) = probs(s,Qbest) + (1-params.epsilon);
            %}
            if all(Qmean(s,:)==max(Qmean(s,:)))
                probs(s,Qbest) = 1/numel(Qbest);
            else
                probs(s,Qbest) = (1-params.epsilon)/numel(Qbest);
                probs(s,Qmean(s,:)<max(Qmean(s,:))) = params.epsilon/sum(Qmean(s,:)<max(Qmean(s,:)));
            end
        end
    case 'softmax'
        for s=1:size(Qmean,1)
            if any(isinf(exp(Qmean(s,:)./params.softmaxT)))
                epsilon = 0;
                Qbest = find(Qmean(s,:)==max(Qmean(s,:)));
                if numel(Qbest)>1 % If there are multiple actions with max Q...
                    Qbest = Qbest(randi(numel(Qbest),1)); % pick each with equal probability
                end
                probs(s,:) = repmat(epsilon/size(probs,2),1,size(probs,2));
                probs(s,Qbest) = probs(s,Qbest) + (1-epsilon);
            else
                probs(s,:) = exp(Qmean(s,:)./params.softmaxT) ./ sum(exp(Qmean(s,:)./params.softmaxT));
            end
        end
    case 'VPI'
        probs=1;
    case 'thompson_sampling'
        xmax = max(max(Qmean + 10*sqrt(Qvar)));
        allActions = 1:size(Qmean,2);
        dx = xmax/10;
        while max(abs(nansum(probs,2)-1))>1e-10
            dx = dx/10;
            X = -xmax:dx:xmax;
            for s=1:size(Qmean,1)
                if ~all(Qvar(s,:)==0)
                    for a=1:size(Qmean,2)
                        prodCDFs = prod(normcdf(X,repmat(Qmean(s,allActions(allActions~=a))',1,length(X)),repmat(sqrt(Qvar(s,allActions(allActions~=a)))',1,length(X))));
                        thisPDF = normpdf(X,Qmean(s,a),eps+sqrt(Qvar(s,a)));
                        probs(s,a) = sum((prodCDFs .* thisPDF) * dx);
                    end
                else
                    probs(s,Qmean(s,:) == max(Qmean(s,:))) = 1./sum(Qmean(s,:) == max(Qmean(s,:)));
                    probs(s,Qmean(s,:) ~= max(Qmean(s,:))) = 0;
                end
            end
        end
    otherwise
        error('Unrecognized strategy');
end

assert(all(abs(sum(probs,2)-1)<1e-3), 'Probabilities do not sum to one');





function [gain, pA_pre, pA_post, Qpre, Qpost] = gainTerm(Q,P,expList,nSteps,params)

Pvar = reshape(diag(reshape(P,numel(Q),numel(Q))),size(Q,1),size(Q,2)); % P=Var(Q)

Qmean = Q(expList(:,1),:); Qpre = Qmean;
Qvar = Pvar(expList(:,1),:);
% Calculate probabilities BEFORE backups
pA_pre = pAct(Qmean,Qvar,params.planPolicy,params);

% Find the mean and variance of the value of stp1
stp1Means = max(Q(expList(:,4),:),[],2); % Value of state stp1
bestActions = Q(expList(:,4),:) == max(Q(expList(:,4),:),[],2); % Logical matrix with one entry per action (=1 if action is one of the optimal ones; =0 if not)
stp1Vars = max(Pvar(expList(:,4),:) .* bestActions,[],2); % The variance of stp1 is the maximum variance of the action(s) that have highest mean (notice that this assumes that the best action in the new state is the one with highest value AND highest variance)

% Calculate means and variances AFTER backups
if params.copyQinGainCalc
    actTaken = sub2ind(size(expList), (1:size(expList,1))', expList(:,2)); % index of expList matrix corresponding to the action taken (i.e., 2nd column)
    Qmean(actTaken) = expList(:,3) + (params.gamma.^nSteps) .* stp1Means; % The new mean is [R + (gamma^n)*V(stp1)]
    Qvar(actTaken) = params.reward_noise + ((params.gamma.^nSteps).^2) .* stp1Vars; % The new variance is [Var(R) + (gamma^2n)*Var(stp1)]
else
    if params.bayesVersion
        for e=1:size(expList,1)
            atp1_plan = bestAction(Q(expList(e,4),:),squeeze(P(expList(e,4),:,expList(e,4),:)));
            [Qktd,Pktd,~,~] = KTDSARSA(expList(e,1),expList(e,2),expList(e,3),expList(e,4),atp1_plan,Q,P,params,0,nSteps(e));
            Qmean(e,expList(e,2)) = Qktd(expList(e,1),expList(e,2));
            Qvar(e,expList(e,2)) = Pktd(expList(e,1),expList(e,2),expList(e,1),expList(e,2));
        end
    else
        actTaken = sub2ind(size(expList), (1:size(expList,1))', expList(:,2)); % index of expList matrix corresponding to the action taken (i.e., 2nd column)
        Qtarget = expList(:,3) + (params.gamma.^nSteps) .* stp1Means;
        Qmean(actTaken) = Qmean(actTaken) + params.alpha * (Qtarget-Qmean(actTaken));
    end
end

% Calculate probabilities AFTER backups
pA_post = pAct(Qmean,Qvar,params.planPolicy,params);

% Calculate gain
EVpre = sum(pA_pre .* Qmean,2);
EVpost = sum(pA_post .* Qmean,2);
gain = EVpost-EVpre;
Qpost = Qmean;






function [need,SR,SD] = needTerm(sti,T,expList,params)

% Set diagonal to zero
T(logical(eye(size(T)))) = 0;

% Normalize rows to sum to 1
rowTotal = nansum(T,2);
rowTotal(rowTotal~=0) = 1./rowTotal(rowTotal~=0);
T = T.* repmat(rowTotal,1,numel(rowTotal));

switch params.onlineVSoffline
    
    case 'online' % Calculate the successor representation
        
        % Calculate Successor Representation
        SR = inv(   eye(size(T)) - params.gamma*T   );
        SRi = SR(sti,:); % Calculate the Successor Representation for the current state
        
        % Calculate need-term for each experience in expList
        need = SRi(expList(:,1))';
        
        % Use NaNs for the output SD
        SD = nan(size(SRi));
        
    case 'offline'
        
        % Calculate eigenvectors and eigenvalues
        [V,D,W] = eig(T);
        if abs(D(1,1))-1>1e-10
            display('a');
        end
        SD = abs(W(:,1)');
        
        % Calculate need-term for each experience in expList
        need = SD(expList(:,1))';
        
        %{
        [~,I] = min(abs(abs(diag(D))-1)); % Find eigenvalue closest to one
        
        % Check if a stable stationary distribution can be found
        diffVec = W(:,I)'*T - W(:,I)'; % Check if eigenvalue is one
        if (norm(diffVec)<1e-10)
            % Calculate the stationary distribution of the MDP
            SD = W(:,I)';
            %{
            if all(SD<=1e-10)
                SD = -SD;
            end
            if ~all(SD>=0)
                error('Not all elements of SD are greater than or equal to zero');
            end
            %}
            SD = abs(SD);
            
            % Calculate need-term for each experience in expList
            need = SD(expList(:,1))';
            
        else
            SD = nan(1,size(W,1));
            need = ones(size(expList(:,1)));
        end
        %}
        
        % Use NaNs for the output SR
        SR = nan(size(T));
end




function actIdx = bestAction(Q,P,max_vs_min_var)
%BESTACTION     Choose action with highest Q-value and maximum variance.
%   Q = (1xnActions) array containing Q-values
%   P = (nActionsxnActions) array containing covariance matrix
%       PS: note that only diagonal is used.
%
%   Marcelo G Mattar (mmattar@princeton.edu)    Jan 2017

if ~exist('max_vs_min_var','var')
    max_vs_min_var = 'max';
end

[X,I] = max(Q);
if sum(Q==max(Q))>1
    foo = find(Q==max(Q));
    switch max_vs_min_var
        case 'max'
            [X2,I2] = max(diag(P(foo,foo))); % Choose action with maximum variance
        case 'min'
            [X2,I2] = min(diag(P(foo,foo))); % Choose action with maximum variance
    end
    if sum(diag(P(foo,foo))==X2)>1 % If more than one element has maximum variance
        foo2 = foo(diag(P(foo,foo))==X2);
        actIdx = foo2(randi(numel(foo2))); % Pick one at random
    else
        actIdx = foo(I2);
    end
else
    actIdx = I;
end





function newmap = b2r(cmin_input,cmax_input,middle_color,color_num)
%BLUEWHITERED   Blue, white, and red color map.
%   this matlab file is designed to draw anomaly figures. the color of
%   the colorbar is from blue to white and then to red, corresponding to 
%   the anomaly values from negative to zero to positive, respectively. 
%   The color white always correspondes to value zero. 
%   
%   You should input two values like caxis in matlab, that is the min and
%   the max value of color values designed.  e.g. colormap(b2r(-3,5))
%   
%   the brightness of blue and red will change according to your setting,
%   so that the brightness of the color corresponded to the color of his
%   opposite number
%   e.g. colormap(b2r(-3,6))   is from light blue to deep red
%   e.g. colormap(b2r(-3,3))   is from deep blue to deep red
%
%   I'd advise you to use colorbar first to make sure the caxis' cmax and cmin.
%   Besides, there is also another similar colorbar named 'darkb2r', in which the 
%   color is darker.
%
%   by Cunjie Zhang, 2011-3-14
%   find bugs ====> email : daisy19880411@126.com
%   updated:  Robert Beckman help to fix the bug when start point is zero, 2015-04-08
%   
%   Examples:
%   ------------------------------
%   figure
%   peaks;
%   colormap(b2r(-6,8)), colorbar, title('b2r')
%   


%% check the input
if nargin ~= 2 ;
   %disp('input error');
   %disp('input two variables, the range of caxis , for example : colormap(b2r(-3,3))');
end

if cmin_input >= cmax_input
    disp('input error');
    disp('the color range must be from a smaller one to a larger one');
end

%% control the figure caxis 
lims = get(gca, 'CLim');   % get figure caxis formation
caxis([cmin_input cmax_input]);

%% color configuration : from blue to to white then to red

red_top     = [1 0 0];
if ~exist('middle_color','var')
    middle_color= [1 1 1];
end
blue_bottom = [0 0 1];

%% color interpolation 

if ~exist('color_num','var')
    color_num = 251;   
end
color_input = [blue_bottom;  middle_color;  red_top];
oldsteps = linspace(-1, 1, size(color_input,1));
newsteps = linspace(-1, 1, color_num);  

%% Category Discussion according to the cmin and cmax input

%  the color data will be remaped to color range from -max(abs(cmin_input),cmax_input)
%  to max(abs(cmin_input),cmax_input) , and then squeeze the color data
%  in order to make sure the blue and red color selected corresponded
%  to their math values

%  for example :
%  if b2r(-3,6) ,the color range is from light blue to deep red , so that
%  the light blue valued at -3 correspondes to light red valued at 3


%% Category Discussion according to the cmin and cmax input
% first : from negative to positive
% then  : from positive to positive
% last  : from negative to negative

for j=1:3
   newmap_all(:,j) = min(max(transpose(interp1(oldsteps, color_input(:,j), newsteps)), 0), 1);
end

if (cmin_input < 0)  &&  (cmax_input > 0) ;  
    
    
    if abs(cmin_input) < cmax_input 
         
        % |--------|---------|--------------------|    
      % -cmax      cmin       0                  cmax         [cmin,cmax]
 
       start_point = max(round((cmin_input+cmax_input)/2/cmax_input*color_num),1);
       newmap = squeeze(newmap_all(start_point:color_num,:));
       
    elseif abs(cmin_input) >= cmax_input
        
         % |------------------|------|--------------|    
       %  cmin                0     cmax          -cmin         [cmin,cmax]   
       
       end_point = max(round((cmax_input-cmin_input)/2/abs(cmin_input)*color_num),1);
       newmap = squeeze(newmap_all(1:end_point,:));
    end
    
       
elseif cmin_input >= 0

       if lims(1) < 0 
           disp('caution:')
           disp('there are still values smaller than 0, but cmin is larger than 0.')
           disp('some area will be in red color while it should be in blue color')
       end
       
        % |-----------------|-------|-------------|    
      % -cmax               0      cmin          cmax         [cmin,cmax]
 
       start_point = max(round((cmin_input+cmax_input)/2/cmax_input*color_num),1);
       newmap = squeeze(newmap_all(start_point:color_num,:));

elseif cmax_input <= 0

       if lims(2) > 0 
           disp('caution:')
           disp('there are still values larger than 0, but cmax is smaller than 0.')
           disp('some area will be in blue color while it should be in red color')
       end
       
         % |------------|------|--------------------|    
       %  cmin         cmax    0                  -cmin         [cmin,cmax]      

       end_point = max(round((cmax_input-cmin_input)/2/abs(cmin_input)*color_num),1);
       newmap = squeeze(newmap_all(1:end_point,:));
end
    


