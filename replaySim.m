function simData = replaySim(params)
%REPLAYSIM      Replay simulation on a grid-world.
%   params = Structure with all simulation parameters
%
%   Marcelo G Mattar (mmattar@princeton.edu)    Jan 2017


%% INITIALIZE VARIABLES
% get the initial maze dimensions:
[sideII,sideJJ] = size(params.maze);
nStates = sideII*sideJJ; % maximal number of states:
% on each state, four actions are available
nActions = 4; % 1=UP; 2=DOWN; 3=RIGHT; 4=LEFT

% Preallocate variables
Q = zeros(nStates,nActions); % State-action value function
T = zeros(nStates,nStates); % State-state transition probability
eTr = zeros(nStates,nActions); % Eligibility matrix
expList = nan(0,4); % Array to store individual experiences
expLast_stp1 = nan(nStates,nActions); % <- next state
expLast_rew = nan(nStates,nActions); % <- next reward
numEpisodes = 0; % <- keep track of how many times we reach the end of our maze
lastsize = 0; % Length of progress counter displayed on each iteration of the simulation
ets = []; ts = 0; % keep track of how many timestep we take per episode

simData.expList = nan(0,4);
simData.replay.state = cell(params.MAX_N_STEPS,1);
simData.replay.action = cell(params.MAX_N_STEPS,1);
simData.numEpisodes = nan(params.MAX_N_STEPS,1);

simData.replay.gain = cell(params.MAX_N_STEPS,1);
simData.replay.need = cell(params.MAX_N_STEPS,1);
simData.replay.EVM = cell(params.MAX_N_STEPS,1);

%{
???: Do I need these or not?
% keep track of how many timestep we take per episode
ets = []; ts=0;

% keep track of how many times we have solved our problem in this number of timesteps:
cr = zeros(1,params.MAX_N_STEPS+1); cr(1) = 0;

% Preallocate output variables
simData.Q = cell(params.MAX_N_STEPS,1);
simData.T = cell(params.MAX_N_STEPS,1);
simData.replay.backupsQvals = cell(params.MAX_N_STEPS,1);
simData.replay.state = cell(params.MAX_N_STEPS,1);
simData.replay.action = cell(params.MAX_N_STEPS,1);
simData.replay.backups = cell(params.MAX_N_STEPS,1);
simData.replay.SR = cell(params.MAX_N_STEPS,1);
simData.replay.saGain = cell(params.MAX_N_STEPS,1);
simData.numEpisodes = nan(params.MAX_N_STEPS,1);
simData.backupDeltas = [];
%}

%% PRE-EXPLORE MAZE (have the animal freely explore the maze without rewards to learn action consequences)
fprintf('\nStarting a new simulation...\n')
if params.preExplore
    for sti=1:nStates
        for at=1:nActions
            % Sample action consequences (minus reward, as experiment didn't 'start' yet)
            [st(1),st(2)] = ind2sub( [sideII,sideJJ], sti ); % Convert state index to (i,j)
            if (params.maze(st(1),st(2)) == 0) && ~ismember([st(1) st(2)],params.s_end,'rows') % Don't explore walls or goal state (if goal state is included, the agent will be able to replay the experience of performing the various actions at the goal state)
                [~,~,stp1i] = stNac2stp1Nr(st,at,params); % state and action to state plus one and reward
                expList(size(expList,1)+1,:) = [sti,at,0,stp1i]; % Update list of experiences
                expLast_stp1(sti,at) = stp1i; % stp1 from last experience of this state/action
                expLast_rew(sti,at) = 0; % rew from last experience of this state/action
                T(sti,stp1i) = T(sti,stp1i) + 1; % Update transition matrix
            end
        end
    end
end
% Normalize so that rows sum to one (ie build a proper transition function)
T = T./repmat(nansum(T,2),1,size(T,1));
T(isnan(T)) = 0; % Dividing by zero causes NaNs (i.e., transitions from wall states)
% Add transitions from goal states to start states
if params.Tgoal2start
    for i=1:size(params.s_end,1) % Loop through each goal state
        if ~params.s_start_rand
            gi = sub2ind(size(params.maze),params.s_end(i,1),params.s_end(i,2)); % goal state index
            bi = sub2ind(size(params.maze),params.s_start(mod(i,size(params.s_start,1))+1,1),params.s_start(mod(i,size(params.s_start,1))+1,2)); % beginning state index
            T(gi,:) = 0; % Transitions from goal to anywhere else: 0
            T(gi,bi) = 1; % Transitions from goal to start else: 1
        else % If starting locations are randomized
            validStates = find(params.maze==0); % can start at any non-wall state
            validStates = validStates(~ismember(validStates,sub2ind(size(params.maze),params.s_end(:,1),params.s_end(:,2)))); %... but remove the goal states from the list
            gi = sub2ind(size(params.maze),params.s_end(i,1),params.s_end(i,2)); % goal state index
            T(gi,:) = 0; % Transitions from this goal to anywhere else: 0
            T(gi,validStates) = 1/numel(T(gi,validStates)); % Transitions from this goal to anywhere else: uniform
        end
    end
end


%% PREPARE FIRST TRIAL
% Move the agent to the (first) starting state
if ~params.s_start_rand
    st = params.s_start(1,:);
    sti = sub2ind( [sideII,sideJJ], st(1), st(2) );
else % Start at a random state
    validStates = find(params.maze==0); % can start at any non-wall state
    validStates = validStates(~ismember(validStates,sub2ind(size(params.maze),params.s_end(:,1),params.s_end(:,2)))); %... but remove the goal states from the list
    sti = validStates(randi(numel(validStates)));
    [st(1),st(2)] = ind2sub( [sideII,sideJJ], sti );
end


%% EXPLORE MAZE
for tsi=1:params.MAX_N_STEPS
    if mod(tsi,10)==0
        fprintf(repmat('\b', 1, lastsize));
        lastsize = fprintf('%d steps; %d episodes', tsi, numEpisodes);
    end % Display progress every 100 steps
    
    
    %% PLOT AGENT LOCATION
    if params.PLOT_STEPS && (numEpisodes>=params.PLOT_wait)
        figure(1); clf;
        if params.PLOT_Qvals
            plotMazeWithArrows(st,max(Q,[],2),min(Q(:),max(params.rewMag(:))),params,'b2r(-1,1)','b2r(-1,1)')
        else
            plotMazeWithArrows(st,max(Q,[],2),nan(size(Q(:))),params,'b2r(-1,1)','b2r(-1,1)')
        end
        set(gcf,'Position',[1,535,560,420])
    end
    
    
    %% ACTION SELECTION
    probs = pAct(Q(sti,:),params.actPolicy,params); % Probability of executing each action
    at = find(rand > [0 cumsum(probs)],1,'last'); % Select an action
    
    
    %% PERFORM ACTION
    % Move to state stp1 and collect reward
    [rew,stp1,stp1i] = stNac2stp1Nr(st,at,params); % State and action to state plus one and reward
    
    
    %% UPDATE TRANSITION MATRIX AND EXPERIENCE LIST
    targVec = zeros(1,nStates); targVec(stp1i) = 1; % Update transition matrix
    T(sti,:) = T(sti,:) + params.TLearnRate*(targVec-T(sti,:)); % Shift corresponding row of T towards targVec
    expList(size(expList,1)+1,:) = [sti,at,rew,stp1i]; % Add transition to expList
    expLast_stp1(sti,at) = stp1i; % stp1 from last experience of this state/action
    expLast_rew(sti,at) = rew; % rew from last experience of this state/action
    simData.expList(tsi,:) = [sti,at,rew,stp1i];
    
    
    %% UPDATE Q-VALUES (LEARNING)
    if strcmp(params.onVSoffPolicy,'on-policy')
        stp1Value = sum(Q(stp1i,:) .* pAct(Q(stp1i,:),params.actPolicy,params)); % Expected SARSA (learns Qpi)
    else
        stp1Value = max(Q(stp1i,:)); % Q-learning (learns Q*)
    end
    delta = ( rew + params.gamma*stp1Value - Q(sti,at) ); % Prediction error (Q-learning)
    eTr(sti,at) = 1; eTr(sti,~ismember(1:nActions,at)) = 0; % Update eligibility trace using replacing traces, http://www.incompleteideas.net/book/ebook/node80.html)
    Q = Q + (params.alpha * eTr) * delta; % TD-learning
    eTr = eTr * params.lambda * params.gamma; % Decay eligibility trace
    
    
    %% PLANNING PREP
    p=1; % Initialize planning step counter
    if params.planOnlyAtGorS % Only do replay if either current or last trial was a goal state
        curr_step_is_goal = ismember(expList(end,4),sub2ind(size(params.maze),params.s_end(:,1),params.s_end(:,2))); % Current step is a move towards a goal
        last_step_was_goal = any(ismember([expList(end-1,4);expList(end-1,1)],sub2ind(size(params.maze),params.s_end(:,1),params.s_end(:,2))));  % previous state (notice that we include both start and end state of the previous timestep, because the last row of expList might be a transition from goal to start)
        if ~or( curr_step_is_goal , last_step_was_goal )
            p=Inf; % Otherwise, no planning
        end
    end
    if and(rew==0,numEpisodes==0)
        p=Inf; % Skip planning before the first reward is encountered
    end
    
    % Pre-allocate variables to store planning info
    planning_backups = nan(0,5); % List of planning backups (to be used for creating a plot with the full planning trajectory/trace)
    backupsGain = cell(0,1); % List of GAIN for backups executed
    backupsNeed = cell(0,1); % List of NEED for backups executed
    backupsEVM = nan(0,1); % List of EVM for backups executed
    
    
    %% PLANNING STEPS
    while p <= params.nPlan
        
        % Create a list of 1-step backups based on 1-step models
        planExp = [reshape(repmat((1:nStates)',1,nActions),[],1) , reshape(repmat(1:nActions,nStates,1),[],1) , expLast_rew(:) , expLast_stp1(:)];
        planExp = planExp(~any(isnan(planExp),2),:); % Remove NaNs -- e.g. actions starting from invalid states
        if params.remove_samestate % Remove actions that lead to same state (optional) -- e.g. hitting the wall
            planExp = planExp(~(planExp(:,1)==planExp(:,4)),:);
        end
        planExp = num2cell(planExp,2); % use planExp to hold all steps of any n-step trajectory
        
        % Expand previous backup with one extra action
        if and(params.expandFurther,size(planning_backups,1)>0)
            seqStart = find(planning_backups(:,5)==1,1,'last'); % Find the last entry in planning_backups with that started an n-step backup
            seqSoFar = planning_backups(seqStart:end,1:4);
            sn=seqSoFar(end,4); % Final state reached in the last planning step
            if strcmp(params.onVSoffPolicy,'on-policy')
                probs = pAct(Q(sn,:),params.planPolicy,params); % Appended experience is sampled on-policy
            else
                probs = zeros(size(Q(sn,:)));
                probs(Q(sn,:)==max(Q(sn,:))) = 1/sum(Q(sn,:)==max(Q(sn,:))); % Appended experience is sampled greedily
            end
            an = find(rand > [0 cumsum(probs)],1,'last'); % Select action to append using the same action selection policy used in real experience
            snp1 = expLast_stp1(sn,an); % Resulting state from taking action an in state sn
            rn = expLast_rew(sn,an); % Reward received on this step only

            next_step_is_nan = or(isnan(expLast_stp1(sn,an)),isnan(expLast_rew(sn,an))); % Check whether the retrieved rew and stp1 are NaN
            next_step_is_repeated = ismember(snp1,[seqSoFar(:,1);seqSoFar(:,4)]); % Check whether a loop is formed
            % PS: Notice that we can't enforce that planning is done only when the next state is not repeated or don't form aloop. The reason is that the next step needs to be derived 'on-policy', otherwise the Q-values may not converge.
            if and(~next_step_is_nan  ,  or(params.allowLoops,~next_step_is_repeated)) % If loops are not allowed and next state is repeated, don't expand this backup
                seqUpdated = [seqSoFar;[sn an rn snp1]]; % Add one row to seqUpdated (i.e., append one transition). Notice that seqUpdated has many rows, one for each appended step
                planExp{numel(planExp)+1} = seqUpdated; % Notice that rew=rn here (only considers reward from this step)
            end
        end
        
        % Gain term
        [gain,saGain] = gainTerm(Q,planExp,params);
        if params.setAllGainToOne
            gain = num2cell(ones(numel(planExp),1));
        end
        
        % Need term
        [need,SR_or_SD] = needTerm(sti,T,planExp,params);
        if params.setAllNeedToOne
            need = num2cell(ones(numel(planExp),1));
        end
        if params.setAllNeedToZero
            for e=1:numel(planExp)
                need{e}(:) = 0;
                need{e}(planExp{e}(:,1)==sti) = 1; % Set need to 1 only if updated state is sti
            end
            SR_or_SD = zeros(size(SR_or_SD)); SR_or_SD(sti) = 1;
        end
        
        % Expected value of memories
        EVM = nan(size(planExp));
        for i=1:numel(planExp)
            EVM(i) = sum(need{i}(end) .* max(gain{i},params.baselineGain)); % Use the need from the last (appended) state
        end
        
        if params.PLOT_EVM && (double(rew~=0)+numEpisodes>=params.PLOT_wait)
            figure(1); clf;
            if params.PLOT_Qvals
                plotMazeWithArrows(st,max(Q,[],2),min(Q(:),max(params.rewMag(:))),params,'b2r(-1,1)','b2r(-1,1)')
            else
                plotMazeWithArrows(st,max(Q,[],2),nan(size(Q(:))),params,'b2r(-1,1)','b2r(-1,1)')
            end
            set(gcf,'Position',[1,535,560,420])
            figure(2); clf;
            plotMazeWithArrows(st,SR_or_SD',5*saGain(:),params,'YlGnBu','gray'); % Notice the scaling factor, causing Gain=0.25 to be colored as yellow
            set(gcf,'Position',[562,535,560,420])
        end
        
        
        %% PERFORM THE UPDATE
        opportCost = nanmean(expList(:,3)); % Average expected reward from a random act
        EVMthresh = min(opportCost,params.EVMthresh); % if EVMthresh==Inf, threshold is opportCost
        
        if max(EVM) > EVMthresh
            % Identify state-action pairs with highest priority
            maxEVM_idx = find(EVM(:)==max(EVM));
            
            if numel(maxEVM_idx)>1 % If there are multiple items with equal gain
                switch params.tieBreak
                    case 'max'
                        nSteps = cellfun('size',planExp,1); % number of total steps on this trajectory
                        maxEVM_idx = maxEVM_idx(nSteps(maxEVM_idx) == max(nSteps(maxEVM_idx))); % Select the one corresponding to a longer trajectory
                    case 'min'
                        nSteps = cellfun('size',planExp,1); % number of total steps on this trajectory
                        maxEVM_idx = maxEVM_idx(nSteps(maxEVM_idx) == min(nSteps(maxEVM_idx))); % Select the one corresponding to a longer trajectory
                end
                if numel(maxEVM_idx)>1 % If there are still multiple items with equal gain (and equal length)
                    maxEVM_idx = maxEVM_idx(randi(numel(maxEVM_idx))); % ... select one at random
                end
            end
            
            % Plot planning steps
            if params.PLOT_PLANS && (double(rew~=0)+numEpisodes>=params.PLOT_wait)
                figure(1); clf;
                highlight = zeros(size(Q(:)));
                highlight(sub2ind(size(Q),planExp{maxEVM_idx}(:,1),planExp{maxEVM_idx}(:,2))) = 1; % Highlight extended trace
                if params.PLOT_Qvals
                    plotMazeWithArrows(st,max(Q,[],2),min(Q(:),max(params.rewMag(:))),params,'b2r(-1,1)','b2r(-1,1)',highlight,rew)
                else
                    plotMazeWithArrows(st,max(Q,[],2),nan(size(Q(:))),params,'b2r(-1,1)','b2r(-1,1)',highlight,rew)
                end
                set(gcf,'Position',[1,535,560,420])
            end
            
            for n=1:size(planExp{maxEVM_idx},1)
                % Retrieve information from this experience
                s_plan = planExp{maxEVM_idx}(n,1);
                a_plan = planExp{maxEVM_idx}(n,2);
                stp1_plan = planExp{maxEVM_idx}(end,4); % Notice the use of 'end' instead of 'n', meaning that stp1_plan is the final state of the trajectory
                rewToEnd = planExp{maxEVM_idx}(n:end,3); % Individual rewards from this step to end of trajectory
                r_plan = (params.gamma.^(0:(length(rewToEnd)-1))) * rewToEnd; % Discounted cumulative reward from this step to end of trajectory
                n_plan = length(rewToEnd);
                if strcmp(params.onVSoffPolicy,'on-policy')
                    stp1Value = sum(Q(stp1_plan,:) .* pAct(Q(stp1_plan,:),params.planPolicy,params)); % Learns Qpi -> Expected SARSA(1), or, equivalently, n-step Expected SARSA
                else
                    stp1Value = max(Q(stp1_plan,:)); % Learns Q* (can be thought of as 'on-policy' if the target policy is the optimal policy, since trajectory is sampled greedily)
                end
                Qtarget = r_plan + (params.gamma^n_plan)*stp1Value;
                if params.copyQinPlanBkps
                    Q(s_plan,a_plan) = Qtarget;
                else
                    Q(s_plan,a_plan) = Q(s_plan,a_plan) + params.alpha * (Qtarget-Q(s_plan,a_plan));
                end
            end
            
             % List of planning backups (to be used in creating a plot with the full planning trajectory/trace)
            backupsGain = [backupsGain; gain{maxEVM_idx}]; % List of GAIN for backups executed
            backupsNeed = [backupsNeed; need{maxEVM_idx}]; % List of NEED for backups executed
            backupsEVM = [backupsEVM; EVM(maxEVM_idx)]; % List of EVM for backups executed
            planning_backups = [planning_backups; [planExp{maxEVM_idx}(end,1:4) size(planExp{maxEVM_idx},1)]]; % Notice that the first column of planning_backups corresponds to the start state of the final transition on a multistep sequence
            p = p + 1; % Increment planning counter
        else
            break
        end
    end
    
    
    %% MOVE AGENT TO NEXT STATE
    st = stp1; sti = stp1i;
    
    
    %% COMPLETE STEP
    ts = ts+1; % Timesteps to solution (reset to zero at the end of the episode)
    if ismember(st,params.s_end,'rows') % Agent is at a terminal state
        if params.s_start_rand
            % Pick next start state at random
            validStates = find(params.maze==0); % can start at any non-wall state
            validStates = validStates(~ismember(validStates,sub2ind(size(params.maze),params.s_end(:,1),params.s_end(:,2)))); %... but remove the goal states from the list
            stp1i = validStates(randi(numel(validStates)));
            [stp1(1),stp1(2)] = ind2sub( [sideII,sideJJ], stp1i );
        else
            % Determine which of the possible start states to use
            goalnum = find(ismember(params.s_end,st,'rows'));
            startnum = mod(goalnum,size(params.s_start,1)) + 1;
            stp1 = params.s_start(startnum,:);
            stp1i = sub2ind( [sideII,sideJJ], stp1(1), stp1(2) );
        end
        % Update transition matrix and list of experiences
        if params.Tgoal2start
            targVec = zeros(1,nStates); targVec(stp1i) = 1;
            T(sti,:) = T(sti,:) + params.TLearnRate*(targVec-T(sti,:)); % Shift corresponding row of T towards targVec
            expList(size(expList,1)+1,:) = [sti,nan,nan,stp1i]; % Update list of experiences
        end
        st = stp1; sti = stp1i; % Move the agent to the start location
        ets = [ets; ts]; ts = 0; %#ok<AGROW> % record that we took "ts" timesteps to get to the solution (end state)
        eTr = zeros(size(eTr)); % Reset eligibility matrix
        numEpisodes = numEpisodes+1; % Record that we got to the end
    end
    
    
    %% SAVE SIMULATION DATA
    simData.numEpisodes(tsi) = numEpisodes;
    simData.stepsPerEpisode = ets;
    assert(size(simData.expList,1)==tsi,'expList has incorrect size')
    if size(planning_backups,1)>0 % If there was planning in this timestep
        simData.replay.state{tsi} = planning_backups(:,1)'; % In a multi-step sequence, simData.replay.state has 1->2 in one row, 2->3 in another row, etc
        simData.replay.action{tsi} = planning_backups(:,2)';
        simData.replay.gain{tsi} = backupsGain;
        simData.replay.need{tsi} = backupsNeed;
        simData.replay.EVM{tsi} = backupsEVM;
    end
    
    % If max number of episodes is reached, trim down simData.replay
    if numEpisodes==params.MAX_N_EPISODES
        simData.numEpisodes = simData.numEpisodes(1:tsi);
        simData.replay.state = simData.replay.state(1:tsi);
        simData.replay.action = simData.replay.action(1:tsi);
        simData.replay.gain = simData.replay.gain(1:tsi);
        simData.replay.need = simData.replay.need(1:tsi);
        simData.replay.EVM = simData.replay.EVM(1:tsi);
        fprintf(repmat('\b', 1, lastsize));
        fprintf('%d steps; %d episodes\n', tsi, numEpisodes);
        break
    end
    
end
