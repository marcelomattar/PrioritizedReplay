%% STATE-SPACE PARAMETERS
addpath('../../../');
clear;
setParams;
params.maze             = zeros(1,10); % zeros correspond to 'visitable' states
params.s_start          = [1,1]; % beginning state (in matrix notation)
params.s_start_rand     = false; % Start at random locations after reaching goal
params.s_end            = [1,size(params.maze,2)]; % goal state (in matrix notation)
params.rewMag           = 0; % reward magnitude (rows: locations; columns: values)
params.rewSTD           = 0; % reward Gaussian noise (rows: locations; columns: values)
params.rewProb          = 1; % probability of receiving each reward (columns: values)

saveStr = input('Do you want to produce figures (y/n)? ','s');
if strcmp(saveStr,'y')
    saveBool = true;
else
    saveBool = false;
end


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


%% PLANNING PREP
p=1; % Initialize planning step counter
planning_backups_pre = nan(0,5); % List of planning backups (to be used for creating a plot with the full planning trajectory/trace)


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
    if and(params.expandFurther,size(planning_backups_pre,1)>0)
        seqStart = find(planning_backups_pre(:,5)==1,1,'last'); % Find the last entry in planning_backups with that started an n-step backup
        seqSoFar = planning_backups_pre(seqStart:end,1:4);
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
    
    % EVM
    [gain,saGain] = gainTerm(Q,planExp,params);
    [need,SR_or_SD] = needTerm(sti,T,planExp,params);
    EVM = nan(size(planExp));
    for i=1:numel(planExp)
        EVM(i) = sum(need{i}(end) .* max(gain{i},params.baselineGain)); % Use the need from the last (appended) state
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
        planning_backups_pre = [planning_backups_pre; [planExp{maxEVM_idx}(end,1:4) size(planExp{maxEVM_idx},1)]];
        p = p + 1; % Increment planning counter
    else
        break
    end
end


%% PLANNING PREP
p=1; % Initialize planning step counter
planning_backups_post = nan(0,5); % List of planning backups (to be used for creating a plot with the full planning trajectory/trace)


%% IMPLANT MEMORY
expLast_rew(size(params.maze,2)-1,3) = -1; % this matrix contains the reward obtained from action (i,j), where (i,j) is state/action
expLast_stp1(size(params.maze,2)-1,3) = size(params.maze,2); % this matrix contains the next-state from taking action (i,j), where (i,j) is state/action


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
    if and(params.expandFurther,size(planning_backups_post,1)>0)
        seqStart = find(planning_backups_post(:,5)==1,1,'last'); % Find the last entry in planning_backups with that started an n-step backup
        seqSoFar = planning_backups_post(seqStart:end,1:4);
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
    
    % EVM
    [gain,saGain] = gainTerm(Q,planExp,params);
    [need,SR_or_SD] = needTerm(sti,T,planExp,params);
    EVM = nan(size(planExp));
    for i=1:numel(planExp)
        EVM(i) = sum(need{i}(end) .* max(gain{i},params.baselineGain)); % Use the need from the last (appended) state
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
        planning_backups_post = [planning_backups_post; [planExp{maxEVM_idx}(end,1:4) size(planExp{maxEVM_idx},1)]];
        p = p + 1; % Increment planning counter
    else
        break
    end
end


%% PLOT RESULTS
preShock_backups = planning_backups_pre(:,1); % list of states backed up
postShock_backups = planning_backups_post(:,1); % list of states backed up
shockZone = 9; % State corresponding to the action of entering the shock zone
actProb_pre = any(preShock_backups==shockZone); % Probability that the SZ state is replayed *before* the shock
actProb_post = any(postShock_backups==shockZone);% Probability that the SZ state is replayed *after* the shock

figure(1); clf;
plot(1:2,[actProb_pre actProb_post],'o','Color',[0 0 0],'MarkerFaceColor',[0 0 0]);
xlim([0.5 2.5]); ylim([-0.1 1.1]); grid on;
set(gca,'XTick',1:2,'XTickLabel',{'Pre','Post'}); 
set(gcf,'Position',[664   437   121   277])
ylabel('Activation Probability');
title(sprintf('Replication of\nWu et al (2017)'));


%% EXPORT FIGURE
if saveBool
    % Set clipping off
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    
    set(gcf, 'renderer', 'painters');
    export_fig(['../Parts/' mfilename], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
end

