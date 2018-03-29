%% STATE-SPACE PARAMETERS
params.maze             = zeros(1,10); % zeros correspond to 'visitable' states
params.s_end            = [1,size(params.maze,2)]; % goal state (in matrix notation)
params.s_start          = [1,3]; % beginning state (in matrix notation)
params.s_start_rand     = false; % Start at random locations after reaching goal
params.rewMag           = 1; % reward magnitude (can be a vector)
params.rewSTD           = 0.5; % reward standard deviation (can be a vector)
params.probNoReward     = 0; % probability of receiving no reward

params.gamma            = 0.95; % Discount factor
params.embbedPolicy     = true; % Include policy 
params.policy_strength  = 0.8; % Probability that animal will follow optimal policy
params.zeroDiag         = false; % Set diagonal to zero when calculating the SR

saveBool = true; filename = mfilename;


%% INITIALIZE VARIABLES

% get the initial maze dimensions:
[sideII,sideJJ] = size(params.maze);

% maximal number of states:
nStates = sideII*sideJJ;

% on each grid we can choose from among at most this many actions:
nActions = 4; % 1=UP; 2=DOWN; 3=RIGHT; 4=LEFT

% Transition count
T = zeros(nStates,nStates);


%% PRE-EXPLORE MAZE (have the animal freely explore the maze without rewards to learn action consequences)
for sti=1:nStates
    for at=1:nActions
        % Sample action consequences (minus reward)
        [st(1),st(2)] = ind2sub( [sideII,sideJJ], sti );
        if (params.maze(st(1),st(2)) == 0) && ~ismember([st(1) st(2)],params.s_end,'rows') % Don't explore walls or goal state
            [~,~,stp1i] = stNac2stp1Nr(st,at,params); % state and action to state plus one and reward
            T(sti,stp1i) = T(sti,stp1i) + 1; % Update transition matrix
        end
    end
end
% Normalize so that rows sum to one
T = T./repmat(nansum(T,2),1,size(T,1));
T(isnan(T)) = 0;
% Add transitions from goal states to start states
T(nStates,1) = 1;


%% MOVE AGENT TO REWARD
if params.embbedPolicy
    allStates = 1:nStates;
    trajectory = [...
        1,2;...
        2,3;...
        3,4;...
        4,5;...
        5,6;...
        6,7;...
        7,8;...
        8,9;...
        9,10;...
        ];
    
    for i=1:size(trajectory,1)
        T(trajectory(i,1),trajectory(i,2)) = params.policy_strength;
        allOthers = allStates(allStates~=trajectory(i,2));
        scaleFactor = (1-params.policy_strength) ./ sum(T(trajectory(i,1),allOthers));
        T(trajectory(i,1),allOthers) = T(trajectory(i,1),allOthers) * scaleFactor;
    end
end


%% COMPUTE NEED TERM

% Set diagonal to zero
if params.zeroDiag
    T(logical(eye(size(T)))) = 0;
end

% Normalize rows to sum to 1
rowTotal = nansum(T,2);
rowTotal(rowTotal~=0) = 1./rowTotal(rowTotal~=0);
T = T.* repmat(rowTotal,1,numel(rowTotal));

% Calculate Successor Representation
SR = inv(   eye(size(T)) - params.gamma*T   );


%% VISUALIZE NEED TERM
imagesc(reshape(SR(sub2ind(size(params.maze),params.s_start(1),params.s_start(2)),:),sideII,sideJJ))
colormap('gray')
axis square
axis equal


%% EXPORT FIGURE
if saveBool
    % Set clipping off
    set(gca, 'Clipping', 'off');
    set(gcf, 'Clipping', 'off');
    
    set(gcf, 'renderer', 'painters');
    %export_fig(filename, '-pdf', '-eps', '-png', '-q101', '-nocrop', '-painters');
    print(filename,'-dpdf','-fillpage')
end
