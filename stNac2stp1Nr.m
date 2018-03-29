function [rew,stp1,stp1i] = stNac2stp1Nr(st,at,params)
% STNAC2STP1 - state and action to state plus one and reward 

% Dimensions of the maze
[sideII,sideJJ] = size(params.maze);

% convert to row/column notation: 
ii = st(1); jj = st(2); 

%% MOVE THE AGENT TO THE NEXT POSITION

% incorporate any actions and fix our position if we end up outside the grid:
switch at
 case 1
	%
	% action = UP 
	%
	stp1 = [ii-1,jj];
 case 2
	%
	% action = DOWN
	%
	stp1 = [ii+1,jj];
 case 3
	%
	% action = RIGHT
	%
	stp1 = [ii,jj+1];
 case 4
	%
	% action = LEFT 
	%
	stp1 = [ii,jj-1];
 otherwise
	error(sprintf('unknown value for of action = %d',at));  %#ok<SPERR>
end

% adjust our position if we have fallen outside of the grid:
if( stp1(1)<1 ); stp1(1)=1; end
if( stp1(1)>sideII ); stp1(1)=sideII; end
if( stp1(2)<1 ); stp1(2)=1; end
if( stp1(2)>sideJJ ); stp1(2)=sideJJ; end

% if this trasition leads to a wall, no transition takes place
if( params.maze(stp1(1),stp1(2))==1 ) 
	stp1 = st; 
end

% convert to an index: 
stp1i = sub2ind( [sideII,sideJJ], stp1(1), stp1(2) ); 

%% COLLECT REWARD

if ismember(stp1,params.s_end,'rows')
    if numel(params.rewMag) == size(params.s_end,2) % if there's a different reward for each goal state
        % Determine which reward has been encountered in stp1
        rew = params.rewMag(ismember(params.s_end,stp1,'rows'));
        % Add noise
        rew = rew + randn*params.rewSTD(ismember(params.s_end,stp1,'rows'));
    else % if there's only one reward for all goal states
        rew = params.rewMag; 
        % Add noise
        rew = rew + randn * params.rewSTD;
    end
    rew = max(0,rew); % Make sure reward is positive (negative rewards are weird)
    % Set it to zero with probability params.probNoReward
    if rand<params.probNoReward
        rew = 0; % May change this value
    end
else
	rew = 0;
end
