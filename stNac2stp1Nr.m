function [rew,stp1,stp1i] = stNac2stp1Nr(st,at,params)
% STNAC2STP1 - state and action to next state and reward 

% Dimensions of the maze
[sideII,sideJJ] = size(params.maze);

% convert to row/column notation: 
ii = st(1); jj = st(2); 

%% MOVE THE AGENT TO THE NEXT POSITION

% incorporate any actions and fix our position if we end up outside the grid
switch at
 case 1
	% action = UP 
	stp1 = [ii-1,jj];
 case 2
	% action = DOWN
	stp1 = [ii+1,jj];
 case 3
	% action = RIGHT
	stp1 = [ii,jj+1];
 case 4
	% action = LEFT 
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
    if size(params.rewMag,1) == size(params.s_end,1) % if there's a different reward for each goal state
        thisRew = params.rewMag(ismember(params.s_end,stp1,'rows'),:);
    else
        thisRew = params.rewMag(1,:);
    end
    if size(params.rewSTD,1) == size(params.s_end,1) % if there's a different std for each goal state
        thisSTD = params.rewSTD(ismember(params.s_end,stp1,'rows'),:);
    else
        thisSTD = params.rewSTD(1,:);
    end
    % Draw a sample from the reward magnutude list
    if size(thisRew,2) == size(params.rewProb,2)
        thisProb = params.rewProb/sum(params.rewProb);
        rewIdx = find(rand > [0 cumsum(thisProb)],1,'last'); % Select reward
        thisRew = thisRew(rewIdx);
    else
        rewIdx = 1; % If the probabilities were not correctly specified, use the first column only
        thisRew = thisRew(1);
    end
    % Draw a sample from the reward std list
    if size(thisSTD,2) == size(params.rewProb,2)
        thisSTD = thisSTD(rewIdx);
    else
        thisSTD = thisSTD(1);
    end
    
    % Compute the final reward
    rew = thisRew + randn*thisSTD;
    if params.rewOnlyPositive
        rew = max(0,rew); % Make sure reward+noise is positive (negative rewards are weird)
    end
else
	rew = 0;
end
