function probs = pAct(Qmean,actPolicy,params)
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
        for s=1:size(Qmean,1) % Notice that this loops through states, but often times this function receives only one state
            Qbest = find(Qmean(s,:)==max(Qmean(s,:))); % Find indices of actions with maximum value
            if all(Qmean(s,:)==max(Qmean(s,:)))
                probs(s,Qbest) = 1/numel(Qbest);
            else
                probs(s,Qbest) = (1-params.epsilon)/numel(Qbest);
                probs(isnan(probs)) = 0;
                probs = probs + params.epsilon/numel(Qmean(s,:)); % With probability epsilon, pick a random action
            end
        end
    case 'softmax'
        for s=1:size(Qmean,1)
            probs(s,:) = exp(params.softmaxInvT * Qmean(s,:)) ./ sum(exp(params.softmaxInvT * Qmean(s,:)));
        end
    otherwise
        error('Unrecognized strategy');
end

assert(all(abs(sum(probs,2)-1)<1e-3), 'Probabilities do not sum to one');
