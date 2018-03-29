function [need,SR_or_SD] = needTerm(sti,T,nStepExps,params)

need = cell(1,numel(nStepExps));
switch params.onlineVSoffline
    case 'online' % Calculate the successor representation
        % Calculate Successor Representation
        SR = inv(   eye(size(T)) - params.gamma*T   );
        SRi = SR(sti,:); % Calculate the Successor Representation for the current state
        SR_or_SD = SRi;
    case 'offline'
        % Calculate eigenvectors and eigenvalues
        [V,D,W] = eig(T);
        if abs(D(1,1))-1>1e-10; error('Precision error'); end
        SD = abs(W(:,1)'); % Stationary distribution of the MDP        
        SR_or_SD = SD;
end

% Calculate need-term for each experience in nStepExps
for i=1:numel(nStepExps)
    thisExp = nStepExps{i};
    need{i} = nan(1,size(thisExp,1));
    for j=1:size(thisExp,1)
        need{i}(j) = SR_or_SD(thisExp(j,1));
    end
end
