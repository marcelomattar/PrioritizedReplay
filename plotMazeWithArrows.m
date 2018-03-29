function plotMazeWithArrows(st,squareValues,arrowValues,params,colscale1,colscale2,highlight,rew)

clf;

[numRows,numCols] = size(params.maze);
nStates = numel(params.maze);
nActions = numel(arrowValues)/nStates;
arrowValues = max(0,arrowValues);

%% SET PARAMETERS
stateLineWidth = 0.5;
stateLineColor = [0.5 0.5 0.5];
fontSize =  params.startendSymbSize; % Size of GOAL/START symbols
arrowOffset = params.arrowOffset;
arrowSize = params.arrowSize;
agentSize = params.agentSize;
agentColor = params.agentColor;

highlightInnerColor = [0 0 0];
highlightEdgeColor = [0 0 0];
highlightEdgeWidth = 5;

%% PLOT MAZE
Vmaze = reshape(squareValues,size(params.maze));
Vmaze(params.maze==(unique(params.maze)'*(unique(params.maze)~=0))) = -1; % Set nonzero elements (i.e. walls) to -1
noPlot = Vmaze(:)==-1;
%{
if Vmaze(params.s_end(1),params.s_end(2)) ~= 1
    for i=1:size(params.s_end,1)
        Vmaze(params.s_end(i,1),params.s_end(i,2)) = -1;
    end
    noPlot = Vmaze(:)==-1;
    for i=1:size(params.s_end,1)
        Vmaze(params.s_end(i,1),params.s_end(i,2)) = 0;
    end
else
    noPlot = Vmaze(:)==-1;
end
%}
imagesc( Vmaze ); caxis([-1 1]); hold on;
plot( st(2), st(1), 'o', 'MarkerSize', agentSize, 'MarkerFaceColor', agentColor );

% Plot START state symbols
%{
for i=1:size(params.s_start,1)
    Ssym=text(params.s_start(2),params.s_start(1),'S');
    set(Ssym,'FontSize',fontSize,'FontWeight','bold','HorizontalAlignment','center','VerticalAlignment','middle');
end
%}
% Plot GOAL state symbols
for i=1:size(params.s_end,1)
    if numel(params.rewMag) == size(params.s_end,2) % if there's a different reward for each goal state
        Gsym=text(params.s_end(i,2),params.s_end(i,1),sprintf('%.1f',params.rewMag(i)));
        set(Gsym,'FontSize',fontSize,'FontWeight','bold','HorizontalAlignment','center','VerticalAlignment','middle');
    else
        Gsym=text(params.s_end(i,2),params.s_end(i,1),'G');
        set(Gsym,'FontSize',fontSize,'FontWeight','bold','HorizontalAlignment','center','VerticalAlignment','middle');
    end
end


%% PLOT LINES DEFINING STATES
lrow = nan(numRows+1,1);
lrow(1) = line(xlim,[(0.5) (0.5)]);
set(lrow(1),'LineWidth',stateLineWidth,'Color',stateLineColor);
for r=1:(numRows-1)
    lrow(r+1) = line(xlim,[(r+0.5) (r+0.5)]);
    set(lrow(r+1),'LineWidth',stateLineWidth,'Color',stateLineColor);
end
lrow(end) = line(xlim,[(numRows+0.5) (numRows+0.5)]);
set(lrow(end),'LineWidth',stateLineWidth,'Color',stateLineColor);

lcol = nan(numCols+1,1);
lcol(1) = line([(0.5) (0.5)],ylim);
set(lcol(1),'LineWidth',stateLineWidth,'Color',stateLineColor);
for c=1:(numCols-1)
    lcol(c+1) = line([(c+0.5) (c+0.5)],ylim);
    set(lcol(c+1),'LineWidth',stateLineWidth,'Color',stateLineColor);
end
lcol(end) = line([(numCols+0.5) (numCols+0.5)],ylim);
set(lcol(end),'LineWidth',stateLineWidth,'Color',stateLineColor);


if ~all(isnan(arrowValues))
    %% CALCULATE ARROW COLORS
    
    % If using viridis, use the code below
    if strcmp(colscale1,'viridis')
        N = 100000;
        foo = flipud(viridis(N));
        colNum = round(arrowValues*N);
        arrowCols = foo(max(colNum,1),:);
        arrowCols(isnan(arrowValues),:) = nan;
    elseif strfind(colscale1,'b2r')
        foo = eval(colscale1);
        N = size(foo,1);
        colNum = round((arrowValues*0.5 + 0.5)*N); % Scale between 0.5 and 1.0
        arrowCols = foo(min(colNum,size(foo,1)),:);
        arrowCols(isnan(arrowValues),:) = nan;
    elseif strcmp(colscale1,'PuBuGn')
        % Scale and truncate arrowValues to between 0 and 0.5
        arrowValues = arrowValues(:)/0.5;
        arrowValues(arrowValues<0) = 0;
        arrowValues(arrowValues>1) = 1;
        
        arrowCol_min = [255,247,251]/255;
        arrowCol_med = [103,169,207]/255;
        arrowCol_max = [1,70,54]/255;
        arrowValues = arrowValues(:);
        arrowCols = nan(numel(arrowValues),3);
        for i=1:length(arrowValues)
            if ~isnan(arrowValues(i))
                if arrowValues(i)<0.5
                    arrowCols(i,:) = arrowCol_min + (arrowCol_med-arrowCol_min)*(2*arrowValues(i));
                else
                    arrowCols(i,:) = arrowCol_med + (arrowCol_max-arrowCol_med)*(2*(arrowValues(i)-0.5));
                end
            else
                arrowCols(i,:) = [nan nan nan];
            end
        end
    elseif strcmp(colscale1,'BuPu')
        arrowCol_min = [224,236,244]/255;
        arrowCol_med = [140,150,198]/255;
        arrowCol_max = [110,1,107]/255;
        arrowValues = arrowValues(:);
        arrowCols = nan(numel(arrowValues),3);
        for i=1:length(arrowValues)
            if ~isnan(arrowValues(i))
                if arrowValues(i)<0.5
                    arrowCols(i,:) = arrowCol_min + (arrowCol_med-arrowCol_min)*(2*arrowValues(i));
                else
                    arrowCols(i,:) = arrowCol_med + (arrowCol_max-arrowCol_med)*(2*(arrowValues(i)-0.5));
                end
            else
                arrowCols(i,:) = [nan nan nan];
            end
        end
    elseif strcmp(colscale1,'YlGnBu')
        arrowCol_max = [199,233,180]/255;
        arrowCol_med = [29,145,192]/255;
        arrowCol_min = [8,29,88]/255;
        %arrowCol_min = [237,248,177]/255;
        %arrowCol_med = [65,182,196]/255;
        %arrowCol_max = [37,52,148]/255;
        arrowValues = arrowValues(:);
        arrowCols = nan(numel(arrowValues),3);
        for i=1:length(arrowValues)
            if ~isnan(arrowValues(i))
                if arrowValues(i)<0.5
                    arrowCols(i,:) = arrowCol_min + (arrowCol_med-arrowCol_min)*(2*arrowValues(i));
                else
                    arrowCols(i,:) = arrowCol_med + (arrowCol_max-arrowCol_med)*(2*(arrowValues(i)-0.5));
                end
            else
                arrowCols(i,:) = [nan nan nan];
            end
        end
    end
    
    
    %% PLOT ARROWS
    [rowNum,colNum] = ind2sub([numRows numCols],1:nStates);
    [I,J] = ind2sub([numel(squareValues) numel(arrowValues)/numel(squareValues)],1:numel(arrowValues));
    
    % Remove NaN colors
    nanVals = all(reshape(all(isnan(arrowCols),2),nStates,nActions),2);
    noPlot = or(noPlot,nanVals);
    
    % Remove arrows from the walls and goal state(s)
    colNum = colNum(~noPlot);
    rowNum = rowNum(~noPlot);
    I = I(~(repmat(noPlot,nActions,1)));
    J = J(~(repmat(noPlot,nActions,1)));
    arrowCols = arrowCols(~repmat(noPlot,nActions,1),:);
    
    % Plot with scatter function
    scatter(colNum,rowNum-0.5+arrowOffset,arrowSize,arrowCols(J==1,:),'filled','^','MarkerEdgeColor',[0 0 0]);
    scatter(colNum,rowNum+0.5-arrowOffset,arrowSize,arrowCols(J==2,:),'filled','v','MarkerEdgeColor',[0 0 0]);
    scatter(colNum+0.5-arrowOffset,rowNum,arrowSize,arrowCols(J==3,:),'filled','>','MarkerEdgeColor',[0 0 0]);
    scatter(colNum-0.5+arrowOffset,rowNum,arrowSize,arrowCols(J==4,:),'filled','<','MarkerEdgeColor',[0 0 0]);
    
    plot( st(2), st(1), 'o', 'MarkerSize', agentSize, 'MarkerFaceColor', agentColor );
    
    
    %% HIGHLIGHT ARROWS
    if exist('highlight','var')
        if sum(highlight)>0
            highlight = reshape(highlight,nStates,nActions);
            [sthlg,achlg] = ind2sub([nStates,nActions],find(highlight));
            [rowNum,colNum] = ind2sub([numRows,numCols],sthlg);
            if isnan(highlightInnerColor)
                hlgCols = arrowCols(logical(highlight(~repmat(noPlot,nActions,1))),:);
            else
                hlgCols = repmat(highlightInnerColor,numel(achlg),1);
            end
            for i=1:numel(achlg)
                switch achlg(i)
                    case 1
                        scatter(colNum(i),rowNum(i)-0.5+arrowOffset,arrowSize,hlgCols(i,:),'filled','^','MarkerEdgeColor',highlightEdgeColor,'LineWidth',highlightEdgeWidth);
                    case 2
                        scatter(colNum(i),rowNum(i)+0.5-arrowOffset,arrowSize,hlgCols(i,:),'filled','v','MarkerEdgeColor',highlightEdgeColor,'LineWidth',highlightEdgeWidth);
                    case 3
                        scatter(colNum(i)+0.5-arrowOffset,rowNum(i),arrowSize,hlgCols(i,:),'filled','>','MarkerEdgeColor',highlightEdgeColor,'LineWidth',highlightEdgeWidth);
                    case 4
                        scatter(colNum(i)-0.5+arrowOffset,rowNum(i),arrowSize,hlgCols(i,:),'filled','<','MarkerEdgeColor',highlightEdgeColor,'LineWidth',highlightEdgeWidth);
                end
                
            end
        end
    end
    
end
    
    
%% DISPLAY REWARD
if exist('rew','var')
    title(sprintf('%.2f',rew));
end


%% FINAL ADJUSTMENTS
% Adjust colorscale
if strcmp(colscale2,'viridis')
    colormap(viridis);
    h = colorbar; % Adjust colorbar
elseif strfind(colscale2,'b2r')
    thisColorMap = eval(colscale2);
    thisColorMap(1:(median(1:length(b2r(-1,1)))-1),:) = repmat([0,0,0],median(1:length(b2r(-1,1)))-1,1);
    colormap(thisColorMap);
    h = colorbar; % Adjust colorbar
    %set(h, 'ylim', [0 max(params.rewMag)]);
    set(h, 'ylim', [0 1]);
else
    %caxis([0 max(squareValues(:))]);
    caxis([0 1]);
    colormap(eval(colscale2));
    h = colorbar; % Adjust colorbar
end

axis equal
axis off

pause(params.delay);







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