function [] = pred_heatmap(inputs, y_pred, targets, name)
         
match = (y_pred == targets);         
data = [inputs' y_pred' targets' match']; 

color_data = zeros(size(data));
color_data(:, 1:6) = data(:, 1:6);             
color_data(:, 7:8) = 2 * data(:, 7:8);         
color_data(:, 9) = 3 + data(:, 9);             

color_data = color_data';  

set(gcf, 'Position', [150, 300, 1200, 400]);
imagesc(color_data);

cmap = [1 1 1;    % 0 - white (== 0)
        0 0 1;    % 1 - blue (Input == 1)
        1 0 1;    % 2 - magenta (Pred/Target == 1)
        1 0 0;    % 3 - red (Pred != Target)
        0 1 0];   % 4 - green (Pred == Target);   
colormap(cmap);

axis tight;
daspect([1 1 1]);

xticks(1:size(color_data,2));
yticks(1:9);
yticklabels({'I1','I2','I3','I4','I5','I6','Pred','Target','Match'});

xlabelObj = xlabel('Test Instance');
xlabelObj.Position(2) = xlabelObj.Position(2) + 1;
ylabelObj = ylabel('Inputs / Prediction / Target / Match');
ylabelObj.Position(1) = ylabelObj.Position(1) - 1;

ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'off';

cb = colorbar;

cb.Ticks = 0:4;
cb.TickLabels = {' == 0', ...
                 'Input == 1', ...
                 'Pred/Target == 1', ...
                 'Pred != Target', ...
                 'Pred == Target'};
cb.Label.String = 'Color Legend';
cb.Label.FontWeight = 'bold';

hold on;
[num_rows, num_cols] = size(color_data);

for r = 1:num_rows+1
    y = r - 0.5;
    line([0.5, num_cols + 0.5], [y, y], 'Color', [0 0 0]);
end

for c = 1:num_cols+1
    x = c - 0.5;
    line([x, x], [0.5, num_rows + 0.5], 'Color', [0 0 0]);
end

hold off;
titleObj = title(sprintf('Prediction Heatmap for %s model', name),...
    'FontSize', 12);
titleObj.Position(2) = titleObj.Position(2) - 2;

end

