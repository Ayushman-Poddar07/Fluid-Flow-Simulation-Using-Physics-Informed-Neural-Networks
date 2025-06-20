% Fluid Flow Simulation Using Physics-Informed Neural Networks (PINNs)
% Navier-Stokes Solver in Complex and Simple Geometries

clc; clear; close all;

%% Data Collection and Preprocessing (GUI Inputs)

prompt = {'Grid size in x:', 'Grid size in y:', 'Time steps:', ...
          'Geometry (pipe/channel/nozzle/bend/t-junction/obstacle/cylinder):', ...
          'Inlet velocity type (constant/sine):'};
titleText = 'Simulation Setup';
defaults = {'50', '50', '10', 'pipe', 'sine'};
userInput = inputdlg(prompt, titleText, 1, defaults);

nx = str2double(userInput{1});
ny = str2double(userInput{2});
nt = str2double(userInput{3});
geotype = lower(userInput{4});
inletType = lower(userInput{5});

x = linspace(0, 1, nx);
y = linspace(-1, 1, ny);
t = linspace(0, 1, nt);
[X, Y, T] = ndgrid(x, y, t);
X_data = [X(:), Y(:), T(:)]';  % 3 x N

% Geometry masks
switch geotype
    case 'nozzle'
        mask = ~(X_data(1,:) > 0.5 & abs(X_data(2,:)) > 0.5);
    case 'bend'
        mask = ~(X_data(1,:) > 0.5 & X_data(2,:) > 0.2);
    case 't-junction'
        mask = (X_data(2,:) > -0.3 & X_data(2,:) < 0.3) | ...
               (X_data(1,:) < 0.5 & X_data(2,:) > 0);
    case 'obstacle'
        mask = ~(X_data(1,:) > 0.4 & X_data(1,:) < 0.6 & ...
                 X_data(2,:) > -0.2 & X_data(2,:) < 0.2);
    case 'cylinder'
        mask = sqrt((X_data(1,:) - 0.5).^2 + X_data(2,:).^2) > 0.2;
    otherwise
        mask = true(1, size(X_data, 2));
end
X_data = X_data(:, mask);

rho = 1.0;
mu = 0.01;
nu = mu / rho;

%% Model Development (Neural Network)
layers = [
    featureInputLayer(3,"Normalization","none")
    fullyConnectedLayer(50)
    tanhLayer
    fullyConnectedLayer(50)
    tanhLayer
    fullyConnectedLayer(3)
];
dlnet = dlnetwork(layers);

%% Loss Function Definition with Boundary Conditions and Loss Logging
function [loss, gradients, lossTerms] = modelLoss(dlnet, X_dl, nu, rho, inletType)
    UVP = forward(dlnet, X_dl);
    u = UVP(1,:); v = UVP(2,:); p = UVP(3,:);
    x = X_dl(1,:); y = X_dl(2,:); t = X_dl(3,:);

    du_x = dlgradient(sum(u), x, 'EnableHigherDerivatives', true);
    du_y = dlgradient(sum(u), y, 'EnableHigherDerivatives', true);
    du_t = dlgradient(sum(u), t, 'EnableHigherDerivatives', true);
    du_xx = dlgradient(sum(du_x), x, 'EnableHigherDerivatives', true);
    du_yy = dlgradient(sum(du_y), y, 'EnableHigherDerivatives', true);

    dv_x = dlgradient(sum(v), x, 'EnableHigherDerivatives', true);
    dv_y = dlgradient(sum(v), y, 'EnableHigherDerivatives', true);
    dv_t = dlgradient(sum(v), t, 'EnableHigherDerivatives', true);
    dv_xx = dlgradient(sum(dv_x), x, 'EnableHigherDerivatives', true);
    dv_yy = dlgradient(sum(dv_y), y, 'EnableHigherDerivatives', true);

    dp_x = dlgradient(sum(p), x);
    dp_y = dlgradient(sum(p), y);

    f_u = du_t + u.*du_x + v.*du_y - dp_x./rho - nu*(du_xx + du_yy);
    f_v = dv_t + u.*dv_x + v.*dv_y - dp_y./rho - nu*(dv_xx + dv_yy);
    f_continuity = du_x + dv_y;

    inlet_mask = abs(x) < 1e-6;
    inlet_u = u(inlet_mask);
    inlet_v = v(inlet_mask);
    if strcmp(inletType, 'sine')
        inlet_profile = (1 - y(inlet_mask).^2) .* (1 + 0.5 * sin(2 * pi * t(inlet_mask)));
    else
        inlet_profile = (1 - y(inlet_mask).^2);
    end

    wall_mask = abs(abs(y) - 1) < 1e-3;
    wall_v = v(wall_mask);
    loss_wall = mean(wall_v.^2);

    outlet_mask = abs(x - 1) < 1e-3;
    outlet_u = u(outlet_mask);
    outlet_v = v(outlet_mask);
    du_x_out = dlgradient(sum(outlet_u), x(outlet_mask), 'EnableHigherDerivatives', true);
    dv_x_out = dlgradient(sum(outlet_v), x(outlet_mask), 'EnableHigherDerivatives', true);
    loss_outlet = mean(du_x_out.^2 + dv_x_out.^2);

    loss_bc = mean((inlet_u - inlet_profile).^2 + (inlet_v).^2) + loss_wall + loss_outlet;
    loss_pde = mean(f_u.^2 + f_v.^2 + f_continuity.^2);

    init_mask = abs(t) < 1e-6;
    u_init = u(init_mask);
    v_init = v(init_mask);
    loss_init = mean(u_init.^2 + v_init.^2);

    w_pde = 1.0; w_bc = 10.0; w_init = 1.0;
    loss = w_pde * loss_pde + w_bc * loss_bc + w_init * loss_init;
    lossTerms = [loss_pde, loss_bc, loss_wall, loss_outlet, loss_init];
    gradients = dlgradient(loss + 0*sum(u), dlnet.Learnables);
end

%% Training
X_dl = dlarray(X_data, 'CB');
numEpochs = 1000;
lr = 1e-3;
trailingAvg = [];
trailingAvgSq = [];
lossLog = zeros(numEpochs, 6);
for epoch = 1:numEpochs
    [loss, grads, lossTerms] = dlfeval(@modelLoss, dlnet, X_dl, nu, rho, inletType);
    [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, grads, ...
        trailingAvg, trailingAvgSq, epoch, lr);
    lossLog(epoch, :) = gather([extractdata(loss), extractdata(lossTerms)]);
    if mod(epoch, 100) == 0
        disp("Epoch " + epoch + " | Total Loss = " + lossLog(epoch,1));
    end
end

%% Animation of Velocity Field over Time
answer_ani = inputdlg('Enter time duration for animation (e.g., 1 or 2):', 'Animation Duration', 1, {'1'});
t_end = str2double(answer_ani{1});
t_ani = linspace(0, t_end, 20);
[Xg, Yg] = meshgrid(x, y);
for k = 1:length(t_ani)
    XYT_eval = [Xg(:), Yg(:), t_ani(k)*ones(numel(Xg),1)]';
    XYT_dl = dlarray(XYT_eval, 'CB');
    UVP_out = extractdata(forward(dlnet, XYT_dl));
    U = reshape(UVP_out(1,:), size(Xg));
    V = reshape(UVP_out(2,:), size(Yg));
    Vmag = sqrt(U.^2 + V.^2);
    contourf(Xg, Yg, Vmag, 20, 'LineColor', 'none');
    colorbar;
    hold on;
    quiver(Xg, Yg, U, V, 'k');
    title(sprintf('Velocity Field at t = %.2f', t_ani(k)));
    xlabel('x'); ylabel('y'); axis equal tight;
    drawnow;
    hold off;
end

%% Final Time Plots
answer = inputdlg('Enter time value to visualize final fields (e.g., 0.5, 1, 2):', 'Final Time Selection', 1, {'2'});
t_vis = str2double(answer{1});
XYT_eval = [Xg(:), Yg(:), t_vis*ones(numel(Xg),1)]';
XYT_dl = dlarray(XYT_eval, 'CB');
UVP_out = extractdata(forward(dlnet, XYT_dl));

% Safety check to avoid <missing> errors
if size(UVP_out, 1) < 3
    error('Network output does not contain pressure values. Check final layer size.');
end

U = reshape(UVP_out(1,:), size(Xg));
V = reshape(UVP_out(2,:), size(Yg));
P = reshape(UVP_out(3,:), size(Yg));
Vmag = sqrt(U.^2 + V.^2);

figure;
contourf(Xg, Yg, Vmag, 20, 'LineColor', 'none'); hold on;
streamslice(Xg, Yg, U, V);
title(['Streamlines and Velocity Magnitude at t=', num2str(t_vis), ' - Geometry: ', geotype]);
xlabel('x'); ylabel('y'); axis equal tight; colorbar;

figure;
quiver(Xg, Yg, U, V);
title(['Velocity Field at t=', num2str(t_vis), ' - Geometry: ', geotype]);
xlabel('x'); ylabel('y'); axis equal tight;

figure;
surf(Xg, Yg, P);
title(['Pressure Field at t=', num2str(t_vis), ' - Geometry: ', geotype]);
xlabel('x'); ylabel('y'); shading interp; view(2); colorbar;

figure;
plot(1:numEpochs, lossLog(:,1), '-b', 'LineWidth', 1.5); hold on;
plot(1:numEpochs, lossLog(:,2), '--r', 'LineWidth', 1.5);
plot(1:numEpochs, lossLog(:,3), '--g', 'LineWidth', 1.5);
plot(1:numEpochs, lossLog(:,4), '--m', 'LineWidth', 1.5);
legend('Total Loss', 'PDE Loss', 'BC Loss', 'Wall Loss', 'Outlet Loss', 'Init Loss');
xlabel('Epoch'); ylabel('Loss'); title('Training Loss over Epochs'); grid on;
