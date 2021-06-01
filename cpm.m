function ydata= cpm(data,no_dims,geodist,niter,compel_force,D)
% data: N x P data matrix
% no_dims: mapping dimension
% (optional)compel_force=1 if user wants to pull clusters a bit apart
%                       =0 if user wants the best preservation of the geometry
% (optional) niter : number of iterations
% (optional) D: the pairwise distance matrix to be used to define probabilities, 
%               if not an input, it will be computed from data
% 
% (C) Rongrong Wang, Michigan State University

% Compute the pairwise Euclidean(geodesic) distances
 if ~exist('D')
  if geodist ==0 
     D = compute_dist(data,0); 
  else
      disp('computing the geodesic distance')
      D = compute_dist(data,1,5); 
  end
 end
 if ~exist('compel_force')
     compel_force = 0;
 end
  % compute the Capacity adjusted distance
 D1 = cdist(D,no_dims,compel_force);
 n = size(D1,1);
  % change the distances to probabilities
     vP_D = 1./(D1.^2);
     P_D  = reshape(vP_D,size(vP_D));
     P  = P_D./sum(P_D(:));
     P(1:n+1:end) = 0;
     P = max(P ./ sum(P(:)), realmin); 
     P = 0.5 * (P + P'); 
     const = real(sum(P(:).*log(P(:))));
       
   % minimize KL divergence w.r.t. P_D
    
    P = P*4;
    ydata = 0.0001*randn(n,no_dims);
    
    
    
    %const statement
    momentum = 0.5;
    final_momentum = 0.8;                               % value to which momentum is changed
    mom_switch_iter = 500;                              % iteration at which momentum is changed
    stop_lying_iter = 100;                    % iteration at which lying about P-values is stopped
    if exist('niter')
        max_iter = niter;
    else
        max_iter = 1000;
    end                                               
    epsilon = 500;                                      
    min_gain = .01;  
    y_incs  = zeros(size(ydata));
    gains = ones(size(ydata));
    
    
    % Run the iterations
   
    for iter=1:max_iter
        
        sum_ydata = sum(ydata .^ 2, 2);
        num = 1 ./ (1 + bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')))); % Student-t distribution
        num(1:n+1:end) = 0;                                                 % set diagonal to zero
        Q = max(num ./ sum(num(:)), realmin);                               % normalize to get probabilities
        
        % gradient 
        L = (P - Q) .* num;
        y_grads = 4 * (diag(sum(L, 1)) - L) * ydata;
        
                            
        gains = (gains + .2) .* (sign(y_grads) ~= sign(y_incs)) ...     
              + (gains * .8) .* (sign(y_grads) == sign(y_incs));
        gains(gains < min_gain) = min_gain;
        y_incs = momentum * y_incs - epsilon * (gains .* y_grads);
        ydata = ydata + y_incs;
        ydata = bsxfun(@minus, ydata, mean(ydata, 1));
        
        % 
        if iter == mom_switch_iter
            momentum = final_momentum;
        end
        if iter == stop_lying_iter
            P = P ./ 4;
        end
        
        
        
        
        % print error,keep iteration
        if ~rem(iter, 10)
            cost = const - sum(P(:) .* log(Q(:)));
            disp(['Iteration ' num2str(iter) ': error is ' num2str(cost)]);
  %          figure(1); scatter(ydata(:,1),ydata(:,2),[],label); pause(.2)
        end
        
    end
end

