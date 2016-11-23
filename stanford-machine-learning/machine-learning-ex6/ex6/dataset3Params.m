function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
grid = [0.01 0.03 0.1 0.3 1 3 10 30];
grid_size = size(grid);
C_optima = 0;
sigma_optima = 0;
%val_res = zeros(8,8);
val_res = Inf;
for i=1:grid_size(2)
  C = grid(i);
  for j=1:grid_size(2)
    sigma = grid(j);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    val_res_curr = mean(double(predictions ~= yval));
    if val_res_curr < val_res
      val_res = val_res_curr;
      C_optima = C;
      sigma_optima = sigma;
     end
    %val_res(i,j) = mean(double(predictions ~= yval));
  end
end

C = C_optima;
sigma = sigma_optima;





% =========================================================================

end
