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
testPoints = [.01,.03,.1,.3,1,3,10,30];

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
	min_error = size(X,1);
	for cur_C = testPoints
		for cur_sigma = testPoints
			model = svmTrain(X, y, cur_C, @(x1, x2) gaussianKernel(x1, x2, cur_sigma));
			error = mean(double(svmPredict(model, Xval) ~=yval));
			if (error < min_error)
				C = cur_C;
				sigma = cur_sigma;
				min_error = error;
			end
		end	
	end






% =========================================================================

end
