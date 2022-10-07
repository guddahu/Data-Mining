A = load('spambase.data');
N = size(A, 1);
seed = 10; % seed for random number generator
rng(seed); % for repeatability of your experiment
A = A(randperm(N),:); % this will reshuffle the rows in matrix A

[m,n] = size(A) ;
P = 0.70 ;
idx = randperm(m)  ;
Training = A(idx(1:round(P*m)),:) ; 
Testing = A(idx(round(P*m)+1:end),:) ;

Ytrain = Training(:,end);
Xtrain = Training(:,1:57);

Ytest = Testing(:,end);
Xtest = Testing(:,1:57);

lambda = logspace(-4,3,11); % create a set of candidate lambda values
SVMmodel = fitclinear(Xtrain, Ytrain, 'Kfold', 5, 'Learner', 'svm', 'Lambda', lambda);
%foldNumber = 3; % to examine the model created for the 3rd fold
%SVMmodel.Trained{foldNumber}
ce = kfoldLoss(SVMmodel) % to examine the classification error for each lambda
bestIdx = min(ce); % identify the index of lambda with smallest error
bestLambda = lambda(1,4);

k = 5;
indices = crossvalind('Kfold', 4601, k);
C = zeros(2,2);
error = 0;
for i = 1:k
    test = (indices == i); 
    train = ~test;
    SVMmodel = fitclinear(A(train, 1:57), A(train,end), 'Learner', 'svm', 'Lambda', bestLambda);
    pred = predict(SVMmodel, A(test, 1:57));
    cp = classperf(A(test, end));
    classperf(cp,pred);
    C = C + cp.DiagnosticTable; % to show the confusion matrix
    error = error + cp.ErrorRate; % to show the classification error
end
C
error/k
% C = confusionmat(Ytest, pred)
% 
% [m,n] = size(Ytest);
% error = 0;
% for c = 1:m
%     if Ytest(c,1) ~= pred(c,1)
%         error = error + 1;
%     end
% end    
% errorRate = error/m   

lambda = logspace(-3,3,11); % create a set of lambda values
sigma = logspace(-3,3,11); % create a set of kernel scale values
cvloss = zeros(11,11); % stores the CV error for each (lambda,sigma) pair
for i=1:11
    for j=1:11
        SVMmodel = fitcsvm(Xtrain, Ytrain, 'KernelFunction','RBF',...
'KernelScale',sigma(j),'BoxConstraint',lambda(i),'Kfold', k);
        cvloss(i,j) =  kfoldLoss(SVMmodel);
    end
end

mini = min(min(cvloss));
[row, column] = find(cvloss == mini)
bestLambda = lambda(1, row);
bestSigma = sigma(1, column);



k = 5;
indices = crossvalind('Kfold', 4601, k);
C = zeros(2,2);
error = 0;
for i = 1:k
    test = (indices == i); 
    train = ~test;
    SVMmodel = fitcsvm(A(train, 1:57), A(train,end), 'KernelFunction','RBF',...
'KernelScale',bestSigma,'BoxConstraint',bestLambda);
    pred = predict(SVMmodel, A(test, 1:57));
    %SVMmodel = fitclinear(A(train, 1:57), A(train,end), 'Learner', 'svm', 'Lambda', bestLambda);
    %pred = predict(SVMmodel, A(test, 1:57));
    cp = classperf(A(test, end));
    classperf(cp,pred);
    C = C + cp.DiagnosticTable; % to show the confusion matrix
    error = error + cp.ErrorRate; % to show the classification error
end
C
error/k


