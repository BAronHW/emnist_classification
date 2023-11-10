load("dataset-letters.mat");

images = dataset.images;
keys = dataset.key;
labels = dataset.labels;

data = double(images);

figure(1);
numExamplesToShow = 12;
randdoub = randperm(size(data,1), numExamplesToShow);

for i = 1:numExamplesToShow
    subplot(3, 4, i);
    corespondingrand = randdoub(i);
    imshow(reshape(data(corespondingrand, :), [28, 28]), []); 
    title(char(labels(corespondingrand) + 64));
end
saveas(gcf, 'random_examples.png');

%partition the dataset into training and testing randomly

% Load your dataset (assuming you have 'data' and 'labels' variables)

% Get the total number of examples in the dataset
totalExamples = size(data, 1);

% Shuffle the indices of the data
shuffledIndices = randperm(totalExamples);

% Calculate the index to split the data into training and testing sets
splitIndex = round(0.5 * totalExamples); % 50% of the data for training, 50% for testing

% Split the shuffled indices into training and testing indices
trainIndices = shuffledIndices(1:splitIndex);
testIndices = shuffledIndices(splitIndex + 1:end);

% Create training and testing datasets using the selected indices
training_data = data(trainIndices, :);
training_labels = labels(trainIndices);

testing_data = data(testIndices, :);
testing_labels = labels(testIndices);

% Display the sizes of the training and testing sets
disp(['Training set size: ' num2str(length(trainIndices)) ' examples']);
disp(['Testing set size: ' num2str(length(testIndices)) ' examples']);

% create two histogram to show distribution of training and testing labels.
figure(2);
histogram(training_labels);
xlabel("Letters");
ylabel("Letter Count");
title("training labels")

figure(3);
histogram(testing_labels);
xlabel("Letters");
ylabel("Letter Count");
title("testing Labels")

%built in knn -------------------------------------------------------
tic
knnmodel = fitcknn(training_data,training_labels);
knnpredictions = predict(knnmodel,testing_data);
toc

%Calculate accuracy of knn
correctPredictions = sum(knnpredictions == testing_labels);
totalTestExamples = length(testing_labels);
knnaccuracy = correctPredictions / totalTestExamples * 100;

disp(['Accuracy of the KNN model: ' num2str(knnaccuracy) '%']);

% plot their knn as a confusion matrix
figure(10);
confusionchart(testing_labels, knnpredictions);


% decision tree
tic
decisiontree = fitctree(training_data,training_labels);
dcpredictions = predict(decisiontree,testing_data);
toc

correctPredictionstree = sum(dcpredictions == testing_labels);
totalTestExamplestree = length(testing_labels);
dcaccuracy = correctPredictionstree / totalTestExamplestree * 100;
disp(['Accuracy of the decision tree model: ' num2str(dcaccuracy) '%']);


figure(11);
confusionchart(testing_labels, dcpredictions);


% euclidean distance knn
% ---------------------------------------------------------

k = 1; %set k value as 1

% Initialize an empty vector for predictions.
tepredict = zeros(size(testing_data, 1), 0);

% Loop through each testing sample
tic
for i = 1:size(testing_data, 1)
    %Calculate distance of current testing sample from all training samples
    comp1 = training_data;
    % calc the current testing sample
    comp2 = testing_data(i, :); 
    % compute squared euclidean distance between current testing sample and
    % all training samples
    l2 = sqrt(sum((comp1 - comp2).^2, 2));
    
    %Get minimum k row indices
    [~, ind] = sort(l2);
    ind = ind(1:k);
    
    %labels for the nearest neighbors
    labs = training_labels(ind);
        
    %Assign the most common label among the nearest neighbors as the prediction
    tepredict(i, 1) = mode(labs);
end

calcAcc(testing_labels,tepredict);
toc



figure(12);
confusionchart(testing_labels, tepredict);



%own Cosine
%-------------------------------------------------------------------------------------
tic
k = 1;

% Initialize array for predictions
tepredict = cell(size(testing_data, 1), 1);

% loop through each testing sample
for i = 1:size(testing_data, 1)
    % Calculate distance of current testing sample from all training samples
    point1 = training_data; %training data
    point2 = testing_data(i, :); % current testing data
    dotProduct = sum(comp1 .* point2, 2); %dot product of training and current testing samples
    magnitudep1 = sqrt(sum(point1.^2, 2)); %magnitude of training samples
    magnitudep2 = sqrt(sum(point2.^2, 2)); %magnitude of current testing sample
    cosineSimilarity = dotProduct ./ (magnitudep1 .* magnitudep2); %cosign similarity calculation
    distance = (1 - cosineSimilarity); %convert similarity to distance
    % Get maximum k row indices (k nearest neighbors with highest cosine similarity)
    [~, ind] = sort(distance);
    ind = ind(1:k);
    % Get labels for the nearest neighbors
    labs = training_labels(ind);
    % Assign the most common label among the nearest neighbors as the prediction
    tepredict{i} = mode(labs);
end

% Calculate accuracy
tepredict = cell2mat(tepredict);
calcAcc(testing_labels,tepredict);
toc

%own Cosine

figure(13);
confusionchart(testing_labels, tepredict);

function calcAcc(testing_labels, tepredict)
    correctPredictions = sum(tepredict == testing_labels);
    totalTestExamples = length(testing_labels);
    accuracy = correctPredictions / totalTestExamples * 100;
    display(accuracy)
end


