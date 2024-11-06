function mscnet_old_training()
datadir= tempdir;
downloadCIFARData(datadir);

% Load training and validation data
[XTrain, TTrain, XValidation, TValidation] = loadCIFARData(datadir);

fprintf('Setting up...')
params = mscnet_default_settings();

imageAugmenter = imageDataAugmenter( ...
   'RandXReflection', true, ...
   'RandXTranslation', params.pixel_range, ...
   'RandYTranslation', params.pixel_range);

augimdsTrain = augmentedImageDatastore(params.image_size, XTrain, TTrain, ...
   'DataAugmentation', imageAugmenter, ...
   'OutputSizeMode', "randcrop");
augimdsValidation = augmentedImageDatastore(params.image_size, XValidation, TValidation);

layers = [
    imageInputLayer([32 32 3])
    
    convolution2dLayer(5,32,Padding="same")
    batchNormalizationLayer
    leakyReluLayer
    
    maxPooling2dLayer(2, Stride=2)
    
    convolution2dLayer(3,64, Padding="same")
    batchNormalizationLayer
    leakyReluLayer
    dropoutLayer(0.4)
    
    convolution2dLayer(3,128, Padding="same")
    batchNormalizationLayer
    leakyReluLayer
    
    maxPooling2dLayer(2, Stride=2)
    
    convolution2dLayer(3,256, Padding="same")
    batchNormalizationLayer
    leakyReluLayer
    dropoutLayer(0.4)
    
    fullyConnectedLayer(512)
    leakyReluLayer
    dropoutLayer(0.5)
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

% Training options
valFrequency = floor(size(XTrain, 4) / params.train.mini_batch_size);
options = trainingOptions(params.train.optimization_algorithm, ...
    'InitialLearnRate', params.train.learn_rate, ...
    'MaxEpochs', params.train.max_epochs, ...
    'MiniBatchSize', params.train.mini_batch_size, ...
    'VerboseFrequency', valFrequency, ...
    'Shuffle', params.train.shuffle, ...
    'Plots', params.train.plot, ...
    'Verbose', params.train.verbose, ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', valFrequency, ...
    'LearnRateSchedule', params.train.learn_rate_schedule, ...
    'LearnRateDropFactor', params.train.learn_rate_drop_factor, ...
    'LearnRateDropPeriod', params.train.learn_rate_drop_period);


net = trainNetwork(augimdsTrain,layers,options);

save('trained_mcinet_new_arch.mat', 'net');

% validation
pred = classify(net, XValidation);
validationError = mean(pred ~= TValidation);
disp("Validation error: " + validationError*100 + "%")
validationAccuracy = mean(pred == TValidation);
disp("Validation Accuracy: " + validationAccuracy*100 + "%")
Tpred = classify(net, XTrain);
trainError = mean(Tpred ~= TTrain);
disp("Train error: " + trainError*100 + "%")
trainAccuracy = mean(Tpred == TTrain);
disp("Train Accuracy: " + trainAccuracy*100 + "%")

cm = confusionchart(TValidation,pred);
cm.Title = "Confusion Matrix for Validation Data";
cm.ColumnSummary = "column-normalized";
cm.RowSummary = "row-normalized";
end
