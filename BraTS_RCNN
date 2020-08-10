imageDir = fullfile('/home/priya/Documents/MATLAB');
if ~exist(imageDir,'dir')
    mkdir(imageDir);
end
sourceDataLoc = [imageDir filesep 'Task01_BrainTumour'];
preprocessDataLoc = fullfile(imageDir,'BraTS','preprocessedDataset1');
%preprocessBraTSdataset1(preprocessDataLoc,sourceDataLoc);
%% 
% load tumor Dataset

data = load('tumorDataset.mat');
tumorDataset = data.tumorDataset;
%%
prop = (1/150);
prop2 = (1/1000);
rng(0)
shuffledIndices = randperm(height(tumorDataset));
idx = floor(prop * height(tumorDataset));

trainingIdx = 1:idx;
trainingDataTbl = tumorDataset(shuffledIndices(trainingIdx),:);
validationIdx = idx+1 : idx + 1 + floor(prop2 * length(shuffledIndices) );
validationDataTbl = tumorDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = tumorDataset(shuffledIndices(testIdx),:);
%%
testIdx_short = validationIdx(end)+1 : idx +2 + floor(prop * length(shuffledIndices));
testDataTbl_short = tumorDataset(shuffledIndices(testIdx_short),:);
%%
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'tumor'));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'tumor'));

imdsTest = imageDatastore(testDataTbl_short{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl_short(:,'tumor'));
%% 
% Combine image and box label datastores.

trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);
%% 
% Display one of the training images and box labels.

data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
%% 
% First, specify the network input size. To reduce the computational cost of 
% running the example, specify a network input size of [224 224 3], which is the 
% minimum size required to run the network. 

inputSize = [224 224 3];
%% 
% Next, use |estimateAnchorBoxes| to estimate anchor boxes based on the size 
% of objects in the training data. To account for the resizing of the images prior 
% to training, resize the training data for estimating anchor boxes. Use |transform| 
% to preprocess the training data, then define the number of anchor boxes and 
% estimate the anchor boxes.

numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(trainingData,numAnchors)
%% 
% Now, use |resnet18| to load a pretrained ResNet-18 model. 

% featureExtractionNetwork = resnet18;
featureExtractionNetwork = resnet50;
%% 
% Select the feature extraction layer. 

featureLayer = 'activation_40_relu';
%% 
% Define the number of classes to detect.

numClasses = width(tumorDataset)-1;
%% 
% Create the Faster R-CNN object detection network.

lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);
%% Train Faster R-CNN
% Use |trainingOptions| to specify network training options. Set |'ValidationData'| 
% to the preprocessed validation data. Set |'CheckpointPath'| to a temporary location. 
% This enables the saving of partially trained detectors during the training process. 
% If training is interrupted, such as by a power outage or system failure, you 
% can resume training from the saved checkpoint.
% 
% 

options = trainingOptions('sgdm',...
    'MaxEpochs',1000,...
    'MiniBatchSize',2,...
    'InitialLearnRate',1e-3,...
    'CheckpointPath',tempdir,...
    'ValidationData',validationData);
%% 
% Use |trainFasterRCNNObjectDetector| to train Faster R-CNN object detector 
% if |doTrainingAndEval| is true. Otherwise, load the pretrained network.

gpuDevice(2)
doTrainingAndEval = true
if doTrainingAndEval
    % Train the Faster R-CNN detector.
    % * Adjust NegativeOverlapRange and PositiveOverlapRange to ensure
    %   that training samples tightly overlap with ground truth.
    [detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,options, ...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1]);
else
    'training set to false'
end
%% 
% This example was verified on an Nvidia(TM) Titan X GPU with 12 GB of memory. 
% Training the network took approximately 20 minutes. The training time varies 
% depending on the hardware you use.
%% 
% As a quick check, run the detector on one test image. Make sure you resize 
% the image to the same size as the training images. 

I = imread(testDataTbl_short.imageFilename{5});
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);
%% 
% Display the results.

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)
%% 
% Run the detector on all the test images.

if doTrainingAndEval
    detectionResults = detect(detector,testData,'MinibatchSize',4);
else
    'eval set to false'
end
% if doTrainingAndEval
%     detectionResults2 = detect(detector,trainingData,'MinibatchSize',4);
% else
%     'eval set to false'
% end
%% 
% Evaluate the object detector using the average precision metric.

[ap, recall, precision] = evaluateDetectionPrecision(detectionResults,testData);
%% 
% The precision/recall (PR) curve highlights how precise a detector is at varying 
% levels of recall. The ideal precision is 1 at all recall levels. The use of 
% more data can help improve the average precision but might require more training 
% time. Plot the PR curve.

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
%% 
% Supporting functions

function data = augmentData(data)
% Randomly flip images and bounding boxes horizontally.
tform = randomAffine2d('XReflection',true);
rout = affineOutputView(size(data{1}),tform);
data{1} = imwarp(data{1},tform,'OutputView',rout);
data{2} = bboxwarp(data{2},tform,rout);
end

function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to targetSize.
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
data{2} = bboxresize(data{2},scale);
end
