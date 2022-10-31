project_dir = pwd;
% Load landmarks and images
pathToLandmarks = fullfile(project_dir,'DATA','muct_landmarks');
pathToImages = fullfile(project_dir,'DATA','faces');
faceFiles = dir(fullfile(pathToImages,'*.jpg'));

% Load landmarks    
C = importdata(fullfile(pathToLandmarks,'muct76.csv'));
landmarks = C.data(1:end,2:end);
subj_labels = C.textdata(2:end,1);           

% Extract the landmarks and images we want
expression = 'i\d{3}[qrs]a-[fm]n';
mask_landmarks = ~cellfun(@isempty,regexpi(subj_labels,expression));
mask_images = ~cellfun(@isempty,regexp({faceFiles(:).name}',expression));

faceFiles = faceFiles(mask_images);
allLandmarks = landmarks(mask_landmarks,:)';
im = rgb2gray(imread(fullfile(pathToImages,faceFiles(1).name)));
allLandmarks(1:2:end,:) =    allLandmarks(1:2:end,:) + size(im,2)/2; % Adjust for coordinate differences
allLandmarks(2:2:end,:) = -1*allLandmarks(2:2:end,:) + size(im,1)/2;   
save('DATA/Example_FindFace_landmarks_MUCT','allLandmarks')


fid = fopen('DATA/face_file_names.txt', 'a+');
for i=1:size(allLandmarks, 2)
    fprintf(fid, '%s\n', faceFiles(i).name);
end