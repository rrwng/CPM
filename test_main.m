%% Run this file to see the results of CPM on a number of examples

% The first seven examples are used to show that CPM can preserve geometric structures of the data

% The last two examples use CPM to reveal the geometric structure of the MNIST and the COIL 20 dataset 
clear all; close all
addpath('drtoolbox')
%% Example 1: enclosing clusters
data = [0.1*randn(100,3);randn(100,3)];%;
C = [ones(100,1); 2*ones(100,1)];
figure(1); title(' enclosing clusters')
subplot(2,2,1);scatter3(data(:,1),data(:,2),data(:,3),9,C,'filled'); title('original data');pause(.1);
pause(1);
ydata = compute_mapping(data,'PCA',2);
subplot(2,2,2); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('PCA');pause(.1);
pause(.1);
ydata = compute_mapping(data,'tSNE',2);
subplot(2,2,3); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('tSNE');pause(.1);
pause(.1);
ydata = cpm(data,2,0,1000);
subplot(2,2,4); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('CPM');
disp('press any key to run the next example')
pause;

%% Example 2: 3 clusters of different sizes and distances
close all;
N = 2^8; % number of points considered
data = [rand(N,20);[rand(N,1)+1.5,rand(N,19)];[rand(N,1)+8,rand(N,19)]];
C = [ones(N,1); 2*ones(N,1); 3*ones(N,1)];
figure(1); title(' 3 clusters of different sizes and distances'); 
subplot(2,2,1);scatter(data(:,1),data(:,2),9,C,'filled'); title('original data:'); pause(.1);
ydata = compute_mapping(data,'PCA',2);
subplot(2,2,2); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('PCA'); pause(.1);
ydata = compute_mapping(data,'tSNE',2);
subplot(2,2,3); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('tSNE'); pause(.1);
ydata = cpm(data,2,0,1000);
subplot(2,2,4); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('CPM');
disp('press any key to run the next example')
pause;
%% Example 3: interlacing circles
close all;
t = 2*pi*randn(N,1);
x = cos(t)+randn(N,1)*0.1; y = sin(t)+randn(N,1)*0.1;
data = [x,y];
x =  cos(t)+randn(N,1)*0.1+0.5;  y = sin(t)+randn(N,1)*0.1;
data = [data,zeros(N,1);zeros(N,1),x,y];
C = [ones(N,1); 2*ones(N,1)];
figure(1);  title('interlacing circles')
subplot(2,2,1);scatter3(data(:,1),data(:,2),data(:,3),9,C,'filled'); title('original data'); pause(.1);
ydata = compute_mapping(data,'PCA',2);
subplot(2,2,2); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('PCA');pause(.1);

ydata = compute_mapping(data,'tSNE',2);
subplot(2,2,3); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('tSNE'); pause(.1);
ydata = cpm(data,2,0,1000);
subplot(2,2,4); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('CPM');
disp('press any key to run the next example')
pause;


%% Example 4:three clusters connected by a line.
clear all
close all
p=20;
data = [randn(200,p);randn(200,p)+3/sqrt(p);[0:1:199]'*1/sqrt(p)*ones(1,p)];
data = [data; 3*randn(400,p)+100/sqrt(p)];

C=[ones(200,1);ones(200,1)*2;ones(200,1)*3;ones(400,1)*4];
figure(1); title('Two clusters connected by a line')
subplot(2,2,1);scatter3(data(:,1),data(:,2),data(:,3),9,C,'filled'); title('data: three clusters connected by a line, first 3 dims'); pause(.1);
ydata = compute_mapping(data,'PCA',2);
subplot(2,2,2); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('PCA');pause(.1);
ydata = compute_mapping(data,'tSNE',2);
subplot(2,2,3); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('tSNE');pause(.1);
ydata = cpm(data,2,0,1000);
subplot(2,2,4); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('CPM');
disp('press any key to run the next example')
pause;

%% Example 5: augmented swiss roll
close all
N = 2^10; % number of points considered
t = rand(1,N);
t = sort(2*pi*sqrt(t))'; 
z = 4*pi*rand(N,1); % random heights
x = (t+.1).*cos(t);
y = (t+.1).*sin(t);
w = .5*pi*randn(N,1);
data = [x,y,z,w,randn(N,1)*6,randn(N,1)*6];
%data = data + randn(size(data));
C = ind2rgb(uint8(256*(t.^2+1)/max(t.^2+1)),jet(256));
C = squeeze(C);

figure(1); 
subplot(2,2,1);scatter3(data(:,1),data(:,2),data(:,3),9,C,'filled'); title('data: augmented swiss roll'); pause(.1);
ydata = compute_mapping(data,'Isomap',2);
subplot(2,2,2); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('Isomap');pause(.1);
ydata = compute_mapping(data,'tSNE',2);
subplot(2,2,3); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('tSNE');pause(.1);
ydata = cpm(data,2,1,1000);
subplot(2,2,4); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('CPM');
disp('press any key to run the next example')
pause;

%% Example 6: enclosing balls
clear all
close all
n = 20;
N = 1000;
data=randn(N,n);
X1 = data;
d = sqrt(diag(X1*X1'));
r = prctile(sort(d),30);
    for i=1:size(data,1)
    if norm(data(i,:))<r
        C(i) = 1;
    else
        C(i) = 2;
    end
    end
figure(1);
subplot(2,2,1);scatter3(data(:,1),data(:,2),data(:,3),9,C,'filled'); title(' non-overlapping ball and shell: first three dimensions'); pause(.1);
ydata = compute_mapping(data,'PCA',2);
subplot(2,2,2); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('PCA');pause(.1);
ydata = compute_mapping(data,'tSNE',2);
subplot(2,2,3); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('tSNE');pause(.1);
ydata = cpm(data,2,0,1000);
subplot(2,2,4); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('CPM');
disp('press any key to run the next example')
pause;

%% Example 7: Tree
close all; clear all;
t = 0:0.03:6;
t=t';
branch1 = [zeros(size(t)),t];
branch2 = [2*t, 1/2*t];
branch3 = [2*t, 1/2*t+1/2];
branch4 = [-2*t,1/2*t+1/4];
branch5 = [-2*t,1/2*t+3/4];
data = [branch1;branch2;branch3;branch4;branch5];
C = kron(1:5,ones(1,length(t)));
figure(1);
subplot(2,2,1);scatter(data(:,1),data(:,2),9,C,'filled'); title('True data'); pause(.1);
ydata = compute_mapping(data,'PCA',2);
subplot(2,2,2); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('PCA');pause(.1);
ydata = compute_mapping(data,'tSNE',2);
subplot(2,2,3); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('tSNE');pause(.1);
ydata = cpm(data,2,0,1000);
subplot(2,2,4); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('CPM');
disp('press any key to run the next example')
pause;
%% Example 8: MNIST 
clear all; close all;
load('mnist_test.mat');
k = randperm(10000);
% randomly sample 1000 samples
data = test_X(k(1:1500),:); 
C = test_labels(k(1:1500));
% ind = find(C==5 | C==10);
% x=data(ind,:);
%  C=C(ind);
%  C(C==5)=1;
%  C(C==10)=2;
figure(1); 
disp('Running PCA....')
ydata = compute_mapping(data,'PCA',2);
subplot(2,2,1); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('MNIST result by PCA');pause(.1);
disp('Running tSNE....')
ydata = compute_mapping(data,'tSNE',2,0);
subplot(2,2,2); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('tSNE');pause(.1);
disp('Running CPM....')
ydata = cpm(data,2,1,1000);
subplot(2,2,3); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('CPM');pause(.1);
disp('Running CPM with geodesic distance....')
ydata = cpm(data,2,1,1000,1);
figure(1);
subplot(2,2,4); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('CPM with extra compelling force');
disp('press any key to run the next example')
pause;
%% Example 9: COIL20
close all; clear all;
disp('Importing Coil20 dataset')
load('COIL20');
data = fea; C = gnd;
figure(1); 
disp('Running PCA....')
ydata = compute_mapping(data,'PCA',2); 
subplot(3,2,1); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('COIL 20 result by PCA');pause(.1);
disp('Running tSNE....')
ydata = compute_mapping(data,'tSNE',2,0); 
subplot(3,2,2); scatter(ydata(:,1),ydata(:,2),9,C,'filled'); title('tSNE'); pause(.1);
disp('Running CPM....')
ydata = cpm(data,3,0,1500);
subplot(2,2,3); scatter3(ydata(:,1),ydata(:,2),ydata(:,3),9,C,'filled'); title('CPM with Euclidean distance');pause(.1);
disp('Running CPM with geodesic distance....')
ydata = cpm(data,3,1,600);
subplot(2,2,4); scatter3(ydata(:,1),ydata(:,2),ydata(:,3),9,C,'filled'); title('CPM with geodesic distance');

