X = double([train0(1:50, :);train1(1:50, :);train2(1:50, :);train3(1:50, :)]);
[U,Z,S] = pca(X);

% B = transpose(U);
% N = 50; % number of images associated with each digit
% numCols = 10;
% numRows = ceil(4*N/numCols);
% d = sqrt(size(B,2));
% figure;
% set(gcf,'color','white');
% %set(gcf,'Position',[520 85 1020 720]); % This command will resize the plot
% %subplot(numRows,numCols,1);
% img = reshape(B(2,:),d,d); % convert each row into 28 x 28 matrix
% imagesc(img); % plot the image
% set(gca,'xtick',[]);
% set(gca,'ytick',[]);
% colormap(gray); % convert the images into gray scale

% figure;
% set(gcf,'color','white');
% plot(Z(1:50,1),Z(1:50,2),'r*'); % images for digit 0 is shown as *
% hold on
% plot(Z(51:100,1),Z(51:100,2),'b+'); % images for digit 1 is shown as +
% 
% plot(Z(101:150,1),Z(101:150,2),'ko'); % images for digit 2 is shown as o
% 
% plot(Z(151:200,1),Z(151:200,2),'gv'); % images for digit 3 is shown as triangles
% hold off

% rank = 50;
% W = Z(:,1:rank)*diag(S(1:rank))*transpose(U(:,1:rank));
% 
% N = 50; % number of images associated with each digit
% numCols = 10;
% numRows = ceil(4*N/numCols);
% d = sqrt(size(W,2));
% figure;
% set(gcf,'color','white');
% set(gcf,'Position',[520 85 1020 720]); % This command will resize the plot
% for i=1:size(W,1);
% subplot(numRows,numCols,i);
% img = reshape(W(i,:),d,d); % convert each row into 28 x 28 matrix
% imagesc(img); % plot the image
% set(gca,'xtick',[]);
% set(gca,'ytick',[]);
% end;
% colormap(gray); % convert the images into gray scale

dist = pdist(X);
med = median(dist);
dist = squareform(dist);

gamma = 1/med^2;

K = exp(-gamma * dist.^2);

numNeighbors = 20;
numZeros = size(K,2) - numNeighbors - 1;
M = size(K,1);
[temp,I] = sort(K,2);
J = repmat([1:M],1,numZeros);
I = I(:,1:numZeros);
I = reshape(I,1,[]);

I = sub2ind(size(K),J,I);
K(I) = 0;

oneN = repmat(1/200,200,200);
Kc = K - oneN*K - K*oneN + oneN*K*oneN;

[V,D] = eig(Kc);
W = V(:, (1:2));
Z = Kc*W;

figure;
set(gcf,'color','white');
plot(Z(1:50,1),Z(1:50,2),'r*'); % images for digit 0 is shown as *
hold on
plot(Z(51:100,1),Z(51:100,2),'b+'); % images for digit 1 is shown as +

plot(Z(101:150,1),Z(101:150,2),'ko'); % images for digit 2 is shown as o

plot(Z(151:200,1),Z(151:200,2),'gv'); % images for digit 3 is shown as triangles
hold off