% coord_data = importdata('/home/pingping/Project/tensorflow/Pixel2Mesh/pixel2mesh/utils/ellipsoid/info_ellipsoid.dat');
load('pkl.mat');
H = round(data_all(:,1)'*size(im,1));
% data_all = [H'/size(im,1) lanes'/size(im,2)];
coord = [data_all(3,2:3)' ones(2,1)*data_all(3,1)];
coord = [coord;data_all(9,2:3)' ones(2,1)*data_all(9,1)];
coord = [coord;data_all(19,2:3)' ones(2,1)*data_all(19,1)];
coord = [coord;data_all(47,2:3)' ones(2,1)*data_all(47,1)];
edge = [1,3;3,5;5,7;7,8;8,6;6,4;4,2;2,1;3,4;5,6];
idx = find(edge(:,1)>edge(:,2));
edge(idx,:) = edge(idx,end:-1:1);
edge = unique(edge,'row');
lap = [1,2,3,4,0,0,1,4;
    1,2,3,4,0,0,2,4;
    1,2,3,4,5,6,3,6;
    1,2,3,4,5,6,4,6;
    3,4,5,6,7,8,5,6;
    3,4,5,6,7,8,6,6;
    5,6,7,8,0,0,7,4;
    5,6,7,8,0,0,8,4;
    ];
lap(:,1:end-1) = lap(:,1:end-1)-1;

% 
pool_idx = [1,3;2,4;3,5;4,6;5,7;6,8];
coord2 = [];
for idx = 1:size(pool_idx,1)
    h = round((coord(pool_idx(idx,1),2) + coord(pool_idx(idx,2),2))*size(im,1)/20)*10;
    coord2 = [coord2;data_all(find(H==h),mod(idx+1,2)+2),data_all(find(H==h),1)];
end
coord2 = [coord;coord2];
edge2 = [
       1,2;3,4;5,6;7,8;
       9,1;9,10;9,3;
       11,3;11,5;11,12;
       5,13;13,7;13,14;
       14,8;14,6;
       12,4;12,6;
       10,4;10,2];
idx = find(edge2(:,1)>edge2(:,2));
edge2(idx,:) = edge2(idx,end:-1:1);
edge2 = unique(edge2,'row');
lap2 = [1,2,9,10,0,0,1,4;
    1,2,9,10,0,0,2,4;
    9,10,3,4,11,12,3,6;
    9,10,3,4,11,12,4,6;
    11,12,5,6,13,14,5,6;
    11,12,5,6,13,14,6,6;
    13,14,7,8,0,0,7,4;
    13,14,7,8,0,0,8,4;
    1,2,9,10,3,4,9,6;
    1,2,9,10,3,4,10,6;
    3,4,11,12,5,6,11,6;
    3,4,11,12,5,6,12,6;
    5,6,13,14,7,8,13,6;
    5,6,13,14,7,8,14,6;
    ];
lap2(:,1:end-1) = lap2(:,1:end-1)-1;

% 
pool_idx2 = [5,13;6,14;13,7;14,8];
coord3 = [];
for idx = 1:size(pool_idx2,1)
    h = round((coord2(pool_idx2(idx,1),2) + coord2(pool_idx2(idx,2),2))*size(im,1)/20)*10;
    coord3 = [coord3;data_all(find(H==h),mod(idx+1,2)+2),data_all(find(H==h),1)];
end
coord3 = [coord2;coord3];
edge3 = [
       1,2;3,4;5,6;7,8;13,14
       9,1;9,10;9,3;
       11,3;11,5;11,12;
       5,15;15,13;15,16;
       13,17;17,7;17,18;
       18,8;18,14;14,16;16,6;
       12,4;12,6;
       10,4;10,2];
idx = find(edge3(:,1)>edge3(:,2));
edge3(idx,:) = edge3(idx,end:-1:1);
edge3 = unique(edge3,'row');
lap3 = [1,2,9,10,0,0,1,4;
    1,2,9,10,0,0,2,4;
    9,10,3,4,11,12,3,6;
    9,10,3,4,11,12,4,6;
    11,12,5,6,15,16,5,6;
    11,12,5,6,15,16,6,6;
    17,18,7,8,0,0,7,4;
    17,18,7,8,0,0,8,4;
    1,2,9,10,3,4,9,6;
    1,2,9,10,3,4,10,6;
    3,4,11,12,5,6,11,6;
    3,4,11,12,5,6,12,6;
    15,16,13,14,17,18,13,6;
    15,16,13,14,17,18,14,6;
    5,6,15,16,13,14,16,6;
    5,6,15,16,13,14,16,6;
    13,14,17,18,7,8,17,6;
    13,14,17,18,7,8,18,6;
    ];
lap3(:,1:end-1) = lap3(:,1:end-1)-1;

% pool_idx = [1,9;1,10;9,3;10,4;3,11;4,12;11,5;12,6;5,13;6,14;13,7;14,8];
% coord3 = [];
% for idx = 1:size(pool_idx,1)
%     h = round((coord_(pool_idx(idx,1),2) + coord_(pool_idx(idx,2),2))*size(im,1)/20)*10;
%     coord3 = [coord3;data(find(H==h),mod(idx+1,2)+2),data(find(H==h),1)];
% end
% coord_ = [coord_;coord3];

coord_ = coord3;
figure(2),
% imagesc(imread([root data{100}.raw_file]))
% hold on,plot(coord_(1:2:end,1)*size(im,2),coord_(1:2:end,2)*size(im,1),'or','linewidth',4);
% hold on,plot(coord_(2:2:end,1)*size(im,2),coord_(2:2:end,2)*size(im,1),'ob','linewidth',4);
for i=1:size(edge,1)
    tmp = [coord(edge(i,1),:);coord(edge(i,2),:)];
    hold on,plot(tmp(:,1)*size(im,2),tmp(:,2)*size(im,1),':or','linewidth',4);
end
for i=1:size(edge2,1)
    tmp = [coord2(edge2(i,1),:);coord2(edge2(i,2),:)];
    hold on,plot(tmp(:,1)*size(im,2),tmp(:,2)*size(im,1),':*b','linewidth',3);
end
for i=1:size(edge3,1)
    tmp = [coord3(edge3(i,1),:);coord3(edge3(i,2),:)];
    hold on,plot(tmp(:,1)*size(im,2),tmp(:,2)*size(im,1),':+g');
end
axis ij
axis off
%%
clear pkl;
pkl{1} = coord;%cat(2,coord,zeros(size(coord,1),1));
% 1
pkl{2}{1,1} = cat(1,1:size(coord,1),1:size(coord,1))'-1; %self
pkl{2}{1,2} = 1.0*ones(1,size(coord,1)); %self
pkl{2}{1,3} = [size(coord,1) size(coord,1)]; %self
pkl{2}{2,1} = edge-1;% edge
pkl{2}{2,2} = 1.0*ones(1,size(edge,1)); %self
pkl{2}{2,3} = [size(coord,1) size(coord,1)]; %self
% 2
pkl{3}{1,1} = cat(1,1:size(coord2,1),1:size(coord2,1))'-1; %self)
pkl{3}{1,2} = 1.0*ones(1,size(coord2,1)); %self
pkl{3}{1,3} = [size(coord2,1) size(coord2,1)]; %self
pkl{3}{2,1} = edge2-1;% edge
pkl{3}{2,2} = 1.0*ones(1,size(edge2,1)); %self
pkl{3}{2,3} = [size(coord2,1) size(coord2,1)]; %self
% 3
pkl{4}{1,1} = cat(1,1:size(coord3,1),1:size(coord3,1))'-1; %self
pkl{4}{1,2} = 1.0*ones(1,size(coord3,1)); %self
pkl{4}{1,3} = [size(coord3,1) size(coord3,1)]; %self
pkl{4}{2,1} = edge3-1;% edge
pkl{4}{2,2} = 1.0*ones(1,size(edge3,1)); %self
pkl{4}{2,3} = [size(coord3,1) size(coord3,1)]; %self
% 
pkl{5}{1} = pool_idx-1;
pkl{5}{2} = pool_idx2-1;
% 
pkl{8}{1} = lap;
pkl{8}{2} = lap2;
pkl{8}{3} = lap3;

pkl{6} = coord_;
save('pkl.mat','pkl','data_all','im')


