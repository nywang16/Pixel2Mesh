clear all;
path(path,'./jsonlab');
% path(path,'/home/pingping/Project/SCNN/tools/prob2lines');
if 1
    root = '/home/pingping/Project/dataset/tusimple/train_set/';
    folder = {'/home/pingping/Project/dataset/tusimple/train_set/label_data_0313.json';
        '/home/pingping/Project/dataset/tusimple/train_set/label_data_0531.json';
        '/home/pingping/Project/dataset/tusimple/train_set/label_data_0601.json'};
else
    root = '/home/pingping/Project/dataset/tusimple/test_set/';
    folder = {'/home/pingping/Project/dataset/tusimple/test_set/test_baseline.json'};
end
save = 0;
load('pkl.mat');
load('tusimple_train_data.mat');
edge = pkl{2}{2,1} + 1;
edge2 = pkl{3}{2,1} + 1;
edge3 = pkl{4}{2,1} + 1;
num_all = 0;
if save
    fid = fopen('pixel2mesh.txt','w');
end
for file_n = 1:100:length(folder)
    data = loadjson(folder{file_n});
    for i=1:1:length(data)
        num_all = num_all + 1
        im = imread([root data{i}.raw_file]);
        lanes = data{i}.lanes;
        h_samples = data{i}.h_samples;
        [height,width,~] = size(im);
        
        %spline fit
        lane = [];
        end_position = [];
        point_reduced = [];
        num = 0;
        for j=1:size(lanes,1)
            idx = find(lanes(j,:)~=-2);
            if length(idx)>1
                num = num + 1;
                point = cat(1,lanes(j,idx),h_samples(idx));
                lane(num,:) = lanes(j,:);
                PointList_reduced = DouglasPeucker(point', 1, 0);
                end_position(num) = polyval(polyfit(PointList_reduced(end-1:end,2),PointList_reduced(end-1:end,1),1),size(im,1));
                point_reduced{num} = PointList_reduced;
            end
        end
        [val,loc] = sort(end_position,'ascend');
        end_c = end_position(loc);
        lane = lane(loc,:);
        left_loc = find(end_c == max(end_c(find(end_c<size(im,2)/2))));
        end_num = (1:length(end_c)) - left_loc + 2;
        if length(intersect(end_num,[2,3]))<2
            disp(sprintf('skip-1 %s',data{i}.raw_file));
            continue;
        end
        %%
        coord = pkl{6};
        h = round(coord(:,2)*height);
        for k=1:2
            %             [c2,ia2,ib2] = intersect(h_samples',h(k:2:end));
            %             [~,loc] = sort(ib2,'ascend');
            %             coord(k:2:end,1) = lane(find(end_num==k+1),ia2(loc))'/width;
            [coord(k:2:end,2),coord(k:2:end,1)] = poly_fit_by_T(im,h_samples,lane(find(end_num==k+1),:),H_list(num_all,:),h(k:2:end));
        end
        if sum(isnan(coord(:)))>0 | max(abs(coord(:)))>2
            disp(sprintf('skip-2 %s',data{i}.raw_file));
            continue;
        end
        if save
            fprintf(fid,'%s',data{i}.raw_file);
            for k=1:size(coord,1)
                fprintf(fid,' %.4f %.4f',coord(k,:));
            end
            fprintf(fid,'\n');
        end
        
        if 1
            figure(2),imagesc(imread([root data{i}.raw_file]))
%             for k=1:size(lanes,1)
%                 idx = find(lanes(k,:)~=-2);
%                 hold on,plot(lanes(k,idx),h_samples(idx),'-ob');
%             end
            hold on,plot(coord(1:2:end,1)*size(im,2),coord(1:2:end,2)*size(im,1),'or','linewidth',2);
            hold on,plot(coord(2:2:end,1)*size(im,2),coord(2:2:end,2)*size(im,1),'ob','linewidth',2);
            coord_ = coord;
            for i=1:size(edge3,1)
                tmp = [coord_(edge3(i,1),:);coord_(edge3(i,2),:)];
                hold on,plot(tmp(:,1)*size(im,2),tmp(:,2)*size(im,1),'-+y');
            end
            axis off
        end
    end
end
if save
    fclose(fid)
end