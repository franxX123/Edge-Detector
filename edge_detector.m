
inputImg = rgb2gray(imread('snorlax.png'));

% inputImg = rgb2gray(imread('charizard.png'));
% img_edge = edge(inputImg, 'canny');
% imshow(img_edge, [])
% figure();
s = 13;
thresh_hi = 150; 
new_img = myCanny(inputImg, s, thresh_hi);
imshow(new_img, [])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Question 2 Part 1 %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cannyImg = myCanny(img, sigma, threshold_hi) 
    [img_row, img_col] = size(img);
    filter = fspecial('gaussian',3,sigma);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%% Question 2 Part 2 %%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    % smooth_img = imfilter(double(img), filter, 'conv');
    % We will perform a single value decomposition and extract the row and
    % column vectors
    
    [U,S,V] = svd(filter);
    vert_filter = U(:,1) * sqrt(S(1,1));
    horz_filter = V(:,1)' * sqrt(S(1,1));
    
    smooth_img_v = imfilter(double(img), vert_filter, 'conv');
    smooth_img = imfilter(double(smooth_img_v), horz_filter, 'conv');

    % Smooth Img output
    % imshow(img, [])
    % figure();
    % imshow(smooth_img, [])
    % figure();

    % Step 1: Find the dy and dx of the image I
    filter = fspecial('sobel');
    dx_filter = filter;
    dy_filter = filter';

    img_dy = imfilter(double(smooth_img), dy_filter, 'conv');
    img_dx = imfilter(double(smooth_img), dx_filter, 'conv');

    % Step 2: Find the gradient magnitude and the gradient orientation.
    img_grad = double(sqrt(img_dy.^2 + img_dx.^2));
    img_ort_y = img_dy./img_grad; % Each cell has a discretized 2d gradient unit vector.
    img_ort_x = img_dx./img_grad;

    for i = 1:img_row 
        for j = 1:img_col

            if img_grad(i,j) ~= 0 

                unit_x = (img_dx(i,j) / img_grad(i,j)); 
                unit_y = (img_dy(i,j) / img_grad(i,j));

                % Discretize the unit vectors
                if unit_x < 0
                    unit_x = floor(unit_x);
                else
                    unit_x = ceil(unit_x);
                end

                if unit_y < 0
                    unit_y = floor(unit_y);
                else
                    unit_y = ceil(unit_y);
                end

                img_ort_y(i,j) = unit_x;
                img_ort_x(i,j) = unit_y;

            else
                img_ort_y(i,j) = 0;
                img_ort_x(i,j) = 0;
            end

        end
    end
    
    % Partial derivative outputs 
    % imshow(img_dy, [])
    % figure();
    % imshow(img_grad, [])
    % figure();

    % Step 3: Now we want to do a NMS (Non-maximum suppression). DONE!
    % We pick threshold lo = 30
    import java.util.LinkedList
    edge_img = double(zeros(img_row, img_col));
    edge_queue = LinkedList();
    threshold_lo = 30;

    for i = 1:img_row 
        for j = 1:img_col
            
            isPeak = checkPeak(i,j,img_ort_x, img_ort_y, img_grad);
    
            % Check if a peak and it is higher than the threshold
            if isPeak &&  img_grad(i,j) > threshold_hi
                edge_img(i,j) = 255;
                edge_queue.add([i,j]);
            end
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%% Question 2 Part 3 %%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Step 4. Hysteresis thresholding! DONE!
    while ~edge_queue.isEmpty()
        edge_pixel = edge_queue.pop();
        x = edge_pixel(1);
        y = edge_pixel(2);

        % check all the neigbours around the pixel in all 8 directions
        % place the valid pixels around the edge pixel to the queue again. A
        % valid pixel is considered to be: 1.) existing 2.) satisfies the low threshold
        % and 3.) is not an edge already
        
        neighbours = [x-1, y-1; x-1, y; x-1, y+1; x, y-1; x, y+1; x+1, y-1; x+1, y; x+1, y+1];
        [rows, cols] = size(neighbours);
        
        for row = 1:rows 
            neigh_x = neighbours(row,1);
            neigh_y = neighbours(row,2);
            neighExist = (neigh_x >= 1 && neigh_x <= img_row) && (neigh_y >= 1 && neigh_y <= img_col);
            
            if neighExist 
                
                neighNotAnEdge = (edge_img(neigh_x,neigh_y) == 0);
                neighSatisfyLo = (img_grad(neigh_x,neigh_y) > threshold_lo);
                neighIsAPeak = checkPeak(neigh_x, neigh_y, img_ort_x, img_ort_y, img_grad);
                
                if neighNotAnEdge && neighSatisfyLo && neighIsAPeak
                    edge_img(neigh_x,neigh_y) = 255;
                    edge_queue.add([neigh_x,neigh_y]);
                end
            end
        end
        
    end
    cannyImg = edge_img;
end

function isPeak = checkPeak(i, j, img_ort_x, img_ort_y, img_grad)

        [img_row, img_col] = size(img_grad);
        isGreaterThanRight = 0; % is it a peak?
        isGreaterThanLeft = 0;
        
        % Check if peak
        ort_x = img_ort_x(i,j);
        ort_y = img_ort_y(i,j);

        % check is "right" side coordinates exists/valid
        right_x = i + ort_x;
        right_y = j + ort_y;

        if (right_x >= 1 && right_x <= img_row) && (right_y <= img_col && right_y >= 1)
            % is img_grad(i,j) greater than right pixel value. 
            isGreaterThanRight = img_grad(i,j) >= img_grad(right_x, right_y);
        else
            isGreaterThanRight = 1;
        end

        % check is "left" side coordinates exists/valid
        left_x = i - ort_x;
        left_y = j - ort_y;

        if (left_x >= 1 && left_x <= img_row) && (left_y <= img_col && left_y >= 1)
            % is img_grad(i,j) greater than left pixel value. 
            isGreaterThanLeft = img_grad(i,j) > img_grad(left_x, left_y);
        else
            isGreaterThanLeft = 1;
        end
        
        isPeak = (isGreaterThanRight == 1 && isGreaterThanLeft == 1);
end
