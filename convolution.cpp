#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


void blur_3(double **p_weight){
    p_weight[0][0] = 1/16.;
    p_weight[0][1] = 2/16.;
    p_weight[0][2] = 1/16.;
    p_weight[1][0] = 2/16.;
    p_weight[1][1] = 4/16.;
    p_weight[1][2] = 2/16.;
    p_weight[2][0] = 1/16.;
    p_weight[2][1] = 2/16.;
    p_weight[2][2] = 1/16.;
}

void blur_5(double **p_weight){
    p_weight[0][0] = 1/256.;
    p_weight[0][1] = 4/256.;
    p_weight[0][2] = 6/256.;
    p_weight[0][3] = 4/256.;
    p_weight[0][4] = 1/256.;
    p_weight[1][0] = 4/256.;
    p_weight[1][1] = 16/256.;
    p_weight[1][2] = 24/256.;
    p_weight[1][3] = 16/256.;
    p_weight[1][4] = 4/256.;
    p_weight[2][0] = 6/256.;
    p_weight[2][1] = 24/256.;
    p_weight[2][2] = 36/256.;
    p_weight[2][3] = 24/256.;
    p_weight[2][4] = 6/256.;
    p_weight[3][0] = 4/256.;
    p_weight[3][1] = 16/256.;
    p_weight[3][2] = 24/256.;
    p_weight[3][3] = 16/256.;
    p_weight[3][4] = 4/256.;
    p_weight[4][0] = 1/256.;
    p_weight[4][1] = 4/256.;
    p_weight[4][2] = 6/256.;
    p_weight[4][3] = 4/256.;
    p_weight[4][4] = 1/256.;
}

void edge(double **p_weight){
    p_weight[0][0] = -1;
    p_weight[0][1] = -1;
    p_weight[0][2] = -1;
    p_weight[1][0] = -1;
    p_weight[1][1] = 8;
    p_weight[1][2] = -1;
    p_weight[2][0] = -1;
    p_weight[2][1] = -1;
    p_weight[2][2] = -1;
}

double getActivation(int *p_channel, int *p_in, int *p_im, double ***p_data){       // Relu function

    for(int z=0; z<*p_channel; z++){
        for(int y=0; y<*p_in; y++){
            for(int x=0; x<*p_im; x++){
                if(p_data[z][y][x]<0){
                    p_data[z][y][x] = ( 1 / (1 + exp(-1*p_data[z][y][x])));
                }      
            }
        }
    }  
    return ***p_data;
}

double saturation(int channel, int n, int m, double ***p_data){       // 최소값 0, 최대값 255 지정

    if(p_data[channel][n][m]<0){
        p_data[channel][n][m] = 0;
    }      
    else if(p_data[channel][n][m]>255){
        p_data[channel][n][m] = 255;
    }
            
    return ***p_data;
}

double matrix_convolution(int *p_channel, int *p_in, int *p_im, int *p_wn, int *p_con, int *p_com, int *p_stride, int *p_padding, double ***p_data, double **p_weight,double ***p_paded, double ***p_con_data){

        //Padding Part

    for(int k=0; k<*p_channel; k++){        // channel 수

        for(int i = 0; i< *p_in + 2*(*p_padding); i++){         // p_paded 초기화
            for(int j = 0; j< *p_im + 2*(*p_padding); j++){               
                p_paded[k][i][j] = 0;
            }
        }

        for(int i = *p_padding; i < *p_in + *p_padding; i++){       // p_paded에 input data 입력
            for(int j = *p_padding; j < *p_im + *p_padding; j++){
                p_paded[k][i][j] = p_data[k][i-*p_padding][j-*p_padding];
            }    
        }

        // Convolution Part        

        int sum = 0;
        for(int i = 0; i< *p_con; i++){        // convolution matrix 초기화
            for(int j = 0; j< *p_com; j++){
                p_con_data[k][i][j] = 0;
            }
        }

        // 행렬곱, 출력 
        for(int y = 0; y < *p_con ; y++){
            for(int x = 0; x < *p_com; x++){

                for(int i = y*(*p_stride); i< y*(*p_stride) + (*p_wn); i++){         
                    for(int j = x*(*p_stride); j< x*(*p_stride) + (*p_wn); j++){               
                        sum += p_paded[k][i][j] * p_weight[i-y*(*p_stride)][j-x*(*p_stride)];
                    }
                }
                p_con_data[k][y][x] = sum;
                saturation(k,y,x,p_con_data);
                sum = 0;
            }
        }
    }
}


double max_pooling(int *p_channel, int *p_poolstride, int *p_kernel, int *p_pn, int *p_pm, double ***p_con_data, double ***p_pooled){

    for(int z=0; z <*p_channel; z++){        // channel 수

        int max=0;                          // pooled 함수 초기화
        for(int i = 0; i< *p_pn; i++){
            for(int j = 0; j< *p_pm; j++){
                p_pooled[z][i][j] = 0;
            }
        }
        for(int y = 0; y < *p_pn; y++){             // Convolution 출력값을 kerner, stride에 맞게 pooling
            for(int x = 0; x < *p_pm; x++){
                for(int i = y*(*p_poolstride); i < *p_kernel + y*(*p_poolstride); i++){
                    for(int j = x*(*p_poolstride); j < *p_kernel + x*(*p_poolstride); j++){
                        if(p_con_data[z][i][j] > max){
                            max = p_con_data[z][i][j];
                        }    
                    }    
                }
                p_pooled[z][y][x] = max;
                saturation(z,y,x,p_pooled);
                max = 0;
            }
        }
    }
}

// 이미지 파일 저장 함수
double image_save(int *p_channel, int *p_pn, int *p_pm, double ***p_pooled){

    Mat image_out;
    image_out.create(*p_pn,*p_pm, CV_8UC3);

    for(int z=0; z<*p_channel; z++){
        for(int y=0; y<*p_pn; y++){
            for(int x=0; x<*p_pm; x++){
                Vec3b* pixel_out = image_out.ptr<Vec3b>(y);
                pixel_out[x][z] = p_pooled[z][y][x];      
            }
        }
    }
    namedWindow("Origin",WINDOW_AUTOSIZE);
    imshow("Origin",image_out);
    imwrite("print.jpg",image_out);

    waitKey(0);
}


// 메인 함수
// 동적메모리 할당받음
int main(){
    double ***p_data;           // point input data matrix
    double **p_weight;         // point filter weight matrix
    double ***p_con_data;       // point convolution matrix
    double ***p_paded;          // ponint paded input data matrix
    double ***p_pooled;         // pooled data matrix

    int stride;
    int padding;

    int wn=3;         // filter weight n value
    int dn;         // paded input data n value
    int dm;         // paded input data m value
    int con;        // convolution n value
    int com;        // convolution m value
    int pn;         // pooled data n value
    int pm;         // pooled data n value
    int channel;

    int kernel;
    int poolstride;


    Mat image;          // image 변수선언 
    image = imread("tree.jpg", IMREAD_COLOR);       // 이미지 파일 불러오기
    
    int im = image.cols;        // input data n value
    int in = image.rows;        // input data m value
    int x = 0;
    int y = 0;
    
    if (image.empty())      // 파일 확인
    {
        cout << "Could not open of find the image" << endl;
        return -1;
    }

    printf("채널 수 : ");
    scanf("%d",&channel);
    printf("\nConvolution Part\n\n");
    printf("Stride 값 : ");
    scanf("%d",&stride);
    printf("Padding 값 : ");
    scanf("%d",&padding);
    printf("\nPooling Part\n\n");
    printf("Kernel 값 : ");
    scanf("%d",&kernel);
    printf("Stride 값 : ");
    scanf("%d",&poolstride);
    
    dn = in + 2*padding;
    con = (in-wn+2*padding)/stride + 1;
    pn = (con-kernel)/poolstride + 1;
    dm = im + 2*padding;
    com = (im-wn+2*padding)/stride + 1;
    pm = (com-kernel)/poolstride + 1;

    int *p_channel;
    p_channel = &channel;
    int *p_in;
    p_in = &in;
    int *p_im;
    p_im = &im;
    int *p_pn;
    p_pn = &pn;
    int *p_pm;
    p_pm = &pm;
    int *p_con;
    p_con = &con;
    int *p_com;
    p_com = &com;
    int *p_wn;
    p_wn = &wn;
    int *p_stride;
    p_stride = &stride;
    int *p_padding;
    p_padding = &padding;   

    int *p_poolstride;
    p_poolstride = &poolstride;
    int *p_kernel;
    p_kernel = &kernel;


    //point input data matrix in x in
    p_data = (double***)malloc(channel*sizeof(double**));
    for(int i=0; i<channel; i++){   
        *(p_data+i) = (double**)malloc(in*sizeof(double*));
        for(int j=0; j<in; j++){
            *(*(p_data+i)+j) = (double*)malloc(im*sizeof(double));
        }
    }
 
    //point filter weight matrix wn x wn
    p_weight = (double**)malloc(wn*sizeof(double*));
    for(int i=0; i<wn; i++){
        *(p_weight+i) = (double*)malloc(wn*sizeof(double));
    }
    //point convolution matrix 
    p_con_data = (double***)malloc(channel*sizeof(double**));
    for(int i=0; i<channel; i++){   
        *(p_con_data+i) = (double**)malloc(con*sizeof(double*));
        for(int j=0; j<con; j++){
            *(*(p_con_data+i)+j) = (double*)malloc(com*sizeof(double));
        }
    }

    //point paded data matrix
    //img.cols  img.rows  img.channels()  img.type()
    p_paded = (double***)malloc(channel*sizeof(double**));
    for(int i=0; i<channel; i++){   
        *(p_paded+i) = (double**)malloc(dn*sizeof(double*));
        for(int j=0; j<dn; j++){
            *(*(p_paded+i)+j) = (double*)malloc(dm*sizeof(double));
        }
    }
    //point Pooled data matrix
    p_pooled = (double***)malloc(channel*sizeof(double**));
    for(int i=0; i<channel; i++){   
        *(p_pooled+i) = (double**)malloc(pn*sizeof(double*));
        for(int j=0; j<pn; j++){
            *(*(p_pooled+i)+j) = (double*)malloc(pm*sizeof(double));
        }
    }
 
    // 입력 함수 호출
    //input_data(p_in, p_data);
    //input_weight(p_wn, p_weight);


        /// type 변환
    for(int z=0; z<channel; z++){
        for(y=0; y<in; y++){
            for(x=0; x<im; x++){
                Vec3b* pixel = image.ptr<Vec3b>(y);
                p_data[z][y][x] = pixel[x][z];
            }
        }
    }

    //blur_3(p_weight);
    //blur_5(p_weight);
    edge(p_weight);
    //clock_t begin = clock();
    matrix_convolution(p_channel, p_in, p_im, p_wn, p_con, p_com, p_stride, p_padding, p_data, p_weight, p_paded, p_con_data);
    
    
    Mat dst(con,com,image.type());
    Mat cvkernel = (Mat_<float>(3,3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);

    filter2D(image,dst,image.depth(),cvkernel/*, Point(-1,-1),0,BORDER_DEFAULT*/); 
    //namedWindow("fileter2d demo",WINDOW_AUTOSIZE);  
    imshow("filter2d demo",dst);
    
    Mat compare(con,com, image.type());
    for(int k=0; k<channel; k++){
        for(int i=0; i<con; i++){
            for(int j=0; j<com; j++){
             //compare.at<float>(i,j) = p_con_data[k][i][j] - cvkernel.at<float>(i,j);
                Vec3b* pixel_out2 = compare.ptr<Vec3b>(i);
                pixel_out2[j][k] =abs(p_con_data[k][i][j] - dst.at<Vec3b>(i,j)[k]);
                printf("%d = %lf - %d\n", pixel_out2[j][k], p_con_data[k][i][j], dst.at<Vec3b>(i,j)[k]);
            }   //channel << 어떻게 처리해야될지 생각.
            printf("\n");
        }
    }

    namedWindow("compare",WINDOW_AUTOSIZE);
    imshow("compare",compare);

    //***p_con_data = getActivation(p_channel, p_in, p_im, p_data);
    //max_pooling(p_channel, p_poolstride, p_kernel, p_pn, p_pm, p_con_data, p_pooled); 
    //clock_t end = clock();
    //double elapsed_secs = double(end - begin)/10000000;
    //printf("\n%lf sec\n",elapsed_secs);

    image_save(p_channel, p_con, p_com, p_con_data);        // 이미지 파일 저장
    waitKey(0);

    //free 선언
    free(p_data);
    free(p_weight);
    free(p_con_data);
    free(p_paded);
    free(p_pooled);
}
