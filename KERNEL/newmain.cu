/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <cuda_runtime.h>
#include <cutil_inline.h>
using namespace std;
typedef struct Matrix{
	float * ptr;
	int row;
	int col;
	char flag;
} M;


//bug1
#define getitem(M1,i,j) (((j)==0 && ((M1).flag)==-1)? 1:(M1).ptr[(i)*(M1.col+(M1).flag)+(j)+(M1).flag])
#define index(M1,i,j) (i)*(M1.col+(M1).flag)+(j)+(M1).flag
#define sigmoid(x) (1.0f/(1.0f+exp(-(x))))
__device__ float mul(M M1,M M2,int threadid){
	float a=0.0f;
	for(int i=0;i<(M1).col;i++){
		a+=getitem( M1,threadid/M2.col,i)*getitem(M2,i,threadid%M2.col);
	}
	return a;
}
__device__ float mul_(M M1,M M2,int threadid){
	float a=0.0f;


	//if(threadid==21000)printf("M2(%d,%d)[%d]=%f\n",threadid%M2.row,0,index(M2,threadid%M2.row,0),getitem(M2,0,0));
	for(int i=0;i<(M1).col;i++){
		float t=getitem( M1,threadid/(M2.row),i)*getitem(M2,threadid%M2.row,i);
		a+=t;
	}
	//if(threadid==21000){printf("a(%f)=getitem( M1(%d*%d),threadid/(M2.row)(%d),i)*getitem(M2(%d*%d),threadid%M2.row(%d),i);\n",
	//								a,M1.row,M1.col,threadid/(M2.row),M2.row,M2.col,threadid%M2.row);}
	//if(threadid==21000)printf("M2(%d,%d)[%d]=%f\n",threadid%M2.row,0,index(M2,threadid%M2.row,0),getitem(M2,0,0));
	return a;
}
__device__ float _mul(M M1,M M2,int threadid){
	float a=0.0f;
	for(int i=0;i<(M1).row;i++){
		a+=getitem( M1,i,threadid/M2.col)*getitem(M2,i,threadid%M2.col);
	}
	return a;
}
//参数a2是虚拟化之后的，a3是正常的
__global__ void nnCostFunction1(M theta1,M theta2,M X,int * Y,float lambda,M a2,M a3) {
	int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
	M a1 = X;a1.col++;a1.flag=-1;

	if(gid<(a2.col-1)*a2.row){
		a2.ptr[gid] = sigmoid(mul_(a1,theta1,gid));
	}
}
__global__ void nnCostFunction2(M theta1,M theta2,M X,int * Y,float lambda,M a2,M a3) {
	int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
	if(gid<(a3.col)*a3.row){
		a3.ptr[gid] = sigmoid(mul_(a2,theta2,gid));
		//if(!(gid%1000))printf("%d,%f\n\n",gid,a3.ptr[gid]);

	}
}
__global__ void nnCostFunction3(M theta1,M theta2,M X,int * Y,float lambda,M a2,M a3) {
	int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
	//在这里，又进行了一次map
	if(gid<a3.row){
		a3.ptr[gid*a3.col+Y[gid]-1]-=1;
		//if((gid*a3.col+Y[gid]-1)==4009)printf("%f",a3.ptr[gid*a3.col+Y[gid]-1]);
	}
}
__global__ void nnCostFunction4(M theta1,M theta2,M X,int * Y,float lambda,M a2,M a3) {
	int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
	if(gid<theta2.row*theta2.col){
		theta2.ptr[gid]-=(_mul(a3,a2,gid) + ((gid%theta2.col) ? (lambda*theta2.ptr[gid]) : 0.0f))/(float)X.row;
	}
}
__global__ void nnCostFunction5(M theta1,M theta2,M X,int * Y,float lambda,M a2,M a3) {
//	if(gid==0){
//	for(int i = 0;i<5;i++){
//		for(int j=0;j<5;j++){
//			printf("%f\t",getitem(theta2,i,j));
//		}
//		printf("\n");
//	}
//	}
////	//~debug
	int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
	a2.col--;a2.flag=0;
	M theta2_1=theta2;theta2_1.col--;theta2_1.flag=1;
	if(gid<a2.row*a2.col){
		float temp = a2.ptr[gid];
		a2.ptr[gid]=(mul(a3,theta2_1,gid)*temp*(1-temp));
	}
}
__global__ void nnCostFunction6(M theta1,M theta2,M X,int * Y,float lambda,M a2,M a3) {
	int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
	M a1 = X;a1.col++;a1.flag=-1;
	a2.col--;a2.flag=0;
	if(gid<theta1.col*theta1.row){
		theta1.ptr[gid]-=(_mul(a2,a1,gid)+(gid%theta1.row)?(lambda*theta1.ptr[gid]):0.0f)/(float)X.row;
	}
}
__global__ void printM(M MM,int a,int b){
	for(int i = a;i<5+a;i++){
		for(int j=b;j<5+b;j++){
			printf("%f\t",getitem(MM,i,j));
		}
		printf("\n");
	}
}
/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <typename T = float>
void initdata(char * filename, unsigned int size, T * p){
    fstream file(filename,ios::in);
    for (unsigned int i = 0; i<size; i++) {
        file >> p[i];
    }
    file.close();
}


void inittheta(int len,float* p){
	for(int i=0;i<len;i++){
		p[i]=0.12f;//*(((float)(rand()%1000))/1000.0f*2-1);
	}
}
void printtofile(char * filename,int len,int *ptr ){
    fstream file(filename,ios::out);
    for (int i = 0; i<len; i++) {
        file << ptr[i] << " ";
    }
    file.close();
}
void printtofile(char * filename,int len,float *ptr ){
    fstream file(filename,ios::out);
    for (int i = 0; i<len; i++) {
        file << ptr[i] << " ";
    }
    file.close();
}
int main(void) {

	M  h_X, h_theta1, h_theta2,d_X, d_theta1,d_theta2,d_a3,d_a2;
	h_X.ptr=new float[5000*400];h_X.row=5000;h_X.col=400;h_X.flag=0;
	h_theta1.ptr=new float[401*25];h_theta1.row=25;h_theta1.col=401;h_theta1.flag=0;
	h_theta2.ptr= new float[26*10];h_theta2.row=10;h_theta2.col=26;h_theta2.flag=0;
	d_a3.ptr=0;d_a3.row=5000;d_a3.col=10;d_a3.flag=0;
	d_a2.ptr=0;d_a2.row=5000;d_a2.col=26;d_a2.flag=-1;
	int * h_yptr,* d_yptr;
	double timer, elapsed , elapsedcp;
	h_yptr=new int[5000];

	initdata<int>("Y.dat", 5000, h_yptr);
    initdata("X2.dat", h_X.row*h_X.col, h_X.ptr);

    inittheta(401*25,h_theta1.ptr);
    inittheta(26*10,h_theta2.ptr);

    elapsed_time(&timer);
    d_X=h_X;d_theta1=h_theta1;d_theta2=h_theta2;
    cudaSafeCall(cudaMalloc((void**) &d_X.ptr, sizeof(float) *d_X.col*d_X.row ));
    cudaSafeCall(cudaMalloc((void**) &d_yptr, sizeof(int) *5000 ));
    cudaSafeCall(cudaMalloc((void**) &d_theta1.ptr, sizeof(float) *d_theta1.col*d_theta1.row ));
    cudaSafeCall(cudaMalloc((void**) &d_theta2.ptr, sizeof(float) *d_theta2.col*d_theta2.row ));
    cudaSafeCall(cudaMalloc((void**) &d_a2.ptr, sizeof(float) *5000*25 ));
    cudaSafeCall(cudaMalloc((void**) &d_a3.ptr, sizeof(float) *d_a3.col*d_a3.row ));

    cudaSafeCall(cudaMemcpy(d_X.ptr, h_X.ptr, sizeof(float)*d_X.col*d_X.row,cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_yptr, h_yptr, sizeof(float)*5000,cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_theta1.ptr, h_theta1.ptr, sizeof(float)*d_theta1.col*d_theta1.row,cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_theta2.ptr, h_theta2.ptr, sizeof(float)*d_theta2.col*d_theta2.row,cudaMemcpyHostToDevice));
    elapsedcp = elapsed_time(&timer);
    int cycle=1;
  //for(int i=cycle;i>0;i--){
    	nnCostFunction1<<<5000*25/1024+1,1024>>>(d_theta1,d_theta2,d_X,d_yptr,1.0f,d_a2,d_a3);

    	nnCostFunction2<<<5000*25/1024+1,1024>>>(d_theta1,d_theta2,d_X,d_yptr,1.0f,d_a2,d_a3);

    	nnCostFunction3<<<5000*25/1024+1,1024>>>(d_theta1,d_theta2,d_X,d_yptr,1.0f,d_a2,d_a3);
    	nnCostFunction4<<<5000*25/1024+1,1024>>>(d_theta1,d_theta2,d_X,d_yptr,1.0f,d_a2,d_a3);

    	nnCostFunction5<<<5000*25/1024+1,1024>>>(d_theta1,d_theta2,d_X,d_yptr,1.0f,d_a2,d_a3);

    	nnCostFunction6<<<5000*25/1024+1,1024>>>(d_theta1,d_theta2,d_X,d_yptr,1.0f,d_a2,d_a3);
    	printM<<<1,1>>>(d_theta2,0,0);
    	printM<<<1,1>>>(d_theta1,0,0);
   // }
    elapsed = elapsed_time(&timer);
    cudaSafeCall(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaMemcpy(h_theta1.ptr, d_theta1.ptr,sizeof(float)*d_theta1.col*d_theta1.row, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(h_theta2.ptr, d_theta2.ptr,sizeof(float)*d_theta2.col*d_theta2.row, cudaMemcpyDeviceToHost));
    elapsedcp += elapsed_time(&timer);

	printtofile("Theta1.dat",h_theta1.col*h_theta1.row,h_theta1.ptr);
	printtofile("Theta2.dat",h_theta2.col*h_theta2.row,h_theta2.ptr);

    cudaSafeCall(cudaFree(d_X.ptr));
    cudaSafeCall(cudaFree(d_theta1.ptr));
    cudaSafeCall(cudaFree(d_theta2.ptr));
    cudaSafeCall(cudaFree(d_a2.ptr));
    cudaSafeCall(cudaFree(d_a3.ptr));
    cudaSafeCall(cudaFree(d_yptr));
    cout<<"copy time "<<"cost time (for "<<cycle<<") : "<<elapsed<<"\n for per loop : "<<elapsed/cycle<<"\n for copy: "<<elapsedcp;
    delete [] h_X.ptr;
    delete [] h_theta1.ptr;
    delete [] h_theta2.ptr;
    delete [] h_yptr;
	return 0;
}
