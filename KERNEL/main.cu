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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cutil_inline.h"
using namespace std;
typedef struct Matrix{
	float * ptr;
	int row;
	int col;
	char flag;
} M;



#define getitem(M1,i,j) (((j)==0 && ((M1).flag)==-1)? 1:(M1).ptr[(i)*(M1.col+(M1).flag)+(j)+(M1).flag])
#define sigmoid(x) (1.0f/(1.0f+exp(-(x))))
__device__ float mul(M M1,M M2,int threadid){
	float a=0.0f;
	for(int i=0;i<(M1).col;i++){
		a+=getitem( M1,threadid/M1.col,i)*getitem(M2,i,threadid%M2.row);
	}
	return a;
}
__device__ float mul_(M M1,M M2,int threadid){
	float a=0.0f;
	for(int i=0;i<(M1).col;i++){
		a+=getitem( M1,threadid/M1.col,i)*getitem(M2,threadid%M2.row,i);
	}
	return a;
}
__device__ float _mul(M M1,M M2,int threadid){
	float a=0.0f;
	for(int i=0;i<(M1).row;i++){
		a+=getitem( M1,i,threadid/M1.col)*getitem(M2,i,threadid%M2.row);
	}
	return a;
}
//参数a2是虚拟化之后的，a3是正常的
__global__ void gradient(M theta1,M theta2,M X,int * Y,float lambda,M a2,M a3) {
	int gid = blockIdx.x*blockDim.x+threadIdx.x;
	M a1 = X;a1.col--;a1.flag=-1;
	if(gid<(a2.col-1)*a2.row){
		a2.ptr[gid] = sigmoid(mul_(a1,theta1,gid));
	}
	__syncthreads();
	//以下计算可以再写一个函数
	if(gid<(a3.col)*a3.row){
		a3.ptr[gid] = sigmoid(mul_(a2,theta2,gid));
	}
	__syncthreads();//在这里，又进行了一次map
	if(gid<a3.row){
		a3.ptr[gid*a3.col+Y[gid]]-=1;
	}
	__syncthreads();
	if(gid<theta2.row*theta2.col){
		theta2.ptr[gid]-=(_mul(a3,a2,gid) + (gid%theta2.row) ? (lambda*theta2.ptr[gid]) : 0.0f)/(float)X.row;
	}
	__syncthreads();
	a2.col--;a2.flag=0;
	M theta2_1=theta2;theta2_1.col--;theta2_1.flag=1;
	if(gid<a2.row*a2.col){
		float temp = a2.ptr[gid];
		a2.ptr[gid]=(mul(a3,theta2_1,gid)*temp*(1-temp));
	}
	__syncthreads();
	if(gid<theta1.col*theta1.row){
		theta1.ptr[gid]-=(_mul(a2,a1,gid)+(gid%theta1.row)?(lambda*theta1.ptr[gid]):0.0f)/(float)X.row;
	}
	__syncthreads();
	a2.flag=-1;a2.col++;
}

//this function will ruin all of a2,X and P.
//a2 should be a virtualized matrix!
//results was placed in the first 5000 elements in a2
__global__ void predict(M Theta1, M Theta2, M X, M a2, M P){
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	M a1 = X;
	a1.flag=-1;a1.col++;
	if(gid<5000*25){
		a2.ptr[gid] = sigmoid(mul_(a1,Theta1,gid));
	}
	__syncthreads();
	if(gid<5000*10){
		//如果sigmoid函数单调，那这里进不进行这个运算应该无所谓 -ylxdzsw 2014.12.6
		P.ptr[gid] = sigmoid(mul_(a2,Theta2,gid));
	}
	__syncthreads();
	if(gid < 5000){
		float temp;
		float max = -1.0f;
		for(int i=0;i<10;i++){
			temp = P.ptr[gid*10+i];
			if(temp>max){
				max = temp;
			}
		}
		a2.ptr[gid] = max;
	}

}

//这里使用map reduce大法可以加快速度，但是需要一块与a3+theta1+theta2同大的内存，复制这块内存所花的时间或许不比节约的时间少 -ylxdzsw 2014.12.7
//tempmemory至少要有两个float大

__device__ int lock = 0;

__global__ void costfunction(M a3, M Theta1, M Theta2, int* yptr, float* J, float lambda, float* tempmemory){
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	if(gid<5000*10){
		float ytemp=(float)(yptr[gid/10]==gid%10);
		float ptemp=a3.ptr[gid];
		//此处还可优化，因为如果让ytemp是int的话，在等于1时后面的log就可以不用算了 -ylxdzsw 2014.12.7
		float costbc = (-1.0f*ytemp)*log(ptemp)-(1.0f-ytemp)*log(1-ptemp);
		do {} while(atomicCAS(&lock,0,1)); // set lock
		*J += costbc;
		__threadfence(); // wait for write completion
		lock = 0; // free lock
	}
	//这里也许不需要同步？
	//__syncthreads();
	Theta1.flag=1;Theta1.col--;
	Theta2.flag=2;Theta2.col--;
	int bid = gid-Theta1.row*Theta1.col;
	if(bid < 0){
		float t1temp = getitem(Theta1,gid/Theta1.col,gid%Theta1.col);
		do {} while(atomicCAS(&lock,0,1)); // set lock
		tempmemory[0] += t1temp*t1temp;
		__threadfence(); // wait for write completion
		lock = 0; // free lock
	}
	if(bid >= 0 && bid < Theta2.row*Theta2.col){
		float t2temp = getitem(Theta2,bid/Theta2.col,bid%Theta2.col);
		do {} while(atomicCAS(&lock,0,1)); // set lock
		tempmemory[1] += t2temp*t2temp;
		__threadfence(); // wait for write completion
		lock = 0; // free lock
	}
	__syncthreads();
	if(gid==0){
		*J += (lambda/2.0f)*(tempmemory[0]+tempmemory[1]);
		*J /= 5000.0f;
	}
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
template <typename T>
void initdata(char * filename, unsigned int size, T * p){
    fstream file(filename,ios::in);
    for (unsigned int i = 0; i<size; i++) {
        file >> p[i];
    }
    file.close();
}


void inittheta(int len,float* p){
	for(int i=0;i<len;i++){
		p[i]=0.12f*(((float)(rand()%1000))/1000.0f*2-1);
	}
}
void printtofile(char * filename,int len,int *ptr ){
    fstream file(filename,ios::out);
    for (int i = 0; i<len; i++) {
        file << ptr[i] << " ";
        file << endl;
    }
    file.close();
}
int main(void) {

	M  h_X, h_theta1, h_theta2,d_X, d_theta1,d_theta2,d_a3,d_a2;
	h_X.ptr=new float[5000*400];h_X.row=5000;h_X.col=400;h_X.flag=0;
	h_theta1.ptr=new float[25*401];h_theta1.row=25;h_theta1.col=401;h_theta1.flag=0;
	h_theta2.ptr= new float[10*26];h_theta2.row=10;h_theta2.col=26;h_theta2.flag=0;
	d_a3.ptr=0;d_a3.row=5000;d_a3.col=10;d_a3.flag=0;
	d_a2.ptr=0;d_a2.row=5000;d_a2.col=26;d_a2.flag=-1;
	int * h_yptr,* d_yptr;

	h_yptr=new int[5000];

	initdata<int>("Y.dat", 5000, h_yptr);
    initdata<float>("X2.dat", h_X.row*h_X.col, h_X.ptr);

    inittheta(25*401,h_theta1.ptr);
    inittheta(10*26,h_theta2.ptr);
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
    cout<<"fuck";
	for(int i=0;i<2;i++){
		gradient<<<5000*25/1024+1,1024>>>(d_theta1,d_theta2,d_X,d_yptr,0.1f,d_a2,d_a3);

	}

	cudaSafeCall(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	cudaSafeCall(cudaGetLastError());
//	CUDA_CHECK_RETURN(cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost));
//
//	for (i = 0; i < WORK_SIZE; i++)
//		printf("Input value: %u, device output: %u\n", idata[i], odata[i]);
//
//	CUDA_CHECK_RETURN(cudaFree((void*) d));
//	CUDA_CHECK_RETURN(cudaDeviceReset());
    cudaSafeCall(cudaFree(d_X.ptr));
    cudaSafeCall(cudaFree(d_theta1.ptr));
    cudaSafeCall(cudaFree(d_theta2.ptr));
    cudaSafeCall(cudaFree(d_a2.ptr));
    cudaSafeCall(cudaFree(d_a3.ptr));
    cudaSafeCall(cudaFree(d_yptr));
    delete [] h_X.ptr;
    delete [] h_theta1.ptr;
    delete [] h_theta2.ptr;
    delete [] h_yptr;
	return 0;
}
