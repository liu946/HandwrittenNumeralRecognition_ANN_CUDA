//
//  ANN.h
//  MLCppFunctions
//
//  Created by ylxdzsw on 11/28/14.
//  Copyright (c) 2014 ylxdzsw. All rights reserved.
//
#ifndef ANN_H
#define ANN_H

#include "Matrix.h"
#include <cmath>
using namespace std;

typedef float DT;
typedef Matrix<float> M;

#define SAMPLE_NUM 5000


float Matrix_inner_sum(M X)
{
    float temp = 0.0f;
    X.map([&](float x,int,int){
        temp += x;
        return x;
    });
    return temp;
}

template<typename T>
M Matrix_elementary_multiply(M X,M Y)
{
    return X.map([&](T x, int r, int c){
        return x*Y[r][c];
    });
}

template<typename T>
M Matrix_elementary_minus(M X,M Y)
{
    return X.map([&](T x, int r, int c){
        return x-Y[r][c];
    });
}

template<typename T>
M Matrix_elementary_add(M X,M Y)
{
    return X.map([&](T x, int r, int c){
        return x+Y[r][c];
    });
}

template<typename T>
T sigmoid(T x)
{
    return (T)1 / ((T)1 + exp(-x));
}

template<typename T>
T sigmoidGradient(T x,int,int)
{
    T temp = sigmoid<T>(x);
    temp *= 1-temp;
    return temp;
}

M vectorize_col(M X)
{
    M temp = X;
    temp.row = temp.row * temp.col;
    temp.col = 1;
    return temp;
}

//nn_params是一个大lie向量
M nnCostFunction(Matrix<float> nn_params, int input_layer_size, int hidden_layer_size, int num_lables, Matrix<float> X, Matrix<int> y, DT lambda, float & J)
{
    
    M Theta1 = nn_params.subr(0,theta1r*theta1c);
    Theta1.row = theta1r;
    Theta1.col = theta1c;
    M Theta2 = nn_params.subr(theta2r*theta2c,-1);
    Theta2.row = theta2r;
    Theta2.col = theta2c;
    
    M Theta1_part = Theta1.subc(1,-1);
    M Theta2_part = Theta2.subc(1,-1);
    
    M y_matrix1(SAMPLE_NUM, num_lables);
    M y_matrix = y_matrix1.map([&](DT _, int r, int c){
        return (float)(y[r][c]==c);
    });
    
    M temp1((DT)1,SAMPLE_NUM,1);
    M a1 = temp1.rightlink(X);
    Matrix<float> _theta1 = ~Theta1;
    Matrix<float> _theta2 = ~Theta2;
    M z2 = a1 * _theta1;
    M _a2(1.0f,z2.row,1);
    M ta2= z2.map([&](float x, int, int){return sigmoid(x);});
    M a2=_a2.rightlink(ta2);
    
    
    M z3 = a2 * _theta2;
    M a3 = z3.map([&](float x, int, int){return sigmoid(x);});
    //这里要不要用指针？
    M prediction = a3;
    
    DT a = (DT)1 / SAMPLE_NUM;
    M b = y_matrix.map([&](DT x,int r,int c){
        return -1*x*log(prediction[r][c]);
    });
    M c = y_matrix.map([&](DT x,int r,int c){
        return (1-x)*log(1-prediction[r][c]);
    });
    DT d = lambda / (2 * SAMPLE_NUM);
    
    DT e = Matrix_inner_sum(Theta1_part.map([=](DT x, int , int ){
        return pow(x,(DT)2);
    }));
    DT f = Matrix_inner_sum(Theta2_part.map([=](DT x, int , int ){
        return pow(x,(DT)2);
    }));
    
    //J = sum(a*(b-c))+d*(e+f);
    M temp3(a,y_matrix.row,y_matrix.col);
    J = Matrix_inner_sum(Matrix_elementary_multiply<DT>(Matrix_elementary_minus<DT>(b,c),temp3)) + d*(e+f);
    
    M d3 = Matrix_elementary_minus<DT>(a3, y_matrix);
    
    M temp4 = d3 * Theta2_part;
    M d2 = Matrix_elementary_multiply<DT>(temp4, z2.map(sigmoidGradient<DT>));
    M delta2 = (~ d3 )*a2;
    M delta1 = (~ d2 )*a1;
    M Theta1_grad = delta1.map([=](DT x,int,int){return x / SAMPLE_NUM;});
    M Theta2_grad = delta2.map([=](DT x,int,int){return x / SAMPLE_NUM;});
    
    M reg1 = Theta1.map([=](DT x,int,int){return x * lambda / SAMPLE_NUM;});
    M reg2 = Theta2.map([=](DT x,int,int){return x * lambda / SAMPLE_NUM;});
    
    //这一段代码有点不太和谐，需要修改 -ylxdzsw at 2014.11.30
    auto foo = [=](M & MX){
        for (int i=0;i<MX.row;i++)
        {
            MX[i][0] = 0;
        }
    };
    foo(reg1);foo(reg2);
    
    M _Theta1_grad = Matrix_elementary_add<DT>(Theta1_grad,reg1);
    M _Theta2_grad = Matrix_elementary_add<DT>(Theta2_grad,reg2);
    
    M temp5 = vectorize_col(_Theta1_grad);
    M temp6 = vectorize_col(_Theta2_grad);
    return temp5.underlink(temp6);
}


int gradientDescent(Matrix<float> & pa,int iters,float& cost,Matrix<float> &X,Matrix<int> &Y){
    
    //Matrix<float> grad;
    for (int i=0; i<iters; i++) {
        M grad = nnCostFunction(pa, 400, 25, 10, X, Y, 0.05f, cost);
        pa.changemap([&](float x, int r, int c){
            return x-grad[r][c];
        });
    }
    return 0;
}



#endif