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
void nnCostFunction(Matrix<float>& Theta1,Matrix<float> & Theta2, int input_layer_size, int hidden_layer_size, int num_lables, Matrix<float>& X, Matrix<int>& y, DT lambda, float & J)
{
    
    M y_matrix(SAMPLE_NUM, num_lables);
    y_matrix.changemap([&](DT _, int r, int c){
        return (float)(y[r][0]==(c+1));
    });
    //cout<<endl<<num++<<endl;
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
    
    M & prediction = a3;
    J=0;
    // DT a = 1.0f / SAMPLE_NUM;
    y_matrix.map([&](DT x,int r,int c){
        J+=-1.0f*x*log(prediction[r][c])-(1.0f-x)*log(1-prediction[r][c]);
        return x;
    });
    J/=X.row;
    
    DT d = lambda / (2 * X.row);
    DT e=0.0f;  DT f =0.0f;
    //    M Theta1_part = Theta1.subc(1,-1);
    //    M Theta2_part = Theta2.subc(1,-1);
    Theta1.changemap([&](DT x, int , int c){
        if(c>=1){ e+=x*x;}
        return x;
    });
    Theta2.changemap([&](DT x, int , int c){
        if(c>=1){ f+=x*x;}
        return x;
    });
    
    //J = sum(a*(b-c))+d*(e+f);
    //J = Matrix_inner_sum(Matrix_elementary_multiply<DT>(Matrix_elementary_minus<DT>(b,c),temp3)) + d*(e+f);
    
    J+=d*(e+f);
    //M d3 = Matrix_elementary_minus<DT>(a3, y_matrix);
    M d3 = a3.map([&](float x,int r,int c){
        return x-y_matrix[r][c];
    });
    M Theta2_part = Theta2.subc(1,-1);
    M temp4 = d3 * Theta2_part;
    //M d2 = Matrix_elementary_multiply<DT>(temp4, z2.map(sigmoidGradient<DT>));
    M _z2=temp1.rightlink(z2);
    M _d2=_z2.map(sigmoidGradient<DT>).map([&](float x,int r,int c){
        return x*temp4[r][c];
    });
    M d2=_d2.subc(1, -1);
    // M delta2 = (~ d3 )*a2;
    // M delta1 = (~ d2 )*a1;
    M Theta1_grad = ((~ d2 )*a1).map([=](DT x,int,int){return x / SAMPLE_NUM;});
    M Theta2_grad = ((~ d3 )*a2).map([=](DT x,int,int){return x / SAMPLE_NUM;});
    
    M reg1 = Theta1.map([=](DT x,int,int c){return c==0?0:x * lambda / SAMPLE_NUM;});
    M reg2 = Theta2.map([=](DT x,int,int c){return c==0?0:x * lambda / SAMPLE_NUM;});
    
    //这一段代码有点不太和谐，需要修改 -ylxdzsw at 2014.11.30
    //    auto foo = [=](M & MX){
    //        for (int i=0;i<MX.row;i++)
    //        {
    //            MX[i][0] = 0;
    //        }
    //    };
    //    foo(reg1);foo(reg2);
    
    //    M _Theta1_grad = Matrix_elementary_add<DT>(Theta1_grad,reg1);
    //    M _Theta2_grad = Matrix_elementary_add<DT>(Theta2_grad,reg2);
    Theta1_grad.changemap([&](float x,int r,int c){
        return x+reg1[r][c];
    });
    Theta2_grad.changemap([&](float x,int r,int c){
        return x+reg2[r][c];
    });
    Theta1.changemap([&](float x,int r,int c){
        return x-Theta1_grad[r][c];
    });
    Theta2.changemap([&](float x,int r,int c){
        return x-Theta2_grad[r][c];
    });
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