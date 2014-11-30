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

template<typename T>
T Matrix_inner_sum(M X)
{
	T temp = (T)0;
	X.map([&](T x){
		temp += x;
		return x;
	});
	return temp;
}

template<typename T>
M Matrix_elementary_multiply(M X,M Y)
{
	return X.map([=](T x, int r, int c){
		return x*Y[r][c];
	});
}

template<typename T>
M Matrix_elementary_minus(M X,M Y)
{
	return X.map([=](T x, int r, int c){
		return x-Y[r][c];
	});
}

template<typename T>
M Matrix_elementary_add(M X,M Y)
{
	return X.map([=](T x, int r, int c){
		return x+Y[r][c];
	});
}

template<typename T>
T sigmoid(T x)
{
    return (T)1 / ((T)1 + exp(-x));
}

template<typename T>
T sigmoidGradient(T x)
{
	T temp = sigmoid<T>(x);
	temp *= 1-temp;
	return temp;
}

M vectorize_col(M X)
{
	temp = X;
	temp.row = temp.row * temp.col;
	temp.col = 1;
	return temp;
}

//nn_params是一个大行向量
int nnCostFunction(M nn_params, int input_layer_size, int hidden_layer_size, int num_lables, M X, M y, DT lambda, double & J, M & grad)
{
	DT temp;
	M Theta1 = nn_params.subc(0,theta1r*theta1c);
	Theta1.row = theta1r;
	Theta1.col = theta1c;
	M Theta2 = nn_params.subc(theta2r*theta2c,-1);
	Theta2.row = theta2r;
	Theta2.col = theta2c;

	M Theta1_part = Theta1.subc(2,-1);
	M Theta2_part = Theta2.subc(2,-1);

	M y_matrix = new M(SAMPLE_NUM, num_lables);
	y_matrix = y_matrix.map([=](DT _, int r, int c){
		return y[r]==c;
	});

	M a1 = (new M((DT)1,SAMPLE_NUM,1)).rightlink(X);
	M z2 = a1 * ~Theta1;
	M a2 = z2.map(sigmoid<DT>);
	a2 = (new M((DT)1,a2.row,1)).rightlink(X);
	M z3 = a2 * ~Theta2;
	a3 = z3.map(sigmoid<DT>);
	//这里要不要用指针？
	M prediction = a3;

	DT a = (DT)1 / SAMPLE_NUM;
	M b = y_matrix.map([=](DT x,int r,int c){
		return -1*x*log(prediction[r][c]);
	});
	M c = y_matrix.map([=](DT x,int r,int c){
		return (1-x)*log(1-prediction[r][c]);
	});
	DT d = lambda / (2 * SAMPLE_NUM);

	DT e = Matrix_inner_sum<DT>(Theta1_part.map([=](DT x, int _, int __){
		return pow(x,(DT)2);
	}));
	DT f = Matrix_inner_sum<DT>(Theta2_part.map([=](DT x, int _, int __){
		return pow(x,(DT)2);
	}));

	//J = sum(a*(b-c))+d*(e+f);
	J = Matrix_inner_sum<DT>(Matrix_elementary_multiply<DT>(Matrix_elementary_minus<DT>(b,c),new M(a,y_matrix.row,y_matrix.col))) + d*(e+f);

	M d3 = Matrix_elementary_minus<DT>(a3, y_matrix);
	M d2 = Matrix_elementary_multiply<DT>(d3 * Theta2_part, z2.map(sigmoidGradient<DT>));
	M delta2 = (~d3)*a2;
	M delta1 = (~d2)*a1;
	M Theta1_grad = delta1.map([=](DT x){return x / SAMPLE_NUM;});
	M Theta2_grad = delta2.map([=](DT x){return x / SAMPLE_NUM;});

	M reg1 = Theta1.map([=](DT x){return x * lambda / SAMPLE_NUM;});
	M reg2 = Theta2.map([=](DT x){return x * lambda / SAMPLE_NUM;});
	
	//这一段代码有点不太和谐，需要修改 -ylxdzsw at 2014.11.30
	auto foo = [=](M & MX){
		for (int i=0;i<MX.row;i++)
		{
			MX[i][0] = 0;
		}
	};
	foo(reg1);foo(reg2);

	Theta1_grad = Matrix_elementary_add<DT>(Theta1_grad,reg1);
	Theta2_grad = Matrix_elementary_add<DT>(Theta2_grad,reg2);

	grad = vectorize_col(Theta1_grad).underlink(vectorize_col(Theta2_grad));

	return 0;
}



#endif