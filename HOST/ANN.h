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
using namespace std;

typedef DT float;
typedef Matrix<float> M;

#define SAMPLE_NUM 5000

int nnCostFunction(M nn_params, int input_layer_size, int hidden_layer_size, int num_lables, M X, M y, DT lambda, double* J, M* grad)
{
	//TODO
	M Theta1 = new M();
	M Theta2 = new M();
	
	M Theta1_grad = new M(Theta1);
	Theta1_grad = Theta1_grad.map([=](DT _){return (DT)0;});
	M Theta2_grad = new M(Theta2);
	Theta2_grad = Theta2_grad.map([=](DT _){return (DT)0;});

	M all_combos = new M(new DT[num_lables],num_lables,num_lables);
	all_combos = all_combos.map([=](DT _,DT r,DT c){return r==c;});

	M y_matrix = new M(new DT[Yr*Yc], Yr, Yc);
	//TODO
	M a1 = ...
	M z2 = a1 * ~Theta1;
	M a2 = z2.map(sigmoid<DT>);
	//TODO
	a2 = ...
	M z3 = a2 * ~Theta2;
	a3 = z3.map(sigmoid<DT>);
	//这里要不要用指针？
	M prediction = a3;

	DT a = (DT)1 / SAMPLE_NUM;
	//TODO
	M b = y_matrix.map([=](DT x){return -1*x;}).mul(prediction.map(log));
	auto foo = [=](DT x){return 1-x;};
	M c = y_matrix.map(foo).mul(prediction.map(foo).map(log));
	DT d = lambda / (2 * SAMPLE_NUM);
	DT e = ...
	DT d = ...
	*J = ...

	M d3 = a3 - y_matrix;
	M d2 = ...
	M delta2 = ~d3*a2;
	M delta1 = ~d2*a1;
	Theta1_grad = delta1.map([=](DT x){return (DT)1 / SAMPLE_NUM;});
	Theta2_grad = delta2.map([=](DT x){return (DT)1 / SAMPLE_NUM;});

	M reg1 = Theta1.map([=](DT x){return lambda / SAMPLE_NUM;});
	M reg2 = Theta2.map([=](DT x){return lambda / SAMPLE_NUM;});
	//TODO
}



