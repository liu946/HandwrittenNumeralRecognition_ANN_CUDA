//
//  main.cpp
//  MLCppTool
//
//  Created by liu on 11/21/14.
//  Copyright (c) 2014 liu. All rights reserved.
//

#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
using namespace std;
float sigmoid(float x){
    return 1.0f / (1.0f + exp(-x));
}
template <typename T = float>
void initdata(string filename, unsigned int size, T * p){
    fstream file(filename);
    for (unsigned int i = 0; i<size; i++) {
        file >> p[i];
    }
    file.close();
}

int maxindex(float * p, int len) {
    int a = 0;
    float maxdata = p[0];
    for (int i = 1; i<len; i++) {
        if (p[i]>maxdata) {
            maxdata = p[i];
            a = i;
        }
    }
    return a;
}
#define theta1r 25
#define theta1c 401
#define theta2r 10
#define theta2c 26
#define Xr 5000
#define Xc 400
#define Yr 5000
#define Yc 1

Matrix<int> predict(Matrix<float> Theta1, Matrix<float> Theta2, Matrix<float> X){
    int * p=new int[X.row];
    Matrix<float> _theta1 = ~Theta1;
    Matrix<float> _theta2 = ~Theta2;
    Matrix<float> X1(1.0f, X.row,1);
    Matrix<float> X3 = (X1.rightlink(X)*_theta1).map([&](float x, int, int){return sigmoid(x);});
    Matrix<float> X4(1.0f,X3.row,1);
    Matrix<float> X5=(X4.rightlink(X3)*_theta2).map([&](float x, int, int){return sigmoid(x);});
    for (int i = 0; i<X.row; i++) {p[i]=maxindex(X5[i], X5.col);}
    return Matrix<int>(p,X.row,1);
}


int main(int argc, const char * argv[]) {
    // insert code here...
    float * theta1ptr = new float[theta1r*theta1c];
    float * theta2ptr = new float[theta2r*theta2c];
    float * xptr = new float[Xr*Xc];
    int * yptr = new int[Yr*Yc];
    initdata("Theta1.dat", theta1r*theta1c, theta1ptr);
    initdata("Theta2.dat", theta2r*theta2c, theta2ptr);
    initdata("X.dat", Xr*Xc, xptr);
    initdata<int>("Y.dat", Yr*Yc, yptr);
    Matrix<float> Theta1(theta1ptr, theta1r, theta1c);
    Matrix<float> Theta2(theta2ptr, theta2r, theta2c);
    Matrix<float> X(xptr, Xr, Xc);
    Matrix<int> Y(yptr, Yr, Yc);
    Theta1.print5x5();
    Theta2.print5x5();
    X.print5x5(100, 150);
    Y.print5x5();
    Matrix<int> YY = predict(Theta1, Theta2, X);
    YY.print5x5();
    //YY.printMatrix();
    
    return 0;
}
