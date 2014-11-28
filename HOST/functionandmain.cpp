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
    int m = X.row;
    //int num_lables = Theta2.row;
    Matrix<float> _X2(new float[X.row*(1 + X.col)], X.row, X.col + 1);
    Matrix<float> _theta1 = ~Theta1;
    Matrix<float> _theta2 = ~Theta2;
    Matrix<float> X2 = _X2.map([&](float, int row, int col){
        return col>0 ? X[row][col - 1] : 1.0f;
    });
    
    
    for (int i = 0; i<m; i++) {
        Matrix<float> z2 = (X2.subr(i, i + 1)*_theta1);
        Matrix<float> _z2= z2.map([&](float x, int, int){
            return sigmoid(x);
        });
        Matrix<float> z2_(new float[(_z2.col + 1)*_z2.row], _z2.row, _z2.col + 1);
        Matrix<float> z2__ = z2_.map([&](float, int row, int col){
            return col == 0 ? 1.0 : z2_[row][col];
        });
        Matrix<float> z2___ = (z2__*_theta2).map([&](float x, int, int){
            return sigmoid(x);
        });
        p[i] = maxindex(z2___[0], z2___.col);
    }
    return Matrix<int>( p,m,1);
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
