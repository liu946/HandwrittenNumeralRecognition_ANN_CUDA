//
//  Matrix.h
//  MLCppTool
//
//  Created by liu on 11/21/14.
//  Copyright (c) 2014 liu. All rights reserved.
//
#ifndef MATRIX_H

#define MATRIX_H
#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
using namespace std;
template<typename T>
/*
 *矩阵乘法
 ~转置
 [i][j]元素
 map
 */

class Matrix {
    int row=0;
    int col=0;
    T * dataptr=NULL;
public:
    Matrix(){}
    Matrix(T * _dataptr,int _row,int _col){
        dataptr=_dataptr;row=_row;col=_col;
    }
    Matrix(const Matrix & x){
        row=x.row;
        col=x.col;
        dataptr=new T[row*col];
        for (int i=0; i<row*col; i++) {
            dataptr[i]=x.dataptr[i];
        }
    }
    ~Matrix(){
        //  cout<<"~"<<dataptr<<endl;
        delete[] dataptr;
    }
    Matrix operator*(Matrix & mx2){
        
        if (col!=mx2.row) {
            string err("1 Matrix col != 2 Matrix row");
            throw err;
        }
        Matrix newnx(new T[row*mx2.col],row,mx2.col);
        for (int i=0; i<newnx.row; i++) {
            for (int j=0; j<newnx.col; j++) {
                newnx[i][j]=0;
                for (int k=0; k<col; k++) {
                    newnx[i][j]+= (*this)[i][k] * mx2[k][j];
                }
            }
        }
        return newnx;
    }
    T * operator[](int rowindex){
        if (rowindex>=row) {
            string err("on index of calling row");
            throw err;
        }
        return dataptr + col*rowindex;
    }
    void printMatrix(){
        for (int i=0; i<row; i++) {
            cout<<"\n\n"<<i<<"\n\n";
            for (int j=0; j<col; j++) {
                cout<<(*this)[i][j]<<" ";
            }
            cout<<endl;
        }
    }
    void print5x5(unsigned int startr=0,unsigned int startc=0){
        for (unsigned int i=startr; i<row&&i<5+startr; i++) {
            for (unsigned int j=startc; j<col&&j<5+startc; j++) {
                cout<<(*this)[i][j]<<"\t";
            }
            cout<<endl;
        }
    }
    Matrix operator~(){
        Matrix newnx(new T[row*col],col,row);
        for (int i=0; i<newnx.row; i++) {
            for (int j=0; j<newnx.col; j++) {
                newnx[i][j]=(*this)[j][i];
            }
        }
        //cout<<newnx->dataptr<<endl;
        return newnx;
    }
    Matrix map(function<T (T)>func){
        Matrix newnx(new T[row*col],col,row);
        for (int i=0; i<newnx.row; i++) {
            for (int j=0; j<newnx.col; j++) {
                
                newnx[i][j] = func((*this)[i][j]);
                
            }
        }
        return newnx;
    }
};
void initdata(string filename,unsigned int size,float * p){
    fstream file(filename);
    for (unsigned int i=0; i<size; i++) {
        file>>p[i];
    }
    file.close();
}
#define theta1r 25
#define theta1c 401
#define theta2r 10
#define theta2c 26
#define Xr 5000
#define Xc 400
#define Yr 5000
#define Yc 1
//int main(int argc, const char * argv[]) {
//    // insert code here...
//    float * theta1ptr=new float[theta1r*theta1c];
//    float * theta2ptr=new float[theta2r*theta2c];
//    float * xptr=new float[Xr*Xc];
//    float * yptr=new float[Yr*Yc];
//    initdata("Theta1.dat", theta1r*theta1c, theta1ptr);
//    initdata("Theta2.dat", theta2r*theta2c, theta2ptr);
//    initdata("X.dat", Xr*Xc, xptr);
//    initdata("Y.dat", Yr*Yc, yptr);
//    Matrix<float> Theta1(theta1ptr,theta1r,theta1c);
//    Matrix<float> Theta2(theta2ptr,theta2r,theta2c);
//    Matrix<float> X(xptr,Xr,Xc);
//    Matrix<float> Y(yptr,Yr,Yc);
//    Theta1.print5x5();
//    Theta2.print5x5();
//    X.print5x5(100,150);
//    Y.print5x5();
//    
//    
//    return 0;
//}

#endif
