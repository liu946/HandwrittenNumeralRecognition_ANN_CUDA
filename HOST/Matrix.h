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
    int cal=0;
    T * dataptr=NULL;
public:
    Matrix(){}
    Matrix(T * _dataptr,int _row,int _cal){
        dataptr=_dataptr;row=_row;cal=_cal;
    }
    Matrix(const Matrix & x){
        row=x.row;
        cal=x.cal;
        dataptr=new T[row*cal];
        for (int i=0; i<row*cal; i++) {
            dataptr[i]=x.dataptr[i];
        }
    }
    ~Matrix(){
        cout<<"~"<<dataptr<<endl;
        delete[] dataptr;
    }
    Matrix operator*(Matrix & mx2){
        
        if (cal!=mx2.row) {
            string err("1 Matrix cal != 2 Matrix row");
            throw err;
        }
        Matrix newnx(new T[row*mx2.cal],row,mx2.cal);
        for (int i=0; i<newnx.row; i++) {
            for (int j=0; j<newnx.cal; j++) {
                newnx[i][j]=0;
                for (int k=0; k<cal; k++) {
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
        return dataptr + cal*rowindex;
    }
    void printMatrix(){
        for (int i=0; i<row; i++) {
            for (int j=0; j<cal; j++) {
                cout<<(*this)[i][j]<<" ";
            }
            cout<<endl;
        }
    }
    Matrix operator~(){
        Matrix newnx(new T[row*cal],cal,row);
        for (int i=0; i<newnx.row; i++) {
            for (int j=0; j<newnx.cal; j++) {
                newnx[i][j]=(*this)[j][i];
            }
        }
        //cout<<newnx->dataptr<<endl;
        return newnx;
    }
    Matrix map(function<T (T)>func){
        Matrix newnx(new T[row*cal],cal,row);
        for (int i=0; i<newnx.row; i++) {
            for (int j=0; j<newnx.cal; j++) {

                newnx[i][j] = func((*this)[i][j]);
                
            }
        }
        return newnx;
    }
};
//
//int main(int argc, const char * argv[]) {
//    // insert code here...
//    int *a=new int[2];
//    int *b=new int[2];
//    *a=1;
//    *(a+1)=2;
//    *b=2;
//    *(b+1)=3;
//    
//    Matrix<int> ax(a,2,1);
//    Matrix<int> bx(b,1,2);
//    Matrix<int> cx=(ax*bx).map([=](int x){
//        return x;
//    });
//    cx.printMatrix();
//    return 0;
//}
#endif
