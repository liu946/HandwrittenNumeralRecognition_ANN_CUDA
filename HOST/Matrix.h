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

/*
 *矩阵乘法
 ~转置
 [i][j]元素
 map
 subr，subc取子阵
 */
template<typename T>
class Matrix {
    
    T * dataptr;
public:
    int row;
    int col;
    
    Matrix(T * _dataptr, int _row, int _col){
        
        dataptr = _dataptr; row = _row; col = _col;
    }
    Matrix(T data,int _row,int _col){
        dataptr = new T[_row*_col]; row = _row; col = _col;
        for (int i=0; i<row*col; i++) {
            dataptr[i]=data;
        }
    }
    Matrix(int _row,int _col){
        dataptr = new T[_row*_col]; row = _row; col = _col;
    }
    Matrix(const Matrix & x){
        row = x.row;
        col = x.col;
        dataptr = new T[row*col];
        for (int i = 0; i<row*col; i++) {
            dataptr[i] = x.dataptr[i];
        }
    }
    ~Matrix(){
        //  cout<<"~"<<dataptr<<endl;
        delete[] dataptr;
    }
    Matrix operator*(Matrix & mx2){
        
        if (col != mx2.row) {
            string err("1 Matrix col != 2 Matrix row");
            throw err;
        }
        Matrix newnx(new T[row*mx2.col], row, mx2.col);
        for (int i = 0; i<newnx.row; i++) {
            for (int j = 0; j<newnx.col; j++) {
                newnx[i][j] = 0;
                for (int k = 0; k<col; k++) {
                    newnx[i][j] += (*this)[i][k] * mx2[k][j];
                }
            }
        }
        return newnx;
    }
    T * operator[](int rowindex){
        if (rowindex >= row) {
            string err("on index of calling row");
            throw err;
        }
        return dataptr + col*rowindex;
    }
    void printMatrix(){
        for (int i = 0; i<row; i++) {
            cout << "\n\n" << i << "\n\n";
            for (int j = 0; j<col; j++) {
                cout << (*this)[i][j] << " ";
            }
            cout << endl;
        }
    }
    void print5x5(unsigned int startr = 0, unsigned int startc = 0){
        for (unsigned int i = startr; i<row&&i<5 + startr; i++) {
            for (unsigned int j = startc; j<col&&j<5 + startc; j++) {
                cout << (*this)[i][j] << "\t";
            }
            cout << endl;
        }
    }
    Matrix operator~(){
        Matrix newnx(new T[row*col], col, row);
        for (int i = 0; i<newnx.row; i++) {
            for (int j = 0; j<newnx.col; j++) {
                newnx[i][j] = (*this)[j][i];
            }
        }
        //cout<<newnx->dataptr<<endl;
        return newnx;
    }
    Matrix map(function<T(T, int row, int col)>func){
        T * p = new T[row*col];
        Matrix newnx(p, row, col);
        for (int i = 0; i<newnx.row; i++) {
            for (int j = 0; j<newnx.col; j++) {
                
                
                newnx[i][j] = func((*this)[i][j], i, j);
                
            }
        }
        return newnx;
    }
    //
    //  M.subr(0,M.row); "equal to `M.subr(-1,-1);`"return the whole matrix
    //  M.subr(0,1); return the first row;
    //  this function return a matrix.
    //
    Matrix subr(int startr, int stopr){
        if (stopr <= 0 || stopr>row) stopr = row;
        if (startr<0 || startr >= row)startr = 0;
        if (startr >= stopr) {
            string err("startrow > stoprow");
        }
        Matrix newnx(new T[(stopr - startr)*col], stopr - startr, col);
        //注意，不能在map里面使用this指针
        for (int i = 0; i < stopr - startr; i++)
        {
            for (int j = 0; j < col; j++)
            {
                newnx[i][j] = (*this)[i + startr][j];
            }
        }
        
        return newnx;
    }
    Matrix subc(int startc, int stopc){
        if (stopc <= 0 || stopc>col) stopc = col;
        if (startc<0 || startc >= col)startc = 0;
        if (startc >= stopc) {
            string err("startcol > stopcol");
        }
        Matrix newnx(new T[(stopc - startc)*col], stopc - startc, col);
        //注意，不能在map里面使用this指针
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < stopc-startc; j++)
            {
                newnx[i][j] = (*this)[i][j+startc];
            }
        }
        
        return newnx;
    }
    Matrix rightlink(Matrix & A){
        if (A.row!=row) {
            string err("1.row != 2.row");
            throw err;
        }
        int ncol=A.col+col;
        T * p =new T[row*ncol];
        for (int i=0; i<row; i++) {
            for (int j=0; j<ncol;j++) {
                p[i*ncol+j]= j<col? (*this)[i][j]:A[i][j];
            }
        }
        return Matrix(p,row,ncol);
    }
    Matrix underlink(Matrix & A){
        if (A.col!=col) {
            string err("1.col != 2.col");
            throw err;
        }
        int nrow=A.row+row;
        T * p =new T[nrow*col];
        int i=0;
        while (i<row*col) {
            p[i]=(*this)[0][i];
            i++;
        }
        int j=0;
        while (i<nrow*col) {
            p[i]=A[0][j];
            i++,j++;
        }
        return Matrix(p,nrow,col);
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
