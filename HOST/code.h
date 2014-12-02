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
#include "stl.h"
#define theta1r 25
#define theta1c 401
#define theta2r 10
#define theta2c 26
#define Xr 5000
#define Xc 400
#define Yr 5000
#define Yc 1
//#include "ANN.h"
using namespace std;
int con=0;
/*
 *矩阵乘法
 ~转置
 [i][j]元素
 map
 
 */
template<typename T>
class Matrix {
    T * dataptr;
    
public:
    
    int row;
    int col;
    Matrix(){}
    Matrix(T * _dataptr, int _row, int _col){
        
        dataptr = _dataptr; row = _row; col = _col;
        
        // con++;
    }
    Matrix(T data,int _row,int _col){
        dataptr = new T[_row*_col]; row = _row; col = _col;
        for (int i=0; i<row*col; i++) {
            dataptr[i]=data;
        }
        
        //   con++;
    }
    Matrix(int _row,int _col){
        dataptr = new T[_row*_col]; row = _row; col = _col;
        
        //   con++;
    }
    
    Matrix(const Matrix & x){
        row = x.row;
        col = x.col;
        con++;
        dataptr = new T[row*col];
        for (int i = 0; i<row*col; i++) {
            dataptr[i] = x.dataptr[i];
        }
        
    }
    ~Matrix(){
        // cout<<"~"<<dataptr<<endl;
        //str.deletenode(str.find(dataptr));
        //con--;
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
    void changemap(function<T(T, int row, int col)>func){
        for (int i = 0; i<row; i++) {
            for (int j = 0; j<col; j++) {
                
                
                (*this)[i][j] = func((*this)[i][j], i, j);
                
            }
        }
        return ;
    }
    //
    //  M.subr(0,M.row); "equal to `M.subr(-1,-1);`"return the whole matrix
    //  M.subr(0,1); return the first row;
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
        Matrix newnx(new T[(stopc - startc)*col],row , stopc - startc);
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
    void printtofile(){
        fstream file("Y1p.txt",ios::out);
        for (int i = 0; i<row; i++) {
            for (int j = 0; j<col; j++) {
                file << (*this)[i][j] << " ";
            }
            file << endl;
        }
        file.close();
    }
    Matrix rightlink(Matrix & A){
        if (A.row!=row) {
            string err("1.row != 2.row");
            throw err;
        }
        int ncol=A.col+col;
        Matrix anx =Matrix<T>( new T[row*ncol],row,ncol);
        for (int i=0; i<row; i++) {
            for (int j=0; j<ncol;j++) {
                anx[i][j]= j<col? (*this)[i][j]:A[i][j-col];
            }
        }
        
        return anx;
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
    Matrix copyM(Matrix x){
        return x;
    }
    
};
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

void vectorize_col(M& X)
{
    X.row = X.row * X.col;
    X.col = 1;
}

//nn_params是一个大lie向量
M nnCostFunction(Matrix<float>& nn_params, int input_layer_size, int hidden_layer_size, int num_lables, Matrix<float>& X, Matrix<int>& y, DT lambda, float & J)
{
    
    M Theta1 = nn_params.subr(0,theta1r*theta1c);
    Theta1.row = theta1r;
    Theta1.col = theta1c;
    M Theta2 = nn_params.subr(theta1r*theta1c,-1);
    Theta2.row = theta2r;
    Theta2.col = theta2c;
    
    M Theta1_part = Theta1.subc(1,-1);
    M Theta2_part = Theta2.subc(1,-1);
    
    M y_matrix1(SAMPLE_NUM, num_lables);
    M y_matrix = y_matrix1.map([&](DT _, int r, int c){
        return (float)(y[r][c]==(c+1));
    });
    cout<<endl<<con++<<endl;
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
    DT a = 1.0f / SAMPLE_NUM;
    M b = y_matrix.map([&](DT x,int r,int c){
        return -1*x*log(prediction[r][c]);
    });
    M c = y_matrix.map([&](DT x,int r,int c){
        return (1-x)*log(1-prediction[r][c]);
    });
    DT d = lambda / (2 * SAMPLE_NUM);
    DT e=0.0f;
    Theta1_part.map([&](DT x, int , int ){
        return e+=x*x;
    });
    DT f =0.0f;
    Theta2_part.map([&](DT x, int , int ){
        return f+=x*x;
    });
    //J = sum(a*(b-c))+d*(e+f);
    J=0;
    //J = Matrix_inner_sum(Matrix_elementary_multiply<DT>(Matrix_elementary_minus<DT>(b,c),temp3)) + d*(e+f);
    b.map([&](float x,int r,int cc){
        return J+=(x-c[r][cc])*a;
    });
    cout<<"a"<<a<<"d"<<d<<"e"<<e<<"f"<<f<<endl;
    J+=d*(e+f);
    //M d3 = Matrix_elementary_minus<DT>(a3, y_matrix);
    M d3 = a3.map([&](float x,int r,int c){
        return x-y_matrix[r][c];
    });
    M temp4 = d3 * Theta2_part;
    //M d2 = Matrix_elementary_multiply<DT>(temp4, z2.map(sigmoidGradient<DT>));
    M d2=z2.map(sigmoidGradient<DT>).map([&](float x,int r,int c){
        return x*temp4[r][c];
    });
    
    M delta2 = (~ d3 )*a2;
    M delta1 = (~ d2 )*a1;
    M Theta1_grad = delta1.map([=](DT x,int,int){return x / SAMPLE_NUM;});
    M Theta2_grad = delta2.map([=](DT x,int,int){return x / SAMPLE_NUM;});
    
    M reg1 = Theta1.map([=](DT x,int,int){return x * lambda / SAMPLE_NUM;});
    M reg2 = Theta2.map([=](DT x,int,int){return x * lambda / SAMPLE_NUM;});
    
    //这一段代码有点不太和谐，需要修改 -ylxdzsw at 2014.11.30
    //    auto foo = [=](M & MX){
    //        for (int i=0;i<MX.row;i++)
    //        {
    //            MX[i][0] = 0;
    //        }
    //    };
    //    foo(reg1);foo(reg2);
    for (int i=0; i<reg1.row; i++) {
        reg1[i][0]=0;
    }
    for (int i=0; i<reg2.row; i++) {
        reg2[i][0]=0;
    }
    
    //    M _Theta1_grad = Matrix_elementary_add<DT>(Theta1_grad,reg1);
    //    M _Theta2_grad = Matrix_elementary_add<DT>(Theta2_grad,reg2);
    Theta1_grad.changemap([&](float x,int r,int c){
        return x+reg1[r][c];
    });
    Theta1_grad.print5x5();
    reg1.print5x5();
    Theta2_grad.changemap([&](float x,int r,int c){
        return x+reg2[r][c];
    });
    vectorize_col(Theta1_grad);
    vectorize_col(Theta2_grad);
    return Theta1_grad.underlink(Theta2_grad);
}


//
//float sigmoid(float x){
//    return 1.0f / (1.0f + exp(-x));
//}
template <typename T = float>
void initdata(string filename, unsigned int size, T * p){
    fstream file(filename,ios::in);
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

int gradientDescent(Matrix<float> & pa,int iters,float& cost,Matrix<float> &X,Matrix<int> &Y){
    
    //Matrix<float> grad;
    for (int i=0; i<iters; i++) {
        M grad = nnCostFunction(pa, 400, 25, 10, X, Y, 0.0f, cost);
        pa.changemap([&](float x, int r, int c){
            return x-grad[r][c];
        });
        
        cout<<"!!!!!"<<endl;
        cout<<cost<<"!!!!!!!!!"<<endl;
        
    }
    return 0;
}
Matrix<int> predict(Matrix<float> &Theta1, Matrix<float> &Theta2, Matrix<float> &X){
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
    //     insert code here...
    //    float * theta1ptr = new float[theta1r*theta1c];
    //    float * theta2ptr = new float[theta2r*theta2c];
    float * xptr = new float[Xr*Xc];
    int * yptr = new int[Yr*Yc];
    //    initdata("Theta1.dat", theta1r*theta1c, theta1ptr);
    //    initdata("Theta2.dat", theta2r*theta2c, theta2ptr);
    initdata("X2.dat", Xr*Xc, xptr);
    initdata<int>("Y.dat", Yr*Yc, yptr);
    //    Matrix<float> Theta1(theta1ptr, theta1r, theta1c);
    //    Matrix<float> Theta2(theta2ptr, theta2r, theta2c);
    Matrix<float> X(xptr, Xr, Xc);
    Matrix<int> Y(yptr, Yr, Yc);
    //Theta1.print5x5();
    //Theta2.print5x5();
    //    cout<<"\n\n\n";
    //    X.print5x5(1500, 150);
    //    Y.print5x5();
    //    Matrix<int> YY = predict(Theta1, Theta2, X);
    //    YY.printtofile();
    Matrix<float>pa(0.12f, theta1r*theta1c+theta2r*theta2c, 1);
    float cost;
    
    gradientDescent(pa, 6, cost, X, Y);
    cout<<"!!!!!"<<endl;
    cout<<cost<<"!!!!!!!!!"<<endl;
    M Theta1 = pa.subr(0,theta1r*theta1c);
    Theta1.row = theta1r;
    Theta1.col = theta1c;
    M Theta2 = pa.subr(theta1r*theta1c,-1);
    pa.print5x5();
    Theta2.row = theta2r;
    Theta2.col = theta2c;
    Theta1.print5x5();
    Theta2.print5x5();
    Matrix<int> YY = predict(Theta1, Theta2, X);
    YY.printtofile();
    return 0;
}
