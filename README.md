HandwrittenNumeralRecognition_ANN_CUDA
======================================

Handwritten numeral recognition project using BP ANN with CPU &amp; GPU (CUDA).  


## 公告板

* matlab下面的数据文件我导成字符型文件了50M左右4个文件，存在*46/Tmp/CUDA_ANN_DATA*下，要用先同步一下。
* 第一次文件X.dat文件出问题了，中间貌似有错数据。我这边又生成了，晚上过去传46。
	* 正确文件生成方法如下：
	
```
	f=fopen('X2.dat','w');
	for i = 1:5000
	for j= 1:400
	fprintf(f,'%f ',X(i,j));
	end
	end

```
* map函数加了索引参数。用法如下：

```
	Matrix<float> _X2(new float[X.row*(1 + X.col)], X.row, X.col + 1);
	Matrix<float> X2 = _X2.map([&](float, int row, int col){
        return col>0 ? X[row][col - 1] : 1.0f;
    });
	//利用此方法也可以给矩阵添加一列1.0，也可以先留一下，之后也会添加矩阵拼接方法，方便使用。
```

* subr、subc函数按行列取子阵。

```
	//
    //  M.subr(0,M.row); "equal to `M.subr(-1,-1);`"return the whole matrix
    //  M.subr(0,1); return the first row;
    //  this function return a matrix.
    //
```

* Matrix rightlink(Matrix A) 将A拼在this右侧，行数相同
* Matrix underlink(Matrix & A) 将A拼在this下侧，列数相同
