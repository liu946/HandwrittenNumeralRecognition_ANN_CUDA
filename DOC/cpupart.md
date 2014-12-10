## CPU部分内容
  
为了能与GPU对照以展示并行优化效果，以及为CUDA程序设计提供思路，我们先设计了完全运行于CPU的程序，它也具有我们需要的全部功能。  
CPU段的程序主要分为矩阵类的实现，梯度与代价值的计算以及预测函数的实现。  
  
### 一、矩阵类
  
为简化代码，以及减少错误可能，我们设计了矩阵类。  
  
#### 实现功能

	1. 矩阵构造：需要通过多种方式构造新矩阵，包括指定行列，以及将已有的数据构造成矩阵类型的实例。

	2. 矩阵乘法：我们的算法中矩阵乘法是主要的运算，所以矩阵乘法的实现至关重要。

	3. 矩阵转置：需要得到一个矩阵的转置矩阵。

	4. 矩阵遍历：算法中经常要对一个矩阵所有元素进行操作，将这种行为抽象成矩阵遍历操作可以简化代码以及避免错误。

	5. 添删行列：有时需要对矩阵添删行列。

	6. 矩阵拼接：将两个矩阵拼接在一起。

	6. 矩阵打印：将矩阵输出到文件或者直接输出矩阵的一部分提供预览，这样可以帮助检查程序是否得到正确结果。

#### 设计思路

	- 考虑到矩阵乘法是运算量较大的部分，我们在网上寻找了一些加速矩阵运算的算法，但是我们发现这些算法大多用到递归等操作，并不适合并行化，因此我们还是决定只采用直接的矩阵相乘，以便能与GPU并行程序起到对照效果。

	- 在矩阵复制时将其中的数据指针所指内存也一并复制，在矩阵析构时将数据指针指向内存一并释放，以便使得数据总是和矩阵同生命周期，免去在使用时手动管理内存的困扰。

	- 使用模板类以便能够适应int和float两种类型。

#### 创新点

	- 在矩阵乘法、删除行列等各种操作中进行可能的检查(行列数是否正确等)，可以避免因为C数组越界不报错等问题带来的困扰。

	- 设计遍历操作即map方法和changemap方法，把对矩阵进行倍乘以及两个矩阵直接相加等操作抽象为一个操作，有力地简化了代码。

	- 进行运算符重载，包括方括号运算符[]，乘号运算符*，赋值运算符=，以及取非运算符~等，使得对矩阵的操作的代码得以简化。

#### 实现举例

	1. 矩阵乘法

	```
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
	```

	2. 矩阵遍历

	```
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
	```

### 二、梯度与代价值计算

梯度的计算是功能的核心，而代价值的计算可以用到梯度计算的一部分结果，因此我们将两个功能合并在了一个函数里面。

#### 实现功能

	1. 前向计算估计值，计算方法如下图所示:
		![Pic1-forward-calc][]
		详细计算过程如下(伪代码)：
		```
		a1 = [ones(m, 1) X];
		z2 = a1*Theta1';
		a2 = sigmoid(z2);
		a2 = [ones(size(a2,1), 1) a2];
		z3 = a2 *Theta2';
		a3 = sigmoid(z3);
		```

	2. 代价值计算，计算公式如下图：
		![Pic2-costfunction][]
		详细计算过程如下(伪代码):
		```
		a = (1/m);
		b = (-1*y_matrix).*log(a3);
		c = (1-y_matrix).*log(1-a3);
		d = lambda/(2*m);
		e = (sum(sum(Theta1(:,2:end).^2))); % ^是乘幂运算
		f = (sum(sum(Theta2(:,2:end).^2)));
		cost = sum(sum(a*(b-c)))+ (d*(e+f));
		```

	3. 梯度矩阵计算，计算方法如下图所示：
		![Pic3-backpropagation][]
		详细计算过程如下(伪代码)：
		```
		d3 = a3 - y_matrix;
		d2 = (d3*(Theta2(:,2:end))).* (sigmoidGradient(z2));
		delta2 = d3'*a2;
		delta1 = d2'*a1;
		Theta1_grad = delta1*(1/m);
		Theta2_grad = delta2*(1/m);
		reg1 = (lambda/m) * Theta1;
		reg2 = (lambda/m) * Theta2;
		reg1(:,1)= 0;
		reg2(:,1)= 0;
		Theta1_grad = Theta1_grad + reg1;
		Theta2_grad = Theta2_grad + reg2;
		```

#### 设计思路

	- 为了减少出错可能，我们决定用matlab验证了算法的正确性，然后对应翻译成C++代码，等整体运行成功以后再考虑优化。

	- 通过设计累计变量，然后用map操作进行累加从而实现矩阵元素和的计算。

	- 通过map一个矩阵中的元素，将之与另一矩阵中的对应元素相乘来实现两个矩阵的元素乘法。

#### 创新点

	- 使用lambda表达式来简化一次性函数的声明过程。

	- 通过传引用传参来节省一些矩阵复制时间。

#### 实现举例

	1. 计算d2

	```
    M _z2=temp1.rightlink(z2);
    M _d2=_z2.map(sigmoidGradient<DT>).map([&](float x,int r,int c){
        return x*temp4[r][c];
    });
    M d2=_d2.subc(1, -1);
	```

	2. 计算reg值

	```
    M reg1 = Theta1.map([=](DT x,int,int c){return c==0?0:x * lambda / SAMPLE_NUM;});
    M reg2 = Theta2.map([=](DT x,int,int c){return c==0?0:x * lambda / SAMPLE_NUM;});
	```

### 三、主函数

主函数组织整个流程，包括读取数据，训练神经网络以及完成预测等。

#### 实现功能

	1. 读取文件：从文件中读取数据存放到矩阵中。

	2. 数学函数：包括sigmoid和sigmoidGradient两个函数。

	3. 随机函数：我们在迭代开始之前需要有一个初始参数矩阵，这个矩阵我们决定使用随机数填充。

	4. 预测函数：根据已训练的神经网络参数来预测样本是什么数字。

	5. 输出结果：将预测结果输出到文件。

#### 设计思路
	
	- 为了程序的简洁和可控，我们决定人为选择迭代次数进行简单的for循环迭代，不对梯度下降过程作优化。

	- 在每次迭代中打印代价值，以便追踪检查程序执行效果。

#### 创新点

	- 迭代时通过传引用传参，这样可以在主函数中创建并追踪用到的矩阵，而又不需要在每次调用时复制。

	- 打印迭代前后的Theta矩阵的部分以便快速预览程序运行效果。

#### 实现举例

	1. 迭代过程

	```
    for (int i=0; i<1000; i++) {
        nnCostFunction(Theta1,Theta2,400,25,10,X,Y,1,J);
        cout<<"\n\n"<<i<<" : "<<J<<"\n\n";
    }
	```

	2. 预测函数
	
	```
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
	```



[Pic1-forward-calc]: 
[Pic2-costfunction]:
[Pic3-backpropagation]: