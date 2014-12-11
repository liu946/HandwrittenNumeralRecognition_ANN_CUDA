
# 串行算法设计

  
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
	- 详细计算过程如下(伪代码)：
		
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
	- 详细计算过程如下(伪代码):
	
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
	- 详细计算过程如下(伪代码)：
		
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


[Pic1-forward-calc]:./img/b.png
[Pic2-costfunction]:./img/fun.png
[Pic3-backpropagation]:./img/f.png

# CUDA部分内容

  
在GPU端的算法设计中，我们详细的进行了显存分配、矩阵乘法优化、同步过程设计等内容的讨论和商定。

### 一、显存分配

- 考虑到GPU端显存限制和C++原本设计模式中矩阵大量的内存拷贝带来的空间和时间的消耗。我们将原来使用大量内存完成的矩阵存储最大程度的共用。
- 我们发现，原本的运算中有大量的增加一列1和删除第一列之后进行的运算，在这种情况下，我们实际上并不需要重新拷贝出一个新的矩阵，只需要使用一种特殊的取元素方式来弥补这个操作即可。于是我们设计出一套矩阵虚拟化方案，简单来说就是使用同一块内存存储数据，并用结构体中的一个字段来存储一个矩阵增加一列、减少一列、正常这三种状态的标志。这种方案会在下一章中详细描述。
- 使用一次性计算的方式减少中间变量矩阵的存储。在一个矩阵乘法之后通常还需要对其结果矩阵进行sigmoid、平方等等运算，我们将这些运算进行合并，由每个线程运算之后再存入目标矩阵，这样就可以充分减少中间矩阵的储存。
- 针对算法，将一些已经不会再在本次循环中使用的矩阵存储空间进行二次使用。充分利用空间。
	
- ######分配空间具体介绍
	- 下面我们介绍一下算法中我们需要使用的内存空间。
		1.	X ——X是一个5000x400的矩阵，这块存储空间与a1共用，a1是一个加一列1的虚拟化矩阵。
		2.	a2——a2是一个5000x25的矩阵，保存着中间隐含层的运算结果，由a1和theta1运算得到，其中保存着是原算法中的z2已经进行过sigmoid运算的值，也是一个加一列1的虚拟化矩阵。这块内存还保存着原来算法中的d2。
		3.	a3——a3是一个5000*10的矩阵，矩阵虚拟化状态为未虚拟化，由a2和theta2运算得到，保存着输出层的值。这块内存还保存着原来算法中的d3。
		4.	theta1——theta1是一个25*401的矩阵，保存着训练结果，在每次迭代中会改变。
		5.	theta2——theta2是一个10*26的矩阵，保存着训练结果，在每次迭代中会改变。
		6.	temptheta2——temptheta2这是一个临时保存原始theta2的内存，也是10*26的矩阵，因为在计算theta1的时候还需要theta2的原始值。需要临时保存一下。
		7.	Y——Y大小5000*1。保存预测结果集。


	- 统计大小：
		- 新算法：sizeof(float)x(5000x(400+25+10)+25x401+10x26) = 8.34M.
		- 原始算法，系统运行报告显示 内存占用21.3M.
		![cpumem][]

### 二、核功能函数设计


- 将乘法设计成三种，分别是正常矩阵乘法(mul)，左矩阵转置×右矩阵(\_mul)，左矩阵×右矩阵转置(mul_)。直接从行列上进行设计，将矩阵乘法方式转换成3种这样就可以省去转置运算的时间和需要占用的空间。
		
		__device__ float mul(M M1,M M2,int threadid)
		__device__ float mul_(M M1,M M2,int threadid)
		__device__ float _mul(M M1,M M2,int threadid)
	
- 对乘法进行了并行化设计，乘法函数需要传入线程全局id，每个线程计算目标矩阵的一个位置上的数据，这样设计可以大大提高运算速度，并且方便进行之后的计算。
		
	
- 我们还设置了一些宏函数，将sigmoid等这些函数变成宏嵌入代码中省去了函数调用的堆栈操作。
		

### 三、核过程函数设计
- 在核运算时需要之中的多次全局同步，这些同步的方法我们选择退回主函数同步显存，我们将每次循环分成如下7个过程。

	
	1.	对X进行虚拟化，得到a1，并计算z2=a1*theta2’，a2=sigmoid(z2)映射。

			
			int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
			M a1 = X;a1.col++;a1.flag=-1;

			if(gid<(a2.col-1)*a2.row){
				a2.ptr[gid] = sigmoid(mul_(a1,theta1,gid));
			}
			

	2. 由计算z3 = a2 × theta2’，和 a3=sigmoid(z3)

			
			int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
			if(gid<(a3.col)*a3.row){
				a3.ptr[gid] = sigmoid(mul_(a2,theta2,gid));
			}
			

	3. 计算d3，保存在a3中，d3 = a3 – 结果集

		
			int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
			if(gid<a3.row){
				a3.ptr[gid*a3.col+Y[gid]-1]-=1;
				//if((gid*a3.col+Y[gid]-1)==4009)printf("%f",a3.ptr[gid*a3.col+Y[gid]-1]);
			}
			

	4. 复制theta2到theta2temp。

			
			int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
			if(gid<theta2.row*theta2.col){
				theta2_temp.ptr[gid]=theta2.ptr[gid];
			}
			

	5. 计算d3’*a2和利用theta2原始数据计算出的reg2的和，并在theta2中减这个值，更新theta2。

			
			int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
			if(gid<theta2.row*theta2.col){
				theta2.ptr[gid]-=(_mul(a3,a2,gid) + ((gid%theta2.col) ? (lambda*theta2.ptr[gid]) : 0.0f))/(float)X.row;
			}
			

	6. 计算d2，d2未虚拟化，需要用到的矩阵为a3，theta2的去掉一列虚拟化矩阵和a2，将这个值存到a2中，这里因为每个线程操作的a2中的元素都不同，并不会因为数据源和目标中都有a2位置的内存而产生冲突。

			
			int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
			a2.col--;a2.flag=0;
			M theta2_1=theta2_t;theta2_1.col--;theta2_1.flag=1;
			if(gid<a2.row*a2.col){
				float temp = a2.ptr[gid];
				a2.ptr[gid]=(mul(a3,theta2_1,gid)*temp*(1-temp));
				//if(gid==0)printf("theta2_1(%d,%d)[%d]:%f\n\n\n",0,0,index(theta2_1,0,0),getitem(theta2_1,0,0));
			}
			

	7. 计算d2’\*a1和利用theta1原始数据计算出的reg1的和，并在theta1中减这个值，更新theta1。
			
			
			int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
			M a1 = X;a1.col++;a1.flag=-1;
			a2.col--;a2.flag=0;
			if(gid<theta1.col*theta1.row){
				if(gid==0){
					float ta=_mul(a2,a1,gid);
					float tb=((gid%theta1.col)?(lambda*theta1.ptr[gid]):0.0f);
				}
				theta1.ptr[gid]-=((_mul(a2,a1,gid)+((gid%theta1.col)?(lambda*theta1.ptr[gid]):0.0f))/X.row);
			}
			

### 四、host主函数设计 
- 主函数中，除了进行必要的内存申请分配释放工作外，使用一个循环来进行每次迭代学习，每次迭代中顺序调用7个核过程函数，并进行了计时。

![zhuhanshu][]

[zhuhanshu]:./img/main.png

# CPU与GPU算法比较


##### 一、简单函数实现
- 使用宏代替函数调用

- 例：
	- 伪代码：
			
			
			sigmoid(z) = 1.0 / (1.0 + exp(-z));
			
			
	- CPU实现
		
			
			template<typename T>
			T sigmoid(T x)
			{
    			return (T)1 / ((T)1 + exp(-x));
			}

			
	- 优化
			
			
			#define sigmoid(x) (1.0f/(1.0f+exp(-(x))))
			
			
##### 二、算法优化
- 我们发现如下两个函数之间有如下关系。
	
		
		sigmoidGradient(z)=sigmoid(z).*(1-sigmoid(z));
		
- 我们如果不保留z而是保留```temp=sigmoid(z)```，我们就可以节省一块存储空间并且实现这个函数。```sigmoidGradient(z)=temp*(1-temp)```
		
##### 三、矩阵运算优化
	
- 使用虚拟化矩阵方案，减少了拷贝和新矩阵构建
	- 我们仔细研究了算法中所需要的运算，矩阵中有多次需要将一个矩阵去掉第一列，或将矩阵左侧链接一列1的操作。
	- CPU端，未优化
		- 使用新矩阵来存储加/减行列之后的矩阵
		
		```
		M temp1(1.0f,X.row,1);
		M a1 = temp1.rightlink(X);
		```
	- GPU端，优化后
		- 使用矩阵结构体中的一个标志标志其虚拟化状态，加一列、减一列还是正常未虚拟化矩阵。
		- 我们使用一个宏来返回该矩阵```M(i,j)```需要返回的值。在这个宏定义中，我们将i,j映射到M指针指到的位置上，同时考虑到虚拟化问题。
			
		```
		#define index(M1,i,j) (i)*(M1.col+(M1).flag)+(j)+(M1).flag
		#define getitem(M1,i,j) (((j)==0 && ((M1).flag)==-1)? 1:(M1).ptr[index(M1,i,j)])
		```
		- 需要左侧加一列，只需要调整标志位和列数即可。
		
		```
		M a1 = X;
		a1.col++;
		a1.flag=-1;
		```
- 使用三个乘法运算，修改矩阵乘法法则，省去了矩阵转置这个运算。
	- 由于我们的矩阵运算大都是需要转置的，计算```A'*B , A*B , A*B'```三种形式的矩阵运算。我们优化后使用3个函数```_mul , mul , mul_```实现这三种乘法，使得可以省去转置这个操作，直接得到结果。
	- CPU端
		- 转置之后进行运算
		
		```
		M Theta1_grad = ((~ d2 )*a1).map([=](DT x,int,int){return x / SAMPLE_NUM;});
		```
	- GPU端
		
		```
		a2.ptr[gid] = sigmoid(mul_(a1,theta1,gid));
		```
		
# 结果展示
##### 一、训练与预测集
- 5000个手写图片和结果。
![5000][]
- 训练结果集为
		
		yi=[(i-1)/500](i=1,2,……,5000)
##### 二、时间效率
- 将这5000个数据训练100次。
    - CPU端运行时间：100次每次迭代2.95s。
    ![cpu100][]
    - GPU端运行时间：100次每次迭代0.42s
    ![gpu100][]
- GPU计算1000次：
    ![gpu1000][]
- 加速比
    
    	Sp=T1/Tp=2.95/0.428=6.89
##### 三、训练准确度
- 训练准确度是受算法限制的，并不是我们最关心的，但是我们还是进行了一些分析。
    - 训练深度100：cost = 1.64
    - 训练深度1000： cost = 0.38
    - 深度1000次的预测准确度 82%
    - 我们还制作了预测程序，如下图。
![preexe][]
 
[preexe]:./img/preexe.png
[cpu100]:./img/CPU100.png
[GPU100]:./img/GPU100.png
[GPU1000]:./img/GPU1000.png
[5000]:./img/5000.png
[cpumem]:./img/cpumem.png