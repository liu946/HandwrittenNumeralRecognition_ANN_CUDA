
## CPU算法设计
---

1. 设计Matrix类简化代码
	- 重载运算符使得其使用变得自然
	- 设计map和changemap方法使得矩阵可以方便地进行各种操作

2. 将梯度计算和代价计算合并在一个函数中，可以相互利用计算结果

## GPU算法设计
---

1. 精确使用显存
	- 尽可能减少临时矩阵的存储，将连续的map操作合并到一起
	- 合理安排计算顺序，在一个矩阵使命结束后其空间可以存放另一矩阵
	- 将矩阵数据分储在各个线程的寄存器中，仅在需要做乘法时写入全局内存

2. 虚拟化矩阵设计
	- 算法中出现多次“添加一列”和“删除第一列”操作，而如果改写矩阵的话将会消耗较多的时间，为此我们设计了虚拟化矩阵的方法，即在矩阵结构体描述中增加标志位，在读取第一列的时候查看标志位来决定是读取内存还是直接返回虚拟的值。

3. 核功能函数设计
	- 

4. 核过程函数设计
	- 由于需要全局同步，我们将每次迭代的过程分成了7个阶段，每个阶段之间需要退回host端保证全局同步。我们设计的7个核过程函数如下。
		1. 计算a2

			- 伪代码

			```
			a1 = [ones(m, 1) X];
			z2 = a1*Theta1';
			a2 = sigmoid(z2);
			```

			- GPU实现(主要部分)

			```
			int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
			M a1 = X;a1.col++;a1.flag=-1;

			if(gid<(a2.col-1)*a2.row){
				a2.ptr[gid] = sigmoid(mul_(a1,theta1,gid));
			}
			```

		2. 计算a3

			- 伪代码

			```
			a2 = [ones(size(a2,1), 1) a2];
			z3 = a2 *Theta2';
			a3 = sigmoid(z3);
			```
			- GPU实现(主要部分)

			```
			int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
			if(gid<(a3.col)*a3.row){
				a3.ptr[gid] = sigmoid(mul_(a2,theta2,gid));
				//if(!(gid%1000))printf("%d,%f\n\n",gid,a3.ptr[gid]);

			}
			```

		3. 计算d3

			- 伪代码

			```
			d3 = a3 - y_matrix;
			```
			
			- GPU实现(主要部分)

			```
			int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
			//在这里，又进行了一次map
			if(gid<a3.row){
				a3.ptr[gid*a3.col+Y[gid]-1]-=1;
				//if((gid*a3.col+Y[gid]-1)==4009)printf("%f",a3.ptr[gid*a3.col+Y[gid]-1]);
			}
			```

		4. 复制Theta2

			- 伪代码

			```
			Theta2_tmep = Theta2
			```

			- GPU实现(主要部分)

			```
			int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
			if(gid<theta2.row*theta2.col){
				theta2_temp.ptr[gid]=theta2.ptr[gid];
			}
			```

		5. Theta2 descent

			- 伪代码

			```
			delta2 = d3'*a2;
			Theta2_grad = delta2*(1/m);
			reg2 = (lambda/m) * Theta2;
			reg2(:,1)= 0;
			Theta2_grad = Theta2_grad + reg2;
			Theta2 -= Theta2_grad
			```

			- GPU实现(主要部分)

			```
			int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
			if(gid<theta2.row*theta2.col){
				theta2.ptr[gid]-=(_mul(a3,a2,gid) + ((gid%theta2.col) ? (lambda*theta2.ptr[gid]) : 0.0f))/(float)X.row;
			}
			```

		6. d2的计算

			- 伪代码

			```
			d2 = (d3*(Theta2(:,2:end))).* (sigmoidGradient(z2));
			```

			- GPU实现(主要部分)

			```
			int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
			a2.col--;a2.flag=0;
			M theta2_1=theta2_t;theta2_1.col--;theta2_1.flag=1;
			if(gid<a2.row*a2.col){
				float temp = a2.ptr[gid];
				a2.ptr[gid]=(mul(a3,theta2_1,gid)*temp*(1-temp));
				//if(gid==0)printf("theta2_1(%d,%d)[%d]:%f\n\n\n",0,0,index(theta2_1,0,0),getitem(theta2_1,0,0));
			}
			```

		7. d2的计算

			- 伪代码

			```
			delta1 = d2'*a1;
			Theta1_grad = delta1*(1/m);
			reg1 = (lambda/m) * Theta1;
			reg1(:,1)= 0;
			Theta1_grad = Theta1_grad + reg1;
			Theta1 -= Theta1_grad
			```

			- GPU实现(主要部分)

			```
			int gid =  (blockIdx.x*blockDim.x+threadIdx.x);
			M a1 = X;a1.col++;a1.flag=-1;
			a2.col--;a2.flag=0;
			if(gid<theta1.col*theta1.row){
				if(gid==0){
					float ta=_mul(a2,a1,gid);
					float tb=((gid%theta1.col)?(lambda*theta1.ptr[gid]):0.0f);

					//printf("ta%f tb%f (ta+0)(%f) \n\n\n",ta,tb,((_mul(a2,a1,gid)+((gid%theta1.col)?(lambda*theta1.ptr[gid]):0.0f))/X.row));
				}

				theta1.ptr[gid]-=((_mul(a2,a1,gid)+((gid%theta1.col)?(lambda*theta1.ptr[gid]):0.0f))/X.row);
			}
			```

5. host主函数设计 


## CPU与GPU算法比较
---

1. #####简单函数实现
	- 使用宏代替函数调用

	- 例：
		- 伪代码：
			
			```
			sigmoid(z) = 1.0 / (1.0 + exp(-z));
			
			```
		- CPU实现
		
			```
			template<typename T>
			T sigmoid(T x)
			{
    			return (T)1 / ((T)1 + exp(-x));
			}

			```
		- 优化
			
			```
			#define sigmoid(x) (1.0f/(1.0f+exp(-(x))))
			``` 
			
2. #####算法优化
	- 我们发现如下两个函数之间有如下关系。
	
		```
		sigmoidGradient(z)=sigmoid(z).*(1-sigmoid(z));
		```
		- 我们如果不保留z而是保留```temp=sigmoid(z)```，我们就可以节省一块存储空间并且实现这个函数。```sigmoidGradient(z)=temp*(1-temp)```
		
3. #####矩阵运算优化
	
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
			
## 结果展示
---
