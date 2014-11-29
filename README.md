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