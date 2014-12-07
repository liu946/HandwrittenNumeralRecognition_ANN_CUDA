b=zeros(1,100*20+1);
for i=1:50
a=zeros(20,1);
for j=1:100
a=[a reshape((X((i-1)*100+j,:)./max(X((i-1)*100+j,:))*255)',20,20)];
end
b=[b;a];
end
image(b);colormap gray;