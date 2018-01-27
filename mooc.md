#R语言进行统计分析
###Steps
1. 简单认识R并了解用法
2. 利用一个实例学习如何利用R语言进行统计分析过程  

###Obtain and Install R
[Link to R for Mac OS X](https://cran.r-project.org/bin/macosx/)  
[Link to RStudio for Mac OS X](https://www.rstudio.com/products/rstudio/download/)

###Programming with R
R中常用"<-"进行赋值，等价于Python中赋值语句中的"=" 
  
普通运算如下

| Operator | Functionality| Example |
| :--| --------:|:---:|
| +  | Addition |a = 1+2 |
| -  | Sbutraction | a = 2-1 |
| * | Multiplication | a = 1*2 |
| / | Division | a = 2/1 |
| ^ | Raised to a power | a = 2^2 |

逻辑和关系操作符如下  

| Operator | Functionality| 
| :--| --------:|
| &| And |
| \| |Or |
|! | Not|
|==| Equal|
|!=| Not Equal|
|<| Less than|
|>| Greater than |
|<= | Less than or equal to|
|>= | Greater than or equal to|

###R Programming Example -- Get Data
```
#鸢尾花数据集#
iris
#UCI Machine Learning Repository#
iris <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), header = FALSE) 

```
###Process Data
```
#features#
colnames(iris)
#观察前5行的数据情况#
head(iris,5)
#建立与iris数据的链接，可以避免因为变量名称过长引用麻烦#
attach(iris)
#基本描述iris数据#
summary(iris)
```
1. 花萼长度(Sepal Length)：计算单位是公分。  
2. 花萼宽度(Sepal Width)：计算单位是公分。  
3. 花瓣长度(Petal Length) ：计算单位是公分。  
4. 花瓣宽度(Petal Width)：计算单位是公分。  
5. 类別(Class)：可分为Setosa，Versicolor和Virginica三个品种。
###Plotting
```
#在花瓣长度和宽度上观察种类区分#
plot(Petal.Length, Petal.Width, pch=21, bg=c("red","green3","blue")[unclass(iris$Species)], main="Edgar Anderson's Iris Data")
#在所有四种特征上的区分#
pairs(iris[1:4], main = "Anderson's Iris Data -- 3 species", pch = 21, bg = c("red", "green3", "blue")[unclass(iris$Species)], lower.panel=NULL, labels=c("SL","SW","PL","PW"), font.labels=2, cex.labels=4.5) 
```
[Link for plot symbol] (http://www.sthda.com/english/wiki/r-plot-pch-symbols-the-different-point-shapes-available-in-r)

###评估正态分布--图形法：QQplot
```
qqnorm(Sepal.Length)
qqline(Sepal.Length,col="red")
```
QQ 即 Quantile-Quantile。  
例如5.7的分位数是0.05，在Normal Distribution中对应分位数是0.05的数（qnorm(0.05))是-1.64，对应在QQ plot中点(-1.64, 5.7)。通过以上方法，可以每个样本点描绘出来，这就是QQ plot。  
所以y轴就是你的样本的范围，x轴就是标准正态分布下的z-score。而qqline则代表经过样本中第一和第三分位点（0.25，0.75）的正态分布与标准正态分布形成的直线。  
[qq plot百度百科](https://baike.baidu.com/item/QQPlot%E5%9B%BE)

###检验两个品种的差异性：T-test H0：μ1 = μ2
在做T-test检验均值是否相同之前，需要用F-test检验方差的差异性  
F-test H0: sigma1/sigma2 = 1     

```
xtabs(~Species)  
setosa=subset(iris,Species=="setosa")  
versicolor=subset(iris,Species=="versicolor")
var.test(setosa$Petal.Width, versicolor$Petal.Width)
t.test(setosa$Petal.Width, versicolor$Petal.Width, var.equal=FALSE)
```
###检验三个品种的差异性：ANOVA 
H0:μSetosa=μVersicolor=μVirginica     
H1:至少有一种平均数和其他品种不相等

```
iris.model = lm(Petal.Width ~Species,data = iris)
summary(iris.model)

m1 = aov(Petal.Width ~Species,data = iris)
TukeyHSD(m1)
#可视化展现#
boxplot(Petal.Width ~ Species, data = iris)
```
###分类###
1. 决策树
2. 随机森林
3. SVM

```
library(tree)
#划分训练集#
train_rows = 0.2*nrow(iris)
test.index = sample(1:nrow(iris),train_rows)
train_data = iris[-test.index,]
test_data = iris[test.index,]

#决策树分类#
tree_result = tree(Species~ . ,data=train_data);tree_result
plot(tree_result);text(tree_result)

#得到结果#
tree.pred = predict(treemodel,newdata=test_data,type="class")
table.test = table(Species=test_data$Species,Predicted=tree.pred)
table.test
tree.train = predict(treemodel,newdata=train_data,type="class")
table.train = table(Species=train_data$Species,Predicted=tree.train)
table.train

#随机森林#
library(randomForest)
set.seed(777)#固定数据#
iris.rf=randomForest(Species~.,data=train_data,importane=T,proximity=T)
iris.rf
#特征重要性#
round(importance(iris.rf),2)
#Confusion Matrix#
table.rf=iris.rf$confusion
sum(diag(table.rf)/sum(table.rf))

#predict
rf.pred=predict(iris.rf,newdata=test_data)
table.test=table(Species=test_data$Species,Predicted=rf.pred)
table.test
sum(diag(table.test)/sum(table.test))
rf.train = predict(iris.rf,newdata=train_data)
table.train=table(Species=train_data$Species,Predicted=rf.train)
table.train
sum(diag(table.train)/sum(table.train))

#svm#
library("e1071")
svm_model = svm(Species ~ ., data=train_data)
svm.pred = predict(svm_model, newdata = test_data)
table.test=table(Species=test_data$Species,Predicted=svm.pred)
table.test
sum(diag(table.test)/sum(table.test))
svm.train = predict(svm_model, newdata = train_data)
table.train = table(Species=train_data$Species,Predicted=svm.train)
table.train
sum(diag(table.train)/sum(table.train))

```






