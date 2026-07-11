# Part 1 基本了解

### 1\.注释

代码中使用‘\#’，作注释符号

```Python
#注释符号
```

### 2\.换行

用回车键即可

# Part 2 数据存储与运算

### 1、字面量与变量

①字面量

![image\.png](../assets/python/image%205.png)

②变量

python中一个变量可以存储不同类型的值，但一般不这样做。

③标识符

- 是程序员在代码中为变量、函数、类等元素所起的名字。

- 命名尽量update\_time

python内的关键字主要True,False,None

![image\.png](../assets/python/image%202.png)

### 2、常见数据类型

![image\.png](../assets/python/image%203.png)

通过type\(XX\)查看数据的类型

```Python
print(type("hello"))
```

通过isinstance\(\)检查数据是否属于指定的类型，返回一个bool值，语法isinstance\(数据，类型\)

```Python
num = 6
print(isinstance(num,int))
```

##### \(1\)字符串

1. 定义

```Python
#双引号定义
sl = "Hello"
#单引号定义
s2 = 'Python'
#三引号定义(多行字符串）
s3 = """
尊敬的客户：
感谢您选择我们公司的产品。
我们讲会为您竭诚的服务。
祝好~
"""

#若原居中含'，则使用转义字符“\”
s4 = "It\'s veliy"#（内单外双，内双外单）
```

2. 拼接

    1. 使用\+号——\>对象为字符串

![image\.png](../assets/python/image%204.png)

    2. 字符串格式化——使用%占位符，即%s

![image\.png](../assets/python/image.png)

    3. 最推荐！！！！——f字符串格式化

![image\.png](../assets/python/image%207.png)

### 3、输入与输出

通过input语句获取输入（所有键盘录入的都是字符串类型）

通过print语句输出

```Python
#input(..）)
name=input（"请输入您的姓名："）
print(f"欢迎您，{name}"）
age=input("请输入您的年龄："）
print(f"您今年{age}岁"）
#强制转换
age_new=int(age)+18
```

### 4、运算符

其余与C\+\+几乎相同除

```Python
#幂指数
10**3 #10的三次方

#逻辑运算符
and #&&
or #||
not #！

```

# Part 3 数据的逻辑处理

### if

```Python
a = 10
b = 20 
#if内按照缩进归属代码块
if a<b:
    print("Yes")
elif a==b:
    print("Oh yeah")
else:
    print("No")
```

### match类似C\+\+中的switch\_case语句

```Python
day=input("请输入星期几(1-7):"）
match day:
    case "1":
        print("周一：工作会议日"）
    case "2":
        print（"周二：学习培训日"）
    case "3":
        print（"周三：项目开发日"）
    case "4":
        print（"周四：代码审查日"）
    case "5"：
        print("周五：总结规划日"）
    case "6" | "7"：#其中的|表示或的关系，匹配多个模式中的任意一个————或的意思
        print("周末：休息放松")
    case _:#类似default,且不需要break
        print（"输入错误")
```

### while

```Python
#.while语法结构
while 条件表达式：
    循环体语句1
    循环体语句2
    ..
else: #可有可无
```

### for

```Python
for 元素 in 待处理数据集:
    循环体
else: #可有可无

#定义要遍历的字符串，按照下标访问——用的少
msg = "Hello-Python"
#遍历字符串，并处理
for i in msg:
    print(i)
else:
    print（"循环结束"）
```

# Part 4 数据存储容器

### list

- 列表是数据容器中的一类，是一次性可以存储多个数据（元素）的。

- 定义：列表名称 ＝ \[元素1，元素2，元素3，元素4，元素5\.\.\.\]

s = \[54，15，75，108，23，78，75\]

特点:

- 可以存储不同类型的元素——\>s1 = \[54, 22, "sterr"\]

- 元素有序、可以重复、元素可以修改

- 通过下标操作

##### （1）切片

![image\.png](../assets/python/image%206.png)

##### （2）常见方法

![image\.png](../assets/python/image%201.png)



### str

### tuple

### set

### dict

# Part 5 函数

# Part 6 面向对象基础



