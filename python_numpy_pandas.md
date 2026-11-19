# Python、NumPy 与 Pandas 学习笔记

> 面向 PyTorch、Transformer 与 LLM 学习的基础笔记。重点掌握标有“必学”的内容；低优先级内容保留作查阅，不要求记忆。

## 学习建议

- **Python：**尽量完整掌握，并通过小项目反复使用。
- **NumPy：**重点掌握数组、shape、索引、变形、广播、聚合和矩阵乘法。
- **Pandas：**掌握数据读取、筛选、缺失值、分组、合并和导出即可。
- 学习顺序建议：Python → NumPy → Pandas → PyTorch。


## 第一部分：Python

### 第二章

#### 2.3 字符串（Strings）

##### 一、基本概念

* 字符串：由字符组成的序列
* 表示方式：

  * 单引号：'hello'
  * 双引号："hello"
* 可以灵活嵌套引号

```python
"This is a string."
'This is also a string.'

'I told my friend, "Python is my favorite language!"'
"The language 'Python' is named after Monty Python."
"One of Python's strengths is its diverse and supportive community."
```



##### 二、修改字符串大小写

* `title()`：首字母大写
* `upper()`：全部大写
* `lower()`：全部小写

```python
name = "ada lovelace"

print(name.title())   # Ada Lovelace
print(name.upper())   # ADA LOVELACE
print(name.lower())   # ada lovelace
```



##### 三、在字符串中使用变量

```python
first_name = "ada"
last_name = "lovelace"

full_name = f"{first_name} {last_name}"
print(full_name)   # ada lovelace

print(f"Hello, {full_name.title()}!")   # Hello, Ada Lovelace!
```



##### 四、使用制表符和换行符

* `\t`：制表符
* `\n`：换行符

```python
print("Python")   # Python
print("\tPython")   #     Python

print("Languages:\nPython\nC\nJava")
# Languages:
# Python
# C
# Java

print("Languages:\n\tPython\n\tC\n\tJava")
# Languages:
#     Python
#     C
#     Java
```



##### 五、删除空白

* `rstrip()`：删除右侧空白
* `lstrip()`：删除左侧空白
* `strip()`：删除两侧空白

```python
favorite_language = 'python '

print(favorite_language.rstrip())   # python

favorite_language = ' python '
print(favorite_language.lstrip())   # python 
print(favorite_language.strip())    # python
```



##### 六、避免语法错误

```python
# message = 'One of Python's strengths'

message = "One of Python\'s strengths"
```

#### 2.4 数（Numbers）

##### 一、整数

* 可以进行基本运算：`+`、`-`、`*`、`/`
* 乘方使用 `**`
* 支持运算优先级，可用括号改变顺序 

```python
print(2 + 3)   # 5
print(3 - 2)   # 1
print(2 * 3)   # 6
print(3 / 2)   # 1.5

print(3 ** 2)   # 9
print(3 ** 3)   # 27
print(10 ** 6)  # 1000000

print(2 + 3 * 4)     # 14
print((2 + 3) * 4)   # 20
```



##### 二、浮点数

* 所有带小数点的数都是浮点数
* 可以进行基本运算
* 可能出现精度问题（小数位不精确） 

```python
print(0.1 + 0.1)   # 0.2
print(0.2 + 0.2)   # 0.4
print(2 * 0.1)     # 0.2
print(2 * 0.2)     # 0.4

print(0.2 + 0.1)   # 0.30000000000000004
print(3 * 0.1)     # 0.30000000000000004
```



##### 三、整数和浮点数

* 任意两个数相除，结果都是浮点数 

```python
print(4 / 2)   # 2.0
```



##### 四、数字中的下划线

* 可以使用 `_` 分隔数字，提高可读性

```python
universe_age = 14_000_000_000
print(universe_age)   # 14000000000
```



##### 五、同时给多个变量赋值

```python
x, y, z = 0, 0, 0
print(x, y, z)   # 0 0 0
```



##### 六、常量

* Python 中没有真正的常量
* 通常用全大写变量名表示

```python
MAX_CONNECTIONS = 5000
print(MAX_CONNECTIONS)   # 5000
```

#### 第三章 列表简介（Lists）

##### 一、列表是什么

* 列表由一系列按特定顺序排列的元素组成
* 用方括号 `[]` 表示
* 元素之间用逗号分隔

```python
bicycles = ['trek', 'cannondale', 'redline', 'specialized']
print(bicycles)   # ['trek', 'cannondale', 'redline', 'specialized']
```



##### 二、访问列表元素

* 使用索引访问元素
* 索引从 `0` 开始

```python
bicycles = ['trek', 'cannondale', 'redline', 'specialized']

print(bicycles[0])   # trek
print(bicycles[1])   # cannondale
print(bicycles[3])   # specialized
```

* 使用 `title()` 格式化输出

```python
print(bicycles[0].title())   # Trek
```



##### 三、索引从0而不是1开始

* Python 索引从 `0` 开始
* 最后一个元素索引为 `len(list)-1`

```python
bicycles = ['trek', 'cannondale', 'redline', 'specialized']

print(bicycles[0])   # 第一个元素
print(bicycles[3])   # 第四个元素
```



##### 四、使用负数索引

* `-1` 表示最后一个元素
* `-2` 表示倒数第二个

```python
bicycles = ['trek', 'cannondale', 'redline', 'specialized']

print(bicycles[-1])   # specialized
print(bicycles[-2])   # redline
```



##### 五、使用列表中的各个值

```python
bicycles = ['trek', 'cannondale', 'redline']

message = f"My first bicycle was a {bicycles[0].title()}."
print(message)   # My first bicycle was a Trek.
```



##### 六、修改列表元素

* 直接通过索引赋值

```python
motorcycles = ['honda', 'yamaha', 'suzuki']

motorcycles[0] = 'ducati'
print(motorcycles)   # ['ducati', 'yamaha', 'suzuki']
```



##### 七、在列表中添加元素

###### 1. 使用 `append()`

* 在末尾添加

```python
motorcycles = ['honda', 'yamaha', 'suzuki']

motorcycles.append('ducati')
print(motorcycles)   # ['honda', 'yamaha', 'suzuki', 'ducati']
```



###### 2. 使用 `insert()`

* 在指定位置插入

```python
motorcycles = ['honda', 'yamaha', 'suzuki']

motorcycles.insert(0, 'ducati')
print(motorcycles)   # ['ducati', 'honda', 'yamaha', 'suzuki']
```



##### 八、从列表中删除元素

###### 1. 使用 `del`

* 删除指定位置

```python
motorcycles = ['honda', 'yamaha', 'suzuki']

del motorcycles[0]
print(motorcycles)   # ['yamaha', 'suzuki']
```



###### 2. 使用 `pop()`

* 删除并返回元素

```python
motorcycles = ['honda', 'yamaha', 'suzuki']

popped = motorcycles.pop()
print(motorcycles)   # ['honda', 'yamaha']
print(popped)        # suzuki
```

* 指定索引

```python
motorcycles = ['honda', 'yamaha', 'suzuki']

first = motorcycles.pop(0)
print(first)   # honda
```



###### 3. 使用 `remove()`

* 根据值删除

```python
motorcycles = ['honda', 'yamaha', 'suzuki', 'ducati']

motorcycles.remove('ducati') #只删除第一个ducati
print(motorcycles)   # ['honda', 'yamaha', 'suzuki']

motorcycles = [x for x in motorcycles if x != 'ducati']
```

* 可结合变量

```python
motorcycles = ['honda', 'yamaha', 'ducati']

too_expensive = 'ducati'
motorcycles.remove(too_expensive)

print(motorcycles)   # ['honda', 'yamaha']
print(f"{too_expensive.title()} is too expensive for me.")  
# Ducati is too expensive for me.
```



##### 九、使用方法 `sort()` 对列表永久排序

* 默认按字母顺序

```python
cars = ['bmw', 'audi', 'toyota', 'subaru']

cars.sort()
print(cars)   # ['audi', 'bmw', 'subaru', 'toyota']
```

* 反向排序

```python
cars.sort(reverse=True)
print(cars)   # ['toyota', 'subaru', 'bmw', 'audi']
```



##### 十、使用函数 `sorted()` 临时排序(非原地)

* 不改变原列表

```python
cars = ['bmw', 'audi', 'toyota', 'subaru']

print(sorted(cars))   # ['audi', 'bmw', 'subaru', 'toyota']
print(cars)           # ['bmw', 'audi', 'toyota', 'subaru']
```



##### 十一、倒着打印列表

* 使用 `reverse()`

```python
cars = ['bmw', 'audi', 'toyota', 'subaru']

cars.reverse()
print(cars)   # ['subaru', 'toyota', 'audi', 'bmw']
```



##### 十二、确定列表长度

* 使用 `len()`

```python
cars = ['bmw', 'audi', 'toyota', 'subaru']

print(len(cars))   # 4
```



##### 十三、避免索引错误

* 索引超出范围会报错

```python
cars = ['bmw', 'audi', 'toyota']

# print(cars[3])  # IndexError
```

* 使用 `-1` 安全访问最后一个元素

```python
print(cars[-1])   # toyota
```

#### 第四章 操作列表（Working with Lists）

##### 一、遍历整个列表

* 使用 `for` 循环依次访问列表中的每个元素

```python
magicians = ['alice', 'david', 'carolina']

for magician in magicians:
    print(magician)   # 依次输出每个元素
```

* 在循环中执行更多操作

```python
magicians = ['alice', 'david', 'carolina']

for magician in magicians:
    print(f"{magician.title()}, that was a great trick!")
    # Alice, that was a great trick!
    # David, that was a great trick!
    # Carolina, that was a great trick!

    print(f"I can't wait to see your next trick, {magician.title()}.\n")
```

* 循环结束后的代码

```python 4n8d2k"
print("Thank you, everyone. That was a great magic show!")
```



##### 二、避免缩进错误

* Python 使用缩进表示代码块
* 常见错误：

  * 忘记缩进
  * 不必要的缩进
  * 缩进不一致

```python
for magician in magicians:
print(magician)   # 错误：缺少缩进
```



##### 三、创建数值列表

###### 1. 使用 `range()`

* 生成一系列数字

```python
for value in range(1, 5):
    print(value)
# 1
# 2
# 3
# 4
```

* 转换为列表

```python 
numbers = list(range(1, 6))
print(numbers)   # [1, 2, 3, 4, 5]
```



###### 2. 指定步长

```python 
even_numbers = list(range(2, 11, 2))
print(even_numbers)   # [2, 4, 6, 8, 10]
```



###### 3. 创建平方数列表

```python 
squares = []

for value in range(1, 11):
    squares.append(value**2)

print(squares)
# [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```



##### 四、对数字列表执行简单统计计算

* `min()`：最小值
* `max()`：最大值
* `sum()`：求和

```python 
digits = [1, 2, 3, 4, 5]

print(min(digits))   # 1
print(max(digits))   # 5
print(sum(digits))   # 15
```



##### 五、列表解析（List Comprehension）

* 用一行代码生成列表

```python 
squares = [value**2 for value in range(1, 11)]
print(squares)
# [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```



##### 六、使用列表的一部分（切片）

###### 1. 基本切片

* 语法：`list[start:end]`（左闭右开）

```python 
players = ['charles', 'martina', 'michael', 'florence', 'eli']

print(players[0:3])   # ['charles', 'martina', 'michael']
print(players[1:4])   # ['martina', 'michael', 'florence']
```



###### 2. 省略索引

```python 
print(players[:4])   # 前四个元素
print(players[2:])   # 从第三个到末尾
```



###### 3. 使用负索引

```python 
print(players[-3:])   # 最后三个元素
```



##### 七、遍历切片

```python
players = ['charles', 'martina', 'michael', 'florence', 'eli']

for player in players[:3]:
    print(player.title())
# Charles
# Martina
# Michael
```



##### 八、复制列表

* 使用切片复制

```python 
my_foods = ['pizza', 'falafel', 'carrot cake']
friend_foods = my_foods[:]

print(my_foods)      # ['pizza', 'falafel', 'carrot cake']
print(friend_foods)  # ['pizza', 'falafel', 'carrot cake']
```

* 修改后互不影响

```python
# 如果是my_foods = friend_foods就相互影响了
my_foods.append('cannoli')
friend_foods.append('ice cream')

print(my_foods)
print(friend_foods)
```



##### 九、元组（Tuple）

* 不可变列表
* 使用 `()` 表示

```python
dimensions = (200, 50)
print(dimensions)   # (200, 50)
```

* 访问元素

```python 
print(dimensions[0])   # 200
print(dimensions[1])   # 50
```



##### 十、遍历元组

```python
for dimension in dimensions:
    print(dimension)
```



##### 十一、修改元组变量

* 元组本身不可变，但变量可以重新赋值

```python
dimensions = (200, 50)
print(dimensions)

dimensions = (400, 100)
print(dimensions)
```

#### 第五章 if语句

##### 一、简单 if 语句

* `if` 用于在满足条件时执行代码
* 条件为 `True` 才会执行

```python
age = 18

if age >= 18:
    print("You are old enough to vote!")   # You are old enough to vote!
    print("Have you registered to vote?") # Have you registered to vote?
```



##### 二、条件测试

###### 1. 相等判断

* 使用 `==` 判断是否相等

```python 
car = 'bmw'
print(car == 'bmw')   # True
```

###### 2. 不等判断

* 使用 `!=`

```python 
car = 'audi'
print(car != 'bmw')   # True
```

###### 3. 比较大小

* `>`、`<`、`>=`、`<=`

```python 
age = 18
print(age >= 18)   # True
```



##### 三、检查多个条件

###### 1. 使用 `and`

* 两个条件都为真

```python 
age = 20
print(age > 18 and age < 30)   # True
```

###### 2. 使用 `or`

* 任一条件为真

```python 
age = 17
print(age < 18 or age > 65)   # True
```



##### 四、检查特定值是否在列表中

###### 1. 使用 `in`

```python 
requested = ['mushrooms', 'onions', 'pineapple']
print('mushrooms' in requested)   # True
```

###### 2. 使用 `not in`

```python 
print('pepperoni' not in requested)   # True
```



##### 五、布尔表达式

* 条件测试结果为 `True` 或 `False`

```python 
game_active = True
can_edit = False
```



##### 六、if-else 语句

* 条件成立执行 `if`
* 否则执行 `else`

```python 
age = 17

if age >= 18:
    print("You can vote.")
else:
    print("Sorry, you are too young to vote.")  
    # Sorry, you are too young to vote.
```



##### 七、if-elif-else 结构

* 多条件判断

```python
age = 12

if age < 4:
    print("Your admission cost is $0.")
elif age < 18:
    print("Your admission cost is $10.")  
    # Your admission cost is $10.
else:
    print("Your admission cost is $15.")
```



##### 八、使用多个 elif

```python
age = 65

if age < 4:
    price = 0
elif age < 18:
    price = 10
elif age < 65:
    price = 15
else:
    price = 5

print(f"Your cost is {price}.")   # Your cost is 5.
```



##### 九、省略 else

* 使用多个 `elif`

```python 
age = 30

if age < 4:
    price = 0
elif age < 18:
    price = 10
elif age < 65:
    price = 15

print(price)   # 15
```



##### 十、测试多个条件（多个 if）

* 每个条件都会独立判断

```python 
requested = ['mushrooms', 'extra cheese']

if 'mushrooms' in requested:
    print("Adding mushrooms")   # Adding mushrooms

if 'pepperoni' in requested:
    print("Adding pepperoni")

if 'extra cheese' in requested:
    print("Adding extra cheese")   # Adding extra cheese
```



##### 十一、使用 if 处理列表

###### 1. 特殊值处理

```python 
requested = ['mushrooms', 'green peppers', 'extra cheese']

for topping in requested:
    if topping == 'green peppers':
        print("Sorry, we are out of green peppers.")  
        # Sorry, we are out of green peppers.
    else:
        print(f"Adding {topping}.")
```



###### 2. 检查列表是否为空

```python 
requested = []

if requested:
    for topping in requested:
        print(topping)
else:
    print("Are you sure you want a plain pizza?")  
    # Are you sure you want a plain pizza?
```



###### 3. 使用多个列表

```python 
available = ['mushrooms', 'olives', 'green peppers', 'pepperoni']
requested = ['mushrooms', 'french fries', 'extra cheese']

for topping in requested:
    if topping in available:
        print(f"Adding {topping}.")  
        # Adding mushrooms.
    else:
        print(f"Sorry, we don't have {topping}.")  
        # Sorry, we don't have french fries.
```

#### 第六章 字典（Dictionaries）

##### 一、一个简单的字典

* 字典是键值对（key-value）的集合
* 用 `{}` 表示

```python
alien_0 = {'color': 'green', 'points': 5}

print(alien_0['color'])   # green
print(alien_0['points'])  # 5
```



##### 二、添加键值对

* 通过 `字典[key] = value` 添加

```python
alien_0 = {'color': 'green', 'points': 5}

alien_0['x_position'] = 0
alien_0['y_position'] = 25

print(alien_0)   
# {'color': 'green', 'points': 5, 'x_position': 0, 'y_position': 25}
```



##### 三、先创建空字典

```python 
alien_0 = {}

alien_0['color'] = 'green'
alien_0['points'] = 5

print(alien_0)   # {'color': 'green', 'points': 5}
```



##### 四、修改字典中的值

```python 
alien_0 = {'color': 'green'}

alien_0['color'] = 'yellow'
print(alien_0)   # {'color': 'yellow'}
```



##### 五、根据条件修改值

```python 
alien_0 = {'x_position': 0, 'speed': 'medium'}

if alien_0['speed'] == 'slow':
    x_increment = 1
elif alien_0['speed'] == 'medium':
    x_increment = 2
else:
    x_increment = 3

alien_0['x_position'] += x_increment
print(alien_0['x_position'])   # 2
```



##### 六、删除键值对

* 使用 `del`

```python 
alien_0 = {'color': 'green', 'points': 5}

del alien_0['points']
print(alien_0)   # {'color': 'green'}
```



##### 七、由类似对象组成的字典

```python 
favorite_languages = {
    'jen': 'python',
    'sarah': 'c',
    'edward': 'rust',
}

print(favorite_languages['sarah'])   # c
```



##### 八、使用 `get()` 访问值

* 避免键不存在时报错

```python
alien_0 = {'color': 'green'}

print(alien_0.get('points'))        # None
print(alien_0.get('points', 0))     # 0
```



##### 九、遍历字典

###### 1. 遍历所有键值对 `items()`

```python 
user = {'username': 'efermi', 'first': 'enrico'}

for key, value in user.items():
    print(f"{key}: {value}")
# username: efermi
# first: enrico
```



###### 2. 遍历所有键 `keys()`

```python
favorite_languages = {'jen': 'python', 'sarah': 'c'}

for name in favorite_languages.keys():
    print(name)
# jen
# sarah
```



###### 3. 遍历所有值 `values()`

```python
favorite_languages = {'jen': 'python', 'sarah': 'c'}

for language in favorite_languages.values():
    print(language)
# python
# c
```



##### 十、按顺序遍历字典

* 使用 `sorted()`

```python
favorite_languages = {'jen': 'python', 'sarah': 'c'}

for name in sorted(favorite_languages.keys()):
    print(name)
# jen
# sarah
```



##### 十一、去重遍历值

* 使用 `set()`

```python
favorite_languages = {
    'jen': 'python',
    'sarah': 'c',
    'edward': 'python',
}

for language in set(favorite_languages.values()):
    print(language)
# python
# c
```



##### 十二、嵌套

###### 1. 字典列表

```python 
alien_0 = {'color': 'green'}
alien_1 = {'color': 'yellow'}

aliens = [alien_0, alien_1]

print(aliens)
# [{'color': 'green'}, {'color': 'yellow'}]
```



###### 2. 在列表中生成字典

```python 
aliens = []

for i in range(3):
    alien = {'color': 'green'}
    aliens.append(alien)

print(aliens)
# [{'color': 'green'}, {'color': 'green'}, {'color': 'green'}]
```



###### 3. 字典中存列表

```python
pizza = {
    'toppings': ['mushrooms', 'extra cheese']
}

for topping in pizza['toppings']:
    print(topping)
# mushrooms
# extra cheese
```



###### 4. 字典中存字典

```python
users = {
    'aeinstein': {
        'first': 'albert',
        'last': 'einstein',
    }
}

for username, info in users.items():
    print(username)
    print(info['first'])
# aeinstein
# albert
```

#### 第七章 用户输入和 while循环（User Input and While Loops）

##### 一、函数 `input()`

* 用于获取用户输入
* 返回值是字符串

```python
message = input("Tell me something: ")
print(message)
```



##### 二、编写清晰的提示

* 提示信息应说明输入要求

```python
name = input("Please enter your name: ")
print(f"Hello, {name}!")
```



##### 三、使用 `int()` 获取数值输入

* `input()` 默认返回字符串，需要转换

```python
age = input("How old are you? ")
age = int(age)

print(age >= 18)   # True/False（取决于输入）
```



##### 四、求模运算 `%`

* 判断奇偶

```python
number = int(input("Enter a number: "))

if number % 2 == 0:
    print("Even")   # 偶数
else:
    print("Odd")    # 奇数
```



##### 五、while 循环

* 条件为 `True` 时持续执行

```python
current = 1

while current <= 5:
    print(current)
    current += 1
# 1 2 3 4 5
```



##### 六、让用户选择何时退出

###### 1. 使用标志（flag）

```python
active = True

while active:
    message = input("Enter 'quit' to exit: ")
    if message == 'quit':
        active = False
    else:
        print(message)
```



###### 2. 使用 `break`

```python
while True:
    message = input("Enter 'quit': ")
    if message == 'quit':
        break
    print(message)
```



##### 七、使用 `continue`

* 跳过当前循环

```python
current = 0

while current < 5:
    current += 1
    if current == 3:
        continue
    print(current)
# 1 2 4 5
```



##### 八、避免无限循环

* 确保循环条件最终变为 `False`

```python
# 错误示例（无限循环）
# while True:
#     pass
```



##### 九、使用 while 处理列表

###### 1. 在列表间移动元素

```python
unconfirmed = ['alice', 'brian']
confirmed = []

while unconfirmed:
    current = unconfirmed.pop()
    confirmed.append(current)

print(confirmed)   # ['brian', 'alice']
```



###### 2. 删除特定值

```python
pets = ['dog', 'cat', 'dog', 'fish']

while 'dog' in pets:
    pets.remove('dog')

print(pets)   # ['cat', 'fish']
```



##### 十、使用用户输入填充字典

```python
responses = {}

active = True

while active:
    name = input("Name: ")
    response = input("Answer: ")

    responses[name] = response

    repeat = input("Continue? (yes/no): ")
    if repeat == 'no':
        active = False

print(responses)
```

#### 第八章 函数（Functions）

##### 一、定义函数

* 使用 `def` 定义函数
* 函数名后加括号

```python
def greet_user():
    print("Hello!")   # Hello!

greet_user()
```



##### 二、向函数传递信息（参数）

* 参数是函数定义中的变量

```python
def greet_user(username):
    print(f"Hello, {username.title()}!")

greet_user('jesse')   # Hello, Jesse!
```



##### 三、实参和形参

* 形参：函数定义中的变量
* 实参：调用时传入的值

```python
def greet_user(name):   # name 是形参
    print(name)

greet_user('Tom')   # 'Tom' 是实参
```



##### 四、位置实参

* 按顺序传递

```python
def describe_pet(animal_type, pet_name):
    print(f"I have a {animal_type}.")
    print(f"My {animal_type}'s name is {pet_name}.")

describe_pet('dog', 'willie')
# I have a dog.
# My dog's name is willie.
```



##### 五、关键字实参

* 指定参数名

```python
describe_pet(animal_type='hamster', pet_name='harry')
```



##### 六、默认值

* 为参数提供默认值

```python
def describe_pet(pet_name, animal_type='dog'):
    print(f"{animal_type} named {pet_name}")

describe_pet('willie')   # dog named willie
```



##### 七、等效函数调用

```python
describe_pet('harry')
describe_pet(pet_name='harry')
describe_pet(pet_name='harry', animal_type='hamster')
```



##### 八、返回值 `return`

* 函数可以返回结果

```python
def get_full_name(first, last):
    return f"{first} {last}"

print(get_full_name('ada', 'lovelace'))   # ada lovelace
```



##### 九、返回字典

```python
def build_person(first, last):
    return {'first': first, 'last': last}

person = build_person('ada', 'lovelace')
print(person)   # {'first': 'ada', 'last': 'lovelace'}
```



##### 十、让实参变为可选

```python
def build_person(first, last, age=None):
    person = {'first': first, 'last': last}
    if age:
        person['age'] = age
    return person

print(build_person('ada', 'lovelace'))   
# {'first': 'ada', 'last': 'lovelace'}
```



##### 十一、结合 while 使用函数

```python
def get_formatted_name(first, last):
    return f"{first} {last}"

while True:
    print("Enter q to quit")
    f = input("First: ")
    if f == 'q':
        break
    l = input("Last: ")
    if l == 'q':
        break

    print(get_formatted_name(f, l))
```



##### 十二、传递列表

```python
def greet_users(names):
    for name in names:
        print(f"Hello, {name}")

greet_users(['tom', 'anna'])
# Hello, tom
# Hello, anna
```



##### 十三、在函数中修改列表

```python
def print_models(unprinted, completed):
    while unprinted:
        current = unprinted.pop()
        completed.append(current)

unprinted = ['phone', 'case']
completed = []

print_models(unprinted, completed)
print(completed)   # ['case', 'phone']
```



##### 十四、禁止函数修改列表（切片）

```python
print_models(unprinted[:], completed)
```



##### 十五、传递任意数量的实参

* 使用 `*args`

```python
def make_pizza(*toppings):
    print(toppings)

make_pizza('pepperoni')   # ('pepperoni',)
make_pizza('mushrooms', 'onions')
# ('mushrooms', 'onions')
```



##### 十六、使用任意数量的关键字实参

* 使用 `**kwargs`

```python
def build_profile(first, last, **info):
    profile = {'first': first, 'last': last}
    profile.update(info)
    return profile

print(build_profile('albert', 'einstein', location='princeton'))
# {'first': 'albert', 'last': 'einstein', 'location': 'princeton'}
```



##### 十七、将函数存储在模块中

###### 1. 创建模块

```python
# module.py
def greet():
    print("Hello")
```

###### 2. 导入模块

```python
import module

module.greet()   # Hello
```



##### 十八、导入特定函数

```python
from module import greet

greet()   # Hello
```



##### 十九、使用别名

```python
import module as m
from module import greet as g

g()   # Hello
```



##### 二十、导入所有函数

```python
from module import *
```

#### 第九章 类（Classes）

##### 一、创建和使用类

###### 1. 创建类

* 使用 `class` 定义
* `__init__()` 是构造函数

```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age
```



###### 2. 创建实例

```python
my_dog = Dog('willie', 6)

print(my_dog.name)   # willie
print(my_dog.age)    # 6
```



###### 3. 调用方法

```python
class Dog:
    def __init__(self, name):
        self.name = name

    def sit(self):
        print(f"{self.name} is sitting.")  

dog = Dog('willie')
dog.sit()   # willie is sitting.
```



##### 二、给属性指定默认值

```python
class Car:
    def __init__(self, make, model):
        self.make = make
        self.model = model
        self.odometer = 0

car = Car('audi', 'a4')
print(car.odometer)   # 0
```



##### 三、修改属性的值

###### 1. 直接修改

```python
car.odometer = 100
print(car.odometer)   # 100
```

###### 2. 通过方法修改

```python
class Car:
    def __init__(self):
        self.odometer = 0

    def update_odometer(self, mileage):
        self.odometer = mileage

car = Car()
car.update_odometer(200)
print(car.odometer)   # 200
```

###### 3. 递增属性

```python
class Car:
    def __init__(self):
        self.odometer = 0

    def increment(self, miles):
        self.odometer += miles

car = Car()
car.increment(50)
print(car.odometer)   # 50
```



##### 四、继承

###### 1. 创建子类

```python 
class ElectricCar(Car):
    def __init__(self, make, model):
        super().__init__(make, model)
```



###### 2. 添加属性

```python
class ElectricCar(Car):
    def __init__(self, make, model):
        super().__init__(make, model)
        self.battery_size = 75

car = ElectricCar('tesla', 'model s')
print(car.battery_size)   # 75
```



###### 3. 添加方法

```python
class ElectricCar(Car):
    def describe_battery(self):
        print("This car has a battery.")

car = ElectricCar('tesla', 'model s')
car.describe_battery()   # This car has a battery.
```



###### 4. 重写父类方法

```python
class ElectricCar(Car):
    def fill_gas_tank(self):
        print("No gas needed.")   # No gas needed.
```



##### 五、将实例用作属性

```python
class Battery:
    def __init__(self):
        self.size = 75

class ElectricCar(Car):
    def __init__(self, make, model):
        super().__init__(make, model)
        self.battery = Battery()

car = ElectricCar('tesla', 'model s')
print(car.battery.size)   # 75
```



##### 六、导入类

###### 1. 导入单个类

```python
from car import Car
```

###### 2. 导入多个类

```python
from car import Car, ElectricCar
```

###### 3. 导入整个模块

```python
import car
```



##### 七、Python标准库

* Python 自带模块可直接使用

```python
from random import randint

print(randint(1, 10))   # 随机数
```

##### 特殊命名
```python
class Car():
    def __init__(self,name="RB22"):
        self.__name = name
    
    def speed(self):
        print(275)

class RedBull(Car):
    def speed(self):
        print(250)

RB22 = RedBull()
print(RB22._Car__name) # 不能用RB22.__name
```

#### 第十章 文件和异常（Files and Exceptions）

##### 一、从文件中读取数据

###### 1. 读取整个文件

* 使用 `open()` 打开文件
* 使用 `read()` 读取全部内容

```python
with open('pi_digits.txt') as file_object:
    contents = file_object.read()
    print(contents)   # 文件全部内容
```



###### 2. 逐行读取

```python 
with open('pi_digits.txt') as file_object:
    for line in file_object:
        print(line.rstrip())   # 去除换行符
```



###### 3. 创建包含各行的列表

```python
with open('pi_digits.txt') as file_object:
    lines = file_object.readlines()

for line in lines:
    print(line.strip())
```



##### 二、使用文件内容

```python
with open('pi_digits.txt') as file_object:
    lines = file_object.readlines()

pi_string = ''

for line in lines:
    pi_string += line.strip()

print(pi_string)   # 拼接后的字符串
```



##### 三、写入文件

###### 1. 写入（覆盖模式 `'w'`）

```python
with open('programming.txt', 'w') as file_object:
    file_object.write("I love programming.")
```



###### 2. 写入多行

```python
with open('programming.txt', 'w') as file_object:
    file_object.write("I love programming.\n")
    file_object.write("I love creating new games.\n")
```



###### 3. 追加内容（模式 `'a'`）

```python
with open('programming.txt', 'a') as file_object:
    file_object.write("I also love finding meaning.\n")
```



##### 四、异常处理

###### 1. `try-except`

```python
try:
    print(5 / 0)
except ZeroDivisionError:
    print("You can't divide by zero!")  
    # You can't divide by zero!
```



###### 2. 防止程序崩溃

```python
print("Give me two numbers")

try:
    x = int(input("First: "))
    y = int(input("Second: "))
    print(x / y)
except ZeroDivisionError:
    print("Cannot divide by zero")
```



###### 3. 处理多个异常

```python
try:
    number = int("abc")
except ValueError:
    print("Invalid number")   # Invalid number
```



##### 五、使用 `else`

* `try` 成功执行时运行

```python
try:
    x = int(input("First: "))
    y = int(input("Second: "))
except ValueError:
    print("Invalid input")
else:
    print(x / y)
```



##### 六、处理文件异常

```python
try:
    with open('missing.txt') as f:
        content = f.read()
except FileNotFoundError:
    print("File not found")   # File not found
```



##### 七、分析文本

```python
try:
    with open('text.txt') as f:
        contents = f.read()
except FileNotFoundError:
    print("Missing file")
else:
    words = contents.split()
    print(len(words))   # 单词数
```



##### 八、静默失败 `pass`

* 出现异常时什么也不做

```python
try:
    with open('missing.txt') as f:
        content = f.read()
except FileNotFoundError:
    pass
```



##### 九、存储数据（JSON）

###### 1. 保存数据

```python
import json

numbers = [1, 2, 3]

with open('numbers.json', 'w') as f:
    json.dump(numbers, f)
```



###### 2. 读取数据

```python
import json

with open('numbers.json') as f:
    numbers = json.load(f)

print(numbers)   # [1, 2, 3]
```



###### 3. 保存用户数据

```python
import json

username = input("What is your name? ")

with open('username.json', 'w') as f:
    json.dump(username, f)
```



###### 4. 读取用户数据

```python
import json

with open('username.json') as f:
    username = json.load(f)

print(f"Welcome back, {username}!")
```

#### 补充：包（Packages）

##### 一、什么是包

* 包是“包含多个模块的目录”
* 模块是一个 `.py` 文件
* 用于组织大型项目代码

结构理解：

```text
包（目录）
 ├── 模块1.py
 ├── 模块2.py
 └── __init__.py
```


##### 二、包的基本结构

```text
project/
│
├── main.py
└── utils/
    ├── __init__.py
    ├── math_tools.py
    └── string_tools.py
```



##### 三、创建模块

```python id="p2k9vx"
# math_tools.py
def add(a, b):
    return a + b
```

```python id="m7x4qs"
# string_tools.py
def shout(text):
    return text.upper()
```



##### 四、导入包中的模块

###### 1. 导入模块

```python id="t9v3rm"
import utils.math_tools

print(utils.math_tools.add(2, 3))   # 5
```



###### 2. 导入函数

```python id="k3p8qn"
from utils.math_tools import add

print(add(2, 3))   # 5
```



###### 3. 使用别名

```python id="x6q1mw"
import utils.math_tools as mt

print(mt.add(2, 3))   # 5
```



##### 五、`__init__.py`

* 用于标识“这是一个包”
* 可以为空
* 也可以控制导出内容

```python id="r8m2zc"
# utils/__init__.py
```



##### 六、统一导出（推荐）

```python id="c4x9bt"
# utils/__init__.py
from .math_tools import add
from .string_tools import shout
```

```python id="u7k3pr"
from utils import add, shout

print(add(1, 2))         # 3
print(shout("hello"))    # HELLO
```



##### 七、相对导入

* 使用 `.` 表示当前包

```python id="n5v1qx"
# string_tools.py
from .math_tools import add
```



##### 八、包 vs 模块

* 模块：一个 `.py` 文件
* 包：包含多个模块的目录



##### 九、典型使用场景

* 项目文件较多
* 功能需要分类（工具、数据处理、模型等）



##### 十、完整示例

```text
demo/
│
├── main.py
└── tools/
    ├── __init__.py
    └── calculator.py
```

```python id="p9x2ks"
# calculator.py
def multiply(a, b):
    return a * b
```

```python id="m3v8qn"
# main.py
from tools.calculator import multiply

print(multiply(3, 4))   # 12
```



##### 十一、常见错误

###### 1. 运行路径错误

* 必须在项目根目录运行

```bash
python main.py
```



###### 2. 模块重名

* 避免命名为：

  * `json.py`
  * `random.py`
  * `os.py`



###### 3. 导入失败

* 检查：

  * 目录结构
  * 是否在正确路径运行
  * 是否有 `__init__.py`



##### 十二、核心记忆

```python
import 包.模块
from 包.模块 import 函数
import 包.模块 as 别名
from 包 import 内容
from .模块 import 内容
```

#### 补充：装饰器（Decorators）

##### 一、什么是装饰器

* 装饰器本质是一个函数
* 接收“被装饰函数”作为参数，返回一个新函数
* 常见用途：日志、权限检查、统计耗时、缓存

```python
def decorator(func):
    def wrapper():
        print("函数执行前")
        func()
        print("函数执行后")
    return wrapper
```

##### 二、`@` 语法糖

```python
def decorator(func):
    def wrapper():
        print("函数执行前")
        func()
        print("函数执行后")
    return wrapper

@decorator
def say_hello():
    print("hello")

say_hello()
# 函数执行前
# hello
# 函数执行后
```

* `@decorator` 等价于：`say_hello = decorator(say_hello)`

##### 三、装饰带参数的函数

```python
def log_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"调用函数: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"返回结果: {result}")
        return result
    return wrapper

@log_decorator
def add(a, b):
    return a + b

print(add(3, 5))
```

* 用 `*args` 和 `**kwargs` 能兼容不同参数形式

##### 四、保留原函数信息：`functools.wraps`

* 不用 `wraps` 时，`__name__` 等元信息会变成 `wrapper`
* 使用 `@wraps(func)` 可以保留原函数名称和文档字符串

```python
from functools import wraps

def log_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("开始执行")
        return func(*args, **kwargs)
    return wrapper

@log_decorator
def greet(name):
    """打招呼函数"""
    return f"hello, {name}"

print(greet.__name__)   # greet
print(greet.__doc__)    # 打招呼函数
```

##### 五、带参数的装饰器（装饰器工厂）

```python
from functools import wraps

def repeat(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def hello():
    print("hello")

hello()
```

* 执行顺序：
* `repeat(3)` 先执行，返回 `decorator`
* 再由 `decorator` 接收并包装 `hello`

##### 六、多个装饰器叠加

```python
def deco1(func):
    def wrapper():
        print("deco1 before")
        func()
        print("deco1 after")
    return wrapper

def deco2(func):
    def wrapper():
        print("deco2 before")
        func()
        print("deco2 after")
    return wrapper

@deco1
@deco2
def work():
    print("working")

work()
```

* 等价于：`work = deco1(deco2(work))`
* 调用顺序：从外到内进入，再从内到外退出

##### 七、类装饰器（了解）

* 除了函数，也可以用类实现装饰器
* 核心是实现 `__call__` 方法

```python
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"第 {self.count} 次调用")
        return self.func(*args, **kwargs)

@CountCalls
def greet():
    print("hello")

greet()
greet()
```

##### 八、核心记忆

* 装饰器 = 在不改函数源码前提下增强函数功能
* `@decorator` 是语法糖
* 通用包装函数建议写成 `wrapper(*args, **kwargs)`
* 推荐使用 `functools.wraps` 保留元信息

### 补充：进入 PyTorch 前建议掌握的常用写法

#### 一、`enumerate()`：同时取得索引和值

```python
names = ["Alice", "Bob", "Charlie"]

for i, name in enumerate(names):
    print(i, name)
```

#### 二、`zip()`：并行遍历多个可迭代对象

```python
names = ["Alice", "Bob"]
scores = [85, 92]

for name, score in zip(names, scores):
    print(name, score)
```

#### 三、字典解析与集合解析

```python
squares = {x: x ** 2 for x in range(5)}
unique_squares = {x ** 2 for x in range(-3, 4)}
```

#### 四、`lambda`：简短的匿名函数

```python
students = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
]

students_sorted = sorted(students, key=lambda x: x["score"], reverse=True)
```

`lambda` 适合只使用一次的简单函数；逻辑复杂时应使用 `def`。

#### 五、`pathlib`：更清晰地处理文件路径

```python
from pathlib import Path

data_path = Path("data") / "train.csv"

if data_path.exists():
    text = data_path.read_text(encoding="utf-8")
```

#### 六、程序入口 `if __name__ == "__main__"`

```python
def main():
    print("Program starts here.")


if __name__ == "__main__":
    main()
```

直接运行该文件时，条件成立并执行 `main()`；把该文件作为模块导入时，入口代码不会自动执行。

## 第二部分：NumPy

### 学习重点

以下内容应熟练掌握：

- `ndarray`、`dtype`、`shape`、`ndim`
- 索引、切片和布尔索引
- `reshape`、转置、`squeeze`、`expand_dims`
- broadcasting（广播）
- `concatenate`、`stack`
- `sum`、`mean`、`std`、`max`、`argmax`
- `where`、随机数
- `@` / `matmul`
- copy 与 view

字符串函数、位运算、结构化数据类型、`nditer` 等内容优先级较低，可在实际需要时查阅。

```python
import numpy as np
# axis 表示操作的维度，不应简单记成“行”或“列”
# 对二维数组：axis=0 沿第 0 维操作，聚合后保留各列的结果
# 对二维数组：axis=1 沿第 1 维操作，聚合后保留各行的结果
```

### Ndarray 对象

---

```python
numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
```

| object | 数组或嵌套的数列 |
| --- | --- |
| dtype | 数组元素的数据类型，可选 |
| copy | 对象是否需要复制，可选 |
| order | 创建数组的样式，C为行方向，F为列方向，A为任意方向（默认） |
| subok | 默认返回一个与基类类型一致的数组 |
| ndmin | 指定生成数组的最小维度 |
- 一般都只使用object元素

例如：

```python
#创建一维数组
array_1 = np.array([1,2,3])

#创建二维数组
array_2 = np.array([[1,2],[3,4]])
```

- 常用的还有

```python
a = np.array([1,2],ndmin = 2)
#此时创建的也是二维数组

a = np.array([1,2,3],dtype = complex)
#此时数据类型为复数,但实际上不必如此

a = np.array([1+1j,2+1j,3+2j])
#此时数据类型默认为complex
```
- 类型转换
```python
a = np.arange(1,11,1,dtype = float).reshape(2,5)
a[...,2] = np.nan #int类型不能使用np.nan
# 含 NaN 的浮点数组不能直接安全地转换为整数；应先按任务含义处理缺失值
a = np.nan_to_num(a, nan=0).astype(int)
```
### Ndarray属性

---

| `ndarray.ndim` | 数组的秩（rank），即数组的维度数量或轴的数量，不是矩阵的秩。 |
| --- | --- |
| `ndarray.shape` | 数组的维度，表示数组在每个轴上的大小。对于二维数组（矩阵），表示其行数和列数。 |
| `ndarray.size` | 数组中元素的总个数，等于 `ndarray.shape` 中各个轴上大小的乘积。 |
| `ndarray.dtype` | 数组中元素的数据类型。 |
| `ndarray.itemsize` | 数组中每个元素的大小，以字节为单位。 |
| `ndarray.flags` | 包含有关内存布局的信息，如是否为 C 或 Fortran 连续存储，是否为只读等。 |
| `ndarray.real` | 数组中每个元素的实部（如果元素类型为复数）。 |
| `ndarray.imag` | 数组中每个元素的虚部（如果元素类型为复数）。 |
| `ndarray.data` | 实际存储数组元素的缓冲区，一般通过索引访问元素，不直接使用该属性。 |

ndim,shape,size,real,imag较为常用

### 创建数组

---

- 未初始化数组

```python
numpy.empty(shape, dtype = float, order = 'C')

#创建一个2*3的未初始化数组
np.empty((2,3)) #内容未经初始化，值取决于内存中的残留数据，并非随机数
```

- 零数组

```python
numpy.zeros(shape, dtype = float, order = 'C')

#创建一个2*3的空数组
np.zeros((2,3))
```

- 全一数组

```python
numpy.ones(shape, dtype = None, order = 'C')

#创建一个2*3的空数组
np.ones((2,3))
```

- 对角1数组

```python
np.eye(m,n,k)
#创建m*n矩阵，k为对角线偏移量
```

- 单位矩阵

```python
np.identity(m)
#创建m维单位阵
```

- 随机矩阵

```python
np.random.rand(3,3) #[0,1)均匀分布
np.random.randn(3,3) #标准正态分布
np.random.randint(low,high,size) #[low,high)均匀分布，size为元祖
np.random.choice(mylist,size,replace = True,p = []) #放回取样,必须是一维
np.random.normal(a,b,size) #正态分布，a为均值，b为标准差
```

- 模仿数组

```python
numpy.zeros_like(a, dtype=None, order='K', subok=True, shape=None)
numpy.ones_like(a, dtype=None, order='K', subok=True, shape=None)
#创建与a相同的数组，只不过把元素全部替换为0或者1
```

- 直接构造

```python
#array 深度拷贝
#asarray 浅拷贝 
```

- 二进制生成数组

```python
str_1 = b"abcd"
a = np.frombuffer(str_1,dtype = np.uint8)
```

- 迭代器生成数组

```python
a = (i for i in range(5))
np_1 = np.fromiter(a) #需要指定dtype
```

- 从数组范围创建数组

```python
numpy.arange(start, stop, step, dtype)
np.arange(5) #[0 1 2 3 4]
np.arange(0,10,2) #[0 2 4 6 8]

np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
np.linspace(1,5,5) [1 2 3 4 5]
#endpoint改为False将不包含终点

np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
#等比数列，起始元素base^start,终点元素base^stop,生成num个元素
```

- 扩大数组

```python
a = np.array([1,2,3])
b = np.tile(a,(2,2))
```

- 提取主对角元素

```python
np.diag(a)
```

### Numpy切片与索引

---

一维数组与普通数组类似

二维数组

```python
#方法一
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print (a[...,1])   # 第2列元素
print (a[1,...])   # 第2行元素
print (a[...,1:])  # 第2列及剩下的所有元素

#方法二
x = np.array([[1,  2],  [3,  4],  [5,  6]]) 
y = x[[0,1,2],  [0,1,0]] 
#输出(0,0),(1,1),(2,0)位置的元素

#如果想要切出二维数组
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
rows = np.array([[0,0],[3,3]]) 
cols = np.array([[0,2],[0,2]]) 
y = x[rows,cols]

#方法一和方法二可以组合起来使用

#布尔索引
print(a[a>5])
print(a[~np.isnan(a)])
print(a[np.iscomplex(a)])

#ix_
a = np.arange(32).reshape(4,8)
print(a[np.ix_([1,3],[2,5])])  #输出第二行第四行与第三列第六列的交
```

### Numpy广播

---

四则运算在对应位置上进行

两个数组形状不同时触发广播机制

**简单理解：** 对两个数组，分别比较他们的每一个维度（若其中一个数组没有当前维度则忽略），满足：

- 数组拥有相同形状。
- 当前维度的值相等。
- 当前维度的值有一个是 1。

### Numpy迭代操作

---

依靠`nditer` 

```python
a = np.arange(6).reshape(2,3)
for i in np.nditer(a):
	print(i)
	
#a.T迭代顺序一致，如果需要得到不同的结果，学要用到a.T.copy(order = 'C')
#或者np.nditer(a,order='F')

#广播迭代
a = np.arange(32).reshape(4,8)
b = np.array([1,2,3,4,5,6,7,8])
for x,y in np.nditer([a,b])
#相当于b = np.tile(b,(4,1))
```

---

- 修改数组形状

```python
#reshape
a = np.arange(32).reshape(8,4) #将数组变为8行4列

#flat
for i in a.flat:
	print(i)
#作用与迭代器一致,比nditer好用，转置后依旧按行读取

#flatten(深度拷贝，展开)
a.flatten() #默认按行读取
a.flatten(order = 'F') #按列读取

#ravel(浅拷贝，展开)
a.ravel()
a.ravel(order = 'F')
```

- 反转数组(都不如T) flip好用一点

```python
#transpose(用法与T相同，但可以自定义，还是用T吧)

#rollaxis
numpy.rollaxis(arr, axis, start)

#swapaxes
numpy.swapaxes(arr, axis1, axis2)

#flip
np.flip(a,axis = 0)
```

- 修改数组维度

```python
#broadcast广播
a = np.array([[1],[2],[3]])
b = np.array([1,2,3])
c = np.broadcast(a,b)
d.flat = [u+v for u,v in c]
输出：
[[2 3 4]
 [3 4 5]
 [4 5 6]]
 
 #broadcast_to(感觉不如tile)
 numpy.broadcast_to(array, shape, subok)
 a = np.array([1,2,3]).reshape(1,3)
 print(np.broadcast_to(a,(4,3)))
 输出：
 [[1 2 3]
  [1 2 3]
  [1 2 3]
  [1 2 3]]
  
 #expand_dims
 numpy.expand_dims(arr, axis)
 
 #squeeze(删除一维的轴)
 a = np.arange(9).reshape(1,3,3)
 np.squeeze(a)
 a.shape #(3,3)
```

- 连接数组

```python
#concatenate
numpy.concatenate((a1, a2, ...), axis) #axis默认为0

#stack
#hstack，横着拼
#vstack，竖着拼
```

- 分割数组

```python
#split
numpy.split(ary, indices_or_sections, axis)
a = np.arange(9)
np.split(a,3) #三分数组
np.split(a,[4,7]) #在指定位置分割数组 [0,1,2,3] [4,5,6] [7,8]

#vsplit 即split axis=0的情况

#hsplit 即split axis=1的情况
```

- 数组元素的添加与删除

```python
#resize
# np.resize 与 reshape 不等价：当目标元素更多时，np.resize 会重复原数组元素
np.resize(a, (3, 2))

#append
a = np.array([[1,2,3],[4,5,6]])
np.append(a,[7,8,9]) #[1 2 3 4 5 6 7 8 9]
np.append(a,[[7,8,9]],axis=0)#添加在最下面，不输入axis返回还是一维数组
np.append(a, [[7], [8]], axis=1) #按列添加，除拼接轴外的维度必须一致

#insert（具有广播性质,非原地）
a = np.array([[1,2],[3,4],[5,6]]) 
print (np.insert(a,3,[11,12])) #展开为一维数组 
print (np.insert(a,1,[11],axis = 0)) #插入到第一行 
print (np.insert(a,1,11,axis = 1)) #插入到第一列

#delete
np.delete(arr, obj, axis)
a = np.arange(12).reshape(3,4) 
print (np.delete(a,5)) #展开为一维数组
print (np.delete(a,1,axis = 1)) #删除第一列
a = np.array([1,2,3,4,5,6,7,8,9,10])
print (np.delete(a, np.s_[::2]))

#unique（去除重复元素）其中u是排序好的一维数组
arr：输入数组，如果不是一维数组则会展开
return_index：如果为true，返回新列表元素在旧列表中的位置（下标），并以列表形式储
return_inverse：如果为true，返回旧列表元素在新列表中的位置（下标），并以列表形式储
return_counts：如果为true，返回去重数组中的元素在原数组中的出现次数
numpy.unique(arr, return_index, return_inverse, return_counts) 
a = np.array([5,2,6,2,7,5,6,8,2,9])  
u,indices = np.unique(a,return_index = True) #u是去重好的数组，indices存储u中元素在a对应的下标
u,indices = np.unique(a,return_inverse = True) #u是去重好的数组，indices存储a中元素在u对应的下标
print (u[indices]) #使用下标数组构建原数组
u,indices = np.unique(a,return_counts = True) #u是去重好的数组，indices存储u中元素重复的数量
```

### 位运算

---

| 按位与 | `numpy.bitwise_and(x1, x2)` | 对数组的每个元素执行逐位与操作。 |
| --- | --- | --- |
| 按位或 | `numpy.bitwise_or(x1, x2)` | 对数组的每个元素执行逐位或操作。 |
| 按位异或 | `numpy.bitwise_xor(x1, x2)` | 对数组的每个元素执行逐位异或操作。 |
| 按位取反 | `numpy.invert(x)` | 对数组的每个元素执行逐位取反（按位非）。 |
| 左移 | `numpy.left_shift(x1, x2)` | 将数组的每个元素左移指定的位数。 |
| 右移 | `numpy.right_shift(x1, x2)` | 将数组的每个元素右移指定的位数。 |

具有广播性质

### Numpy字符串函数

---

`numpy.char.add()` 

```python
print (np.char.add(['hello'],[' xyz']))
print (np.char.add(['hello', 'hi'],[' abc', ' xyz']))
输出：
['hello xyz']
['hello abc' 'hi xyz']
```

`np.char.multiply()` 

```python
print(np.char.multiply(np.array([['a '],['b' ]]),3))
输出：
[['a a a ']
 ['b b b ']]
```

`numpy.char.center()`

```python
a = np.array([['a','b'],['c','d']])
print(np.char.center(a,5,'^'))
输出：
[['^^a^^' '^^b^^']
 ['^^c^^' '^^d^^']]

```

`np.char.capitalize()` 第一个字母转大写

```python
a = np.array([['a','b'],['c','d']])
print(np.char.capitalize(a))
输出：
[['A' 'B']
 ['C' 'D']]
```

`np.char.title()` 将每个字母的首字母转成大写

```python
a = np.array([['aa','bb'],['cc','dd']])
print(np.char.title(a))
输出：
[['Aa' 'Bb']
 ['Cc' 'Dd']]
```

`np.char.lower()` `np.char.upper()` 分别为转小写转大写

`np.char.split()`

```python
a = np.array([['a.a','b.b'],['c.c','d.d']])
print(np.char.split(a,'.'))
输出：
[[list(['a', 'a']) list(['b', 'b'])]
 [list(['c', 'c']) list(['d', 'd'])]]
```

`np.char.splitlines()` 以换行符作为分隔符

`np.char.strip()` 移除首尾特定的字符

```python
print(np.char.strip(['arunooba','admin','java'],'a'))
输出：
['runoob' 'dmin' 'jav']
```

`np.char.join()`

```python
print (np.char.join(':','runoob')) 
print (np.char.join([':','-'],['runoob','google']))
输出：
r:u:n:o:o:b
['r:u:n:o:o:b' 'g-o-o-g-l-e']
```

`np.char.replace()`

```python
print (np.char.replace ('i like runoob', 'oo', 'cc'))
输出：
i like runccb
```

`np.char.encode()` `np.char.decode()`

```python
np.char.encode('abc','gbk')
```

### Numpy数学函数

---

- 三角函数，需转为弧度制使用

```python
a = np.arange(32).reshape(4,8)
c = np.radians(a) #角度值转为弧度制
sin = np.sin(a*np.pi/180)
cos = np.cos(a*np.pi/180)
tan = np.tan(a*np.pi/180)
arcsin = np.arcsin(sin)
arccos = np.arccos(cos)
arctan = np.arctan(tan)
b = np.degrees(arcsin) #转为角度制，和a相等
```

- np.round(a,2) 保留两位小数
- np.floor(a) 向下取整数
- np.ceil(a) 向上取整数
- np.cumprod() 连乘
- np.cumsum() 连加
- np.diff() 和前一个元素作差
- np.power()

```python
a = np.array([1,2,3])
b = np.array([1,2,3])
print(np.power(a,b))
print(np.power(a,2))
输出：
[1 4 27]
[1 4 9]
```

- np.mod()

```python
a = np.array([10,20,30])
b = np.array([3,7,9])
print(np.mod(a,b))
输出：
[1 6 3]
```

- np.amin() np.amax

```python
a = np.arange(9).reshape(3,3)
print(np.amin(a,axis = 0))
print(np.amin(a,axis = 1))
输出：
[0 1 2]
[0 3 6]
```

- np.ptp() 计算数组中最大值减去最小值

```python
a = np.arange(9).reshape(3,3)
print(np.ptp(a))
print(np.ptp(a,axis = 0))
print(np.ptp(a,axis = 1))
输出：
8
[6 6 6]
[2 2 2]
```

- np.percentile()

```python
a = np.array([[10, 7, 4], [3, 2, 1]])  
print (np.percentile(a, 50))  
print (np.percentile(a, 50, axis=0))  
print (np.percentile(a, 50, axis=1))  
print (np.percentile(a, 50, axis=1, keepdims=True)) #keepdims = True保持维度不变
输出:
3.5
[6.5 4.5 2.5]
[7 2]
[[7]
 [2]]
```

- np.median() 求中位数
- np.mean() 求平均数
- np.average() 加权平均数

```python
numpy.average(a, axis=None, weights=None, returned=False)
#weights权重 returned = True同时返回加权平均数和权重总和
a = np.arange(9).reshape(3,3)
weights = np.array([1,2,3])
print(np.average(a, weights=weights, axis=1,returned=True))
输出：
(array([1.33333333, 4.33333333, 7.33333333]), array([6., 6., 6.]))
```

- np.std() 总体标准差
- np.var() 总体方差
- np.cov(rowvar = True) 样本协方差矩阵，默认每行为一个变量
- np.corrcoef() 样本相关系数矩阵
- np.conj(a)取共轭

想要过滤np.nan可在函数名称前加上nan

ddof = 0表示计算总体，ddof = 1表示计算样本

### Numpy排序、条件筛选函数

---

- np.sort()

```python
a = np.array([[8,2],[5,3]])
print(np.sort(a,axis = 0))
print(np.sort(a,axis = 1))
输出：
[[5 2]
 [8 3]]
[[2 8]
 [3 5]]
#order 用法得搭配dtype
dt = np.dtype([('name', 'S10'), ('age', int)])
a = np.array([('b',1),('a',2)], dtype=dt)
print(np.sort(a,order='age'))
输出：
[(b'b', 1) (b'a', 2)]
```

- np.argsort() 与sort用法相同，区别是他返回的是索引数组
- np.lexsort() 返回排序后的索引数列

```python
nm =  ('raju','anil','ravi','amar') 
dv =  ('f.y.',  's.y.',  's.y.',  'f.y.') 
ind = np.lexsort((dv,nm))   
print (ind) 
输出：
[3 1 0 2]
先按nm降序排，再按dv降序排
```

- np.sort_complex() 对虚数先按real排列，在按照imag排列
- np.partition()

```python
a = np.array([5,2,3,0,8,-1])
print(np.partition(a,(1,3)))
#比1小的排前面，比3小的排后面
输出：
[-1 0 2 3 8 5] 顺序不是很好
```

- np.argmax() np.argmin() 返回指定axis上最大值和最小值的索引
- np.nonzero() 返回数组中非零元素的索引
- np.where() 返回数组中满足条件元素的索引，也可以进行条件赋值

```python
a = np.array([1,2,3,4,5,6,7,8]).reshape(2,4)
print(np.where(a>5))
print(np.where(a>5,a,0))
输出：
(array([1, 1, 1]), array([1, 2, 3]))
[[0 0 0 0]
 [0 6 7 8]]
```

- np.extract() 根据某个条件从数组抽取元素，返回满条件的元素

```python
a = np.arange(9).reshape(3,3)
print(np.extract(np.mod(a,2)==0,a))
输出：
[0 2 4 6 8]
```
- `np.count_nonzero(a==0)`统计，与数组的`count()`函数类似
### Numpy副本与视图（深度拷贝与浅拷贝）

---

```python
a = np.array([1,2,3])
b = a #a,b同一个内存地址
c = a.view() #c,a不是同一个内存地址，但仍然关联
d = a[1:] #修改d会影响a
#修改数据会影响，但修改维度不影响

e = a.copy() #深度拷贝
```

### Numpy线性代数

| `dot` | 点积；对二维数组表现为矩阵乘法，并非元素对应相乘。 |
| :--- | --- |
| `vdot` | 两个向量的点积 |
| `inner` | 两个数组的内积 |
| `matmul` | 两个数组的矩阵积 |
| `determinant` | 数组的行列式 |
| `solve` | 求解线性矩阵方程 |
| `inv` | 计算矩阵的乘法逆矩阵 |
| `roots` | 解多项式方程 |

---

- 元素对应相乘：`a * b` 或 `np.multiply(a, b)`
- 矩阵乘法优先使用 `a @ b` 或 `np.matmul(a, b)`；二维数组也可使用 `np.dot(a, b)`
- np.vdot(a,b) 复数变为共轭计算,内积和
- np.inner(a,b) 内积，行乘行
- np.linalg.det() 计算行列式
- np.linalg.solve() 求解线性方程

```python
a = np.array([[1,2],[5,4]])
b = np.array([1,2])
print(np.linalg.solve(a,b))
输出：
[0 0.5]
```

- np.linalg.inv() 求解逆矩阵
- np.roots()

```python
coefficients = [1, -5, 6]  # x² -5x +6 =0
roots = np.roots(coefficients)  # 解为 [3., 2.]
print(roots)
```

- 计算特征值和特征向量`numpy.linalg.eig`

### 知识点补充

**通用函数（ufunc）**
- 一元函数
    ```python
    a = np.array([1,4,9])

    print(np.sqrt(a))   # 平方根 → [1. 2. 3.]
    print(np.exp(a))    # e^x
    print(np.log(a))    # 对数
    print(np.abs(a))    # 绝对值
    print(np.sign(a))   # 符号函数
    ```
- 二元函数
    ```python
    a = np.array([1,2,3])
    b = np.array([4,5,6])

    print(np.add(a,b))       # [5 7 9]
    print(np.subtract(a,b))  # [-3 -3 -3]
    print(np.multiply(a,b))  # [4 10 18]
    print(np.divide(a,b))    # [0.25 0.4 0.5]

    print(np.maximum(a,b))   # [4 5 6]
    print(np.minimum(a,b))   # [1 2 3]

    reduce / accumulate

    a = np.array([1,2,3,4])

    print(np.add.reduce(a))       # 10
    print(np.add.accumulate(a))   # [1 3 6 10]
    ```

**条件表达式（where）**
```python
a = np.array([1,2,3,4])

print(np.where(a > 2, a, 0))
# [0 0 3 4]

#条件组合
a = np.array([1,2,3,4])

print(a[(a > 1) & (a < 4)])
# [2 3]

#clip（截断）
a = np.array([1,5,10])
print(np.clip(a, 2, 8))
# [2 5 8]

a = np.array([1,2,2,5,5,5,8,8])
print(np.clip(a, 2, 8))
# [2 2 2 5 5 5 8 8]
```

**dtype（数据类型）**
```python
a = np.array([1,2,3])
print(a.dtype)  # int64

#指定类型
a = np.array([1,2,3], dtype=np.float32)

#类型转换
a = np.array([1.1,2.2,3.3])
print(a.astype(np.int32))  # [1 2 3]

#常见dtype

np.int32
np.float64
np.bool_
np.complex128
np.object_

#结构化dtype（了解）
dt = np.dtype([('name','U10'),('age','i4')])
a = np.array([('Tom',10),('Bob',20)], dtype=dt)
print(a['name'])  # ['Tom' 'Bob']
```

**文件读写**
```python
#保存数组
a = np.array([1,2,3])
np.save("a.npy", a)

#读取数组
b = np.load("a.npy")

#保存多个数组
np.savez("data.npz", a=a, b=b)

#读取
data = np.load("data.npz")
print(data['a'])
```

## 第三部分：Pandas

### 学习重点

对后续机器学习与 LLM 数据处理，重点掌握：

- `Series`、`DataFrame`
- `read_csv`、`head`、`shape`、`columns`
- `loc`、`iloc`、布尔筛选
- 缺失值与重复值处理
- `groupby`、`agg`
- `merge`、`concat`
- 字符串处理 `.str`
- `to_csv`

```python
import pandas as pd
import numpy as np
# axis 表示操作的维度，不应死记为“行”或“列”
# 对二维 DataFrame：axis=0 通常表示沿行方向操作，结果按列得到
# 对二维 DataFrame：axis=1 通常表示沿列方向操作，结果按行得到
```

### Pandas的主要数据结构
**Series**
```python
a = pd.Series([1,2,3,4],index = ['a','b','c','d']) #如果没有指定index，默认为0,1,2...

# 读取
b = a['a']
c = a[['a','b','c']]
d = a[a>1]

data = dict(a=1,b=2,c=3)
index = ["b","a","d"]
a = pd.Series(data,index=index) #index会与字典的key对应，没有的话value显示为Nan
'''
输出：
b    2.0
a    1.0
d    NaN
dtype: float64
'''

# 用isnull和notnull来检验是否有缺失值
print(pd.isnull(a)) #或者a.isnull()
'''
输出：
b    False
a    False
d     True
dtype: bool
'''

index_1 = ["a","b","c"]
b = pd.Series(data,index=index_1)
print(a+b) #会自动对齐标签
'''
输出：
a    2.0
b    4.0
c    NaN
d    NaN
dtype: float64
'''

# 可以给Series和index命名
a.name = "Sheet1"
a.index.name = "index_0"

# index可以直接赋值更改
a.index = ["x","y","z"]
```

**DataFrame**
```python
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
'year': [2000, 2001, 2002, 2001, 2002, 2003],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data) #可以通过columns（不在字典内数据显示为NaN）和index参数设置列名和行名
print(frame)
'''
输出：
    state  year  pop
0    Ohio  2000  1.5
1    Ohio  2001  1.7
2    Ohio  2002  3.6
3  Nevada  2001  2.4
4  Nevada  2002  2.9
5  Nevada  2003  3.2
'''

# 可通过以下方式获取一个列（Series）
a = frame["state"]
b = frame.year
c = frame.loc[3] #如果index是自己命名的，则通过设置的行名获取

frame2 = pd.DataFrame(data,columns = ["state","year","pop","debt"])
'''
输出：
    state  year  pop debt
0    Ohio  2000  1.5  NaN
1    Ohio  2001  1.7  NaN
2    Ohio  2002  3.6  NaN
3  Nevada  2001  2.4  NaN
4  Nevada  2002  2.9  NaN
5  Nevada  2003  3.2  NaN
'''

# 对列进行赋值
frame2["debt"] = 1 #整列都变为1
frame2["debt"] = np.arange(1,7) #数量要对应
val = pd.Series([-1.2, -1.5, -1.7], index=[2, 4, 5])
frame2['debt'] = val #用Series来赋值
frame2['eastern'] = frame2.state == 'Ohio' #新增列并用bool赋值，不能用frame2.eastern

# 删除列
del frame2['eastern'] #不能用frame2.eastern

# DataFrame转置
frame3 = frame2.T #columns和index会进行交换

# 还有一种创建方式不用考虑数量是否对应
pop = {'Nevada': {2001: 2.4, 2002: 2.9},'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = pd.DataFrame(pop)
'''
输出：
      Nevada  Ohio
2001     2.4   1.7
2002     2.9   3.6
2000     NaN   1.5
'''
# 或者指定索引
pd.DataFrame(pop, index=[2001, 2002, 2003])

# 也可以用Series
pdata = {'Ohio': frame3['Ohio'][:-1],'Nevada': frame3['Nevada'][:2]}
pd.DataFrame(pdata) #也是和索引对应

# 同样，DataFrame,index,columns也可以被命名
# index不可以单个修改，要改就全部替换
```

### 基本功能
**重新索引`reindex`**
```python
# Series
a = pd.Series([1,2,3,4,5],index = ["a","b","c","d","e"])
b = a.reindex(["a","b","c","e","d","f"]) #根据新的索引重排，缺失值填NaN

c = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
d = c.reindex(range(5),method = "ffill") #使用method方法实现前向值填充，fill_value可定义填充的值

# DataFrame
frame = pd.DataFrame(np.arange(9).reshape((3, 3)),index=['a', 'c', 'd'],columns=['Ohio', 'Texas', 'California'])
frame2 = frame.reindex(['a', 'b', 'c', 'd']) #缺失行填充
states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states) #重排columns
```

**删除`drop`**
```python
# Series
a = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
b = a.drop('c')
c = a.drop(["a","b"])

# DataFrame
data = pd.DataFrame(np.arange(16).reshape((4, 4)),index=['Ohio', 'Colorado', 'Utah', 'New York'],columns=['one', 'two', 'three', 'four'])
b = data.drop(['Colorado', 'Ohio']) #删除行
c = data.drop('two', axis=1) #删除列
d = data.drop(['two', 'four'], axis='columns')
# 添加参数inplace = True，可实现原地修改
```

**索引、选取、过滤**
```python
# Series
a = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
'''
输出：
a    0.0
b    1.0
c    2.0
d    3.0
dtype: float64
'''
b = a["a"]
b = a[0]
c = a[0:2]
c_1 = a["a":"b"] #包含末端
d = a[['a', 'b', 'c']]
e = a[[0,1,2]]
f = a[a<3]

# DataFrame
data = pd.DataFrame(np.arange(16).reshape((4, 4)),index=['Ohio', 'Colorado', 'Utah', 'New York'],columns=['one', 'two', 'three', 'four'])
a = data["one"]
b = data[["one","two"]]
# 取行
c = data[:2]
d = data[data["one"]>4]
```

**用`loc`和`iloc`进行选取**
```python
data = pd.DataFrame(np.arange(16).reshape((4, 4)),index=['Ohio', 'Colorado', 'Utah', 'New York'],columns=['one', 'two', 'three', 'four'])
a = data.loc['Colorado', ['two', 'three']] #会转变成Series
a_1 = data.loc[['Ohio','Colorado'], ['two', 'three']] #此时依旧为DataFrame
b = data.iloc[2, [3, 0, 1]] #用整数索引来取，依旧行列
# loc和iloc可以使用切片
```

**算数**
```python
# Series和DataFrame都支持加减操作，缺失位置填充NaN
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)),columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)),columns=list('abcde'))
# 用填充值替代缺失值
a = df1.add(df2,fill_value = 0) #如果df1中没有df2中相应位置的元素，转变为0+df2，对于df2来说也是如此
```

**DataFrame和Series之间的运算**
```python
frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),columns=list('bde'),index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series2 = pd.Series(range(3), index=['b', 'e', 'f'])
a = frame + series2 #series2会转变为行，用index去匹配frame的columns
series3 = pd.Series(range(4), index=frame.index)
b = frame.sub(series3, axis='index') #用Series的index匹配DataFrame的index
```

**基本运算函数**(`r`开头要颠倒，比如`a.radd(b)`等价于`b+a`)
`add:+` `radd:+`
`sub:-` `rsub:-`
`div:/` `rdiv:/`
`floordiv://` `rfloordiv://`
`mul:*` `rmul:*`
`pow:**` `rpow:**`    

**函数运用与映射**
```python
frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),index=['Utah', 'Ohio', 'Texas', 'Oregon'])
f = lambda x:x.max()-x.min()
a = frame.apply(f) #默认axis = 0，完成后转换为Series

def f(x):
    return pd.Series([x.min(),x.max()],index = ["min","max"])
a = frame.apply(f) #返回多个值组成的Series
# applymap()用于辐射到每一个元素，Series中用map()
```

**排序和排名**
```python
# Series
a = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
b = a.sort_index() #默认升序,accending = True
c = a.sort_values()
d = a.rank() #平均排名，值相同取平均，比如5，6名相同，取6.5，默认accending = True
e = a.rank(method = "first") #此时不取平均排名，相同值按照出现顺序
# method = "max" 取最大排名

# DataFrame
frame = pd.DataFrame(np.arange(8).reshape((2, 4)),index=['three', 'one'],columns=['d', 'a', 'b', 'c'])
a = frame.sort_index() #默认按照index
b = frame.sort_index(axis = 1) #按照columns
c = frame.sort_values(by = "d")
d = frame.sort_values(by = ["d","a"]) #先按d，如果值相同，按照a
#rank同理
```

**带有重复标签的轴索引**
```python
obj = pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
print(obj.index.is_unique) #检验是否唯一
print(obj["a"]) #返回一个Series表，对应多值
```

### 汇总和计算描述统计
```python
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],[np.nan, np.nan], [0.75, -1.3]],index=['a', 'b', 'c', 'd'],columns=['one', 'two'])
a = df.sum() #自动忽略NaN
b = df.sum(skipna = False) #不忽略NaN
c = df.idxmax() #得到每一列最大值所在的索引
d = df.cumsum() #累加
e = df.cumprod() #累乘
print(df.describe()) #数值型
'''
输出：
            one       two
count  3.000000  2.000000
mean   3.083333 -2.900000
std    3.493685  2.262742
min    0.750000 -4.500000
25%    1.075000 -3.700000
50%    1.400000 -2.900000
75%    4.250000 -2.100000
max    7.100000 -1.300000
'''
obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
print(obj.describe()) #非数值型
'''
输出：
count     16
unique     3
top        a
freq       8
dtype: object
'''
```

**相关系数与协方差**
```python
frame = pd.DataFrame(np.arange(16).reshape(4,4),index = [1,2,3,4],columns = ["a","b","c","d"])
print(frame['a'].cov(frame['b'])) #样本协方差
print(frame['a'].corr(frame['b'])) #样本相关系数
print(frame.cov()) #样本相关阵
print(frame.corr()) #样本相关系数阵
print(frame.corrwith(frame["b"])) #计算每列和b的样本相关系数，传入axis = "columns"可以进行行运算
```

**唯一值、值计数以及成员资格**
```python
a = pd.Series(["a","a","b"]*4)
uniques = a.unique() #返回a中的唯一值数组，未排序，可进行sort()
count = a.value_counts() #计算各值出现的频率，按照值频率降序排列，sort = False可取消
mask = a.isin(["a","d"]) #判断元素是否在["a","d"]中
a[mask] #可得到包含其中的子集

to_match = pd.Series(['c', 'a', 'b', 'b', 'c', 'a'])
unique_vals = pd.Series(['c', 'b', 'a'])
pd.Index(unique_vals).get_indexer(to_match) #匹配下标
'''
输出：
array([0, 2, 1, 1, 0, 2])
'''

data = pd.DataFrame({'Qu1': [1, 3, 4, 3, 4],'Qu2': [2, 3, 1, 2, 3],'Qu3': [1, 5, 2, 4, 4]})
result = data.apply(lambda column: column.value_counts()).fillna(0) #统计每列中各值出现的次数
```

### 补充：Pandas 数据处理必备操作

#### 一、文件读写

```python
import pandas as pd

df = pd.read_csv("data.csv")
df_excel = pd.read_excel("data.xlsx")
df_json = pd.read_json("data.json")

# index=False：不把 DataFrame 的行索引额外写入文件
df.to_csv("output.csv", index=False)
```

读取后通常先检查：

```python
df.head()
df.shape
df.columns
df.dtypes
df.info()
```

#### 二、缺失值处理

```python
# 判断缺失值
df.isna()
df.notna()

# 统计每列缺失值数量
df.isna().sum()

# 删除含缺失值的行
df_clean = df.dropna()

# 只要求指定列非缺失
df_clean = df.dropna(subset=["text", "label"])

# 用固定值或统计量填充
df["score"] = df["score"].fillna(0)
df["score"] = df["score"].fillna(df["score"].mean())
```

不要不加判断地删除或填充缺失值；具体方法取决于数据含义和后续任务。

#### 三、分组统计：`groupby()` 与 `agg()`

```python
df = pd.DataFrame({
    "model": ["Qwen", "Qwen", "Llama", "Llama"],
    "method": ["DPO", "SimPO", "DPO", "SimPO"],
    "score": [0.80, 0.90, 0.70, 0.82],
})

# 每个模型的平均分
mean_score = df.groupby("model")["score"].mean()

# 按多个变量分组
group_score = df.groupby(["model", "method"])["score"].mean()

# 一次计算多个统计量
summary = df.groupby("model").agg(
    mean_score=("score", "mean"),
    max_score=("score", "max"),
    n=("score", "count"),
)
```

#### 四、数据合并：`merge()` 与 `concat()`

```python
rewards = pd.DataFrame({
    "user_id": [1, 2],
    "reward": [0.8, 0.9],
})

preferences = pd.DataFrame({
    "user_id": [1, 3],
    "preference": ["A", "B"],
})

# 按键横向合并
inner = pd.merge(rewards, preferences, on="user_id", how="inner")
left = pd.merge(rewards, preferences, on="user_id", how="left")
outer = pd.merge(rewards, preferences, on="user_id", how="outer")

# 纵向拼接结构相同的表
all_rows = pd.concat([rewards, rewards], axis=0, ignore_index=True)

# 横向拼接
all_columns = pd.concat([rewards, preferences], axis=1)
```

- `inner`：只保留两边都能匹配的键。
- `left`：保留左表全部键。
- `outer`：保留两边所有键，无法匹配的位置为缺失值。

#### 五、字符串处理 `.str`

```python
df = pd.DataFrame({"text": [" Hello ", "WORLD", "hello pandas"]})

df["text_clean"] = df["text"].str.strip().str.lower()
df["contains_hello"] = df["text"].str.contains("hello", case=False, na=False)
df["text_length"] = df["text"].str.len()
```

#### 六、重复值

```python
# 判断整行是否重复
df.duplicated()

# 按指定列判断重复
df.duplicated(subset=["user_id"])

# 删除重复行
df_unique = df.drop_duplicates()
df_unique = df.drop_duplicates(subset=["user_id"], keep="first")
```

#### 七、类型转换

```python
df["label"] = df["label"].astype("int64")
df["score"] = df["score"].astype("float32")
df["text"] = df["text"].astype("string")
```

类型转换失败时，不要盲目强制转换；先检查异常值。数值列可使用：

```python
df["score"] = pd.to_numeric(df["score"], errors="coerce")
```

无法转换的值会变为 `NaN`，随后再按任务要求处理。

### Pandas 学习边界

掌握本节以及前文的 `Series`、`DataFrame`、`loc`、`iloc`、筛选、排序、`value_counts()` 和 `describe()` 后，就足以进入 PyTorch。复杂透视表、时间序列和性能优化可在实际项目中按需学习。
