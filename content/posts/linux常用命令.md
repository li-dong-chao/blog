---
title: linux常用命令
date: 2023-10-01
categories: ["linux"]
---

## ln

ln命令可以用来创建链接，包括 **软链接** 和 **硬链接** 。

### 软链接与硬链接概念明晰

在介绍这两个概念之前，需要先说一下 **索引节点** 这个概念。

linux下“一切皆文件”，系统会给所有保存在磁盘中的文件分配一个编号，这个编号就是 **索引节点编号** ，
它是文件或者目录在linux系统下的唯一标识。

linux允许多个文件名同时指向一个索引节点，这种情况就是硬链接。
通俗的说，就是假设现在存在一个文件 `A` ，为了防止文件被误删，我又创建一个文件 `B` ，
并让 `B` 的索引节点编号和 `A` 的索引节点编号相同，此时 `B` 就是 `A` 的硬链接，
更准确的说是两者互为对方的硬链接。

使用硬链接可以有效的防止文件误删，但是有以下需要注意的地方：

- 删除硬链接文件或者删除源文件任意之一，文件实体并不会被删除，只有删除了源文件和所有对应的硬链接文件，文件实体才会被删除；
- 硬链接文件删除方式与普通文件相同，使用rm命令即可删除；
- 对于静态文件（没有进程正在调用），当硬链接数为0时文件就被删除。注意：如果有进程正在调用，则无法删除或者即使文件名被删除但空间不会释放；
- 不能对目录创建硬链接，不能对不同文件系统创建硬链接，不能对不存在的文件创建硬链接。

软链接的概念理解起来就相对容易一些，它和windows下的快捷方式类似。
软连接是一个普通文件，不过该文件记录了其源文件的路径指向，根据记录的内容，软连接可以快速定位到其源文件。

关于这两个概念，这里提供两篇文章，方便更好的理解。

- <https://xzchsia.github.io/2020/03/05/linux-hard-soft-link/>
- <https://blog.csdn.net/LEON1741/article/details/100136449>

### ln用法

```shell
ln 参数 源文件 目标文件
```

注意：

- 如果不加参数，则创建的是硬链接
- 源目录和目标目录都必须是 **绝对路径**

常用的参数如下：

- `-i`: 交互模式，文件存在则提示用户是否覆盖
- `-s`: 软链接（符号链接）
- `-d`: 允许超级用户制作目录的硬链接
- `-b`: 删除，覆盖以前建立的链接
- `-f`: 强制执行
- `-n`: 把符号链接视为一般目录
- `-v`: 显示详细的处理过程

### 使用场景举例

安装python到 `/usr/local/python` 目录下，由于该目录不是全局环境变量，因此无法全局使用 `python3` 命令。
可以使用 `ln` 命令创建一个软链接到 `/usr/bin` 目录下，这样便可以全局使用 `python3` 命令了。

```shell
ln -s /usr/local/python/bin/python3 /usr/bin/python3
```

## source

### 用法

```shell
source filename
. filename  # 也可以直接使用点命令
```

`source` 命令和 `.` 命令作用相同，区别在于前者来源于 `C Shell` ，而后者来源于 `Bourne Shell` .

### 作用

source命令的作用的是在当前的bash环境中读取并执行filename中的内容。

### 使用场景

- 通常用于重新执行刚修改的初始化文档，如 .bash_profile 和 .profile 等。
- 需要在当前shell中执行某些命令，或者初始化某些变量。

> 与 `sh` 的区别在于： `source` 是在当前shell中执行，而 `sh` 会新创建一个shell执行。
> 当新创建一个shell时，作用域中的内容会有所不同。

## echo

使用linux下的 `echo` 命令可以控制输出文字的颜色，用法如下：

```shell
echo -e "\033[x;ymtext\033[0m"
```

解释一下上面的命令。

首先， `echo -e` 表示使用 `echo` 命令输出字符串，
其中 `-e` 参数表示启用转义字符，即使用-e选项时，
若字符串中出现以下字符，将单独进行处理，而不会将它当成一般文字输出。

- `\a`: 发出警告声
- `\b`: 删除前一个字符
- `\c`: 不产生进一步输出 (\c 后面的字符不会输出)
- `\f`: 换行但光标仍旧停留在原来的位置
- `\n`: 换行且光标移至行首
- `\r`: 光标移至行首，但不换行
- `\t`: 插入tab
- `\v`: 与\\f相同
- `\\`: 插入\字符
- `\nnn`: 插入 `nnn` （八进制）所代表的ASCII字符；

`\033` 引导非常规字符序列，
`m` 意味着设置属性然后结束非常规字符序列。

`text` 即为需要进行特殊显示的字符串。

`x` 和 `y` 则表示了不同的显示样式，多个显示样式之间用 `;` 分割开。

支持的样式可以分为三大类。

1. 字体颜色

   - 30: 黑色字
   - 31: 红色字
   - 32: 绿色字
   - 33: 黄色字
   - 34: 蓝色字
   - 35: 紫色字
   - 36: 天蓝字
   - 37: 白色字

2. 背景颜色

   - 40: 黑底
   - 41: 红底
   - 42: 绿底
   - 43: 黄底
   - 44: 蓝底
   - 45: 紫底
   - 46: 天蓝底
   - 47: 白底

3. 其他控制项

   - 0: 重置所有属性
   - 1: 设置高亮度
   - 4: 下划线
   - 5: 闪烁
   - 7: 反显
   - 8: 消隐

> 这里的分类是笔者为方便记忆理解自己划分的， 从echo的设计上来说，仅仅是不同的数字代表不同的样式而已，并不存在这样的分类。
>

由于 `0` 可以重置所有的属性，因此经常会看到`echo -e "\033[41;36m something here \033[0m"` 这样的命令。
在输出 `something here` 之后，会紧跟一个 `\033[0m` 来清除前面的设置，以确保不会影响后续内容的正常展示。

使用 `echo` 命令时，添加 `-n` 参数表示不换行显示。

在shell脚本中使用 `echo` 命令可以实现不同颜色日志的输出，可以参考 [](./shell%E8%84%9A%E6%9C%AC%E7%BC%96%E7%A0%81%E6%89%8B%E5%86%8C.md) 的代码。

## sed

sed命令是linux系统下常用的一种 **行编辑器** ，与vim打开文件并进行编辑的方式不同，
sed可以非交互式的实现对文件内容的增删改查处理。
鉴于sed非交互特性，它经常被写在shell脚本中，以非交互的形式实现对文本文件的增删改查操作。

> sed是一种行编辑器，这是指在使用sed进行文本文件处理时，sed会把行看做一个最小的处理单元。
>

### 用法

`sed` 命令的基本用法如下：

```shell
sed [选项] [脚本命令] 文件名
```

其中支持的选项有：

- `-e<script>`: 该选项会将其后跟的脚本命令添加到已有的命令中；
- `-f<script>`: 该选项会将其后文件中的脚本命令添加到已有的命令中；
- `-n`: 该选项会仅显示发生改动的行，如果不加该选项，会默认输出处理文件的全部内容；
- `-i`: 添加该选项会直接修改文件内容。

脚本命令定义了对文件的处理方式，脚本文件通过动作参数来标识如何进行处理的，支持的动作有：

- a: 新增，a后面接字串，这些字符串会在当前处理行的下一行出现；
- c: 取代，c后面接字符串，可以取代指定位置的内容；
- d: 删除，删除指定的行；
- i: 插入，i后面接字符串，这些字符串会在当前行的上一行出现；
- p: 打印，将指定的内容打印输出；
- s: 取代，可以通过内容匹配来实现对内容的替换。

下面逐个说明上述的动作。

#### s 取代

s动作的基本命令格式是：

```shell
sed [选项] '[address]s/pattern/replacement/flags' [文件]
```

其中，address表示具体要执行操作的行，不写默认全部的行，
pattern表示要被替换的内容（旧的内容），
replacement表示要替换为的内容（新的内容），
flags标记了不同的功能。

flags不同标记对应的功能说明：

| flags标记 |                                      功能                                      |
| :-------: | :----------------------------------------------------------------------------: |
|     n     | 这里的n是代指一个1~512之间的数字，表示指定要替换的字符串出现第几次时才进行替换 |
|     g     | 对数据中所有匹配到的内容进行替换，如果没有该标记，默认只替换第一次匹配到的内容 |
|     p     |                       打印与替换命令中指定的模式匹配的行                       |
|  w file   |               将缓冲区中的内容写到指定的 file 文件中，类似另存为               |
|     &     |                         用正则表达式匹配的内容进行替换                         |

下面通过举例展示s动作的用法。

要处理的文件为test.txt，该文件内容为：

```shell
this is a test test
this is a test test
```

执行命令 `sed 's/test/new/2' test.txt`，输出结果为：

```shell
this is a test new
this is a test new
```

可以看出 `2` 标记使得每一行的第2个test都改为了new。

而如果执行命令 `sed 's/test/new/g' test.txt`，则输出结果为：

```shell
this is a new new
this is a new new
```

可以看出 `g` 标记使得每一行的所有test都改为了new。

> 注意sed在执行脚本命令时，是以行为单位执行的。
> 即会对每一行都执行一遍脚本指令，且脚本指令的作用范围也仅限于每一行。

如果在上面命令的基础上添加 `w new.txt`，改为 `sed 's/test/new/gw new.txt' test.txt`，
那么会在当前目录下重新生成的一个文件 `new.txt`，其内容为上面命令处理的结果。

> 如果要替换的内容中存在路径等，可以使用 `\` 进行转义处理。
> 如 `sed 's/\/bin\/bash/\/bin\/csh/' /etc/passwd`

#### d 删除

基本格式：

```html
sed [选项] '[address]d' [文件]
```

d动作用法比较简单，这里只给出常见用法的说明。

- `sed '1d' test.txt`: 删除test.txt文件的第1行
- `sed '3,10d test.txt'`: 删除test.txt文件第3行到第10行之间的全部内容
- `sed '3,$d test.txt'`: 删除test.txt文件第3行到最后一行的全部内容

#### a 新增 & i 插入

这两个动作的用法基本完全相同，区别在a会添加在当前处理行的下一行，而i会添加在当前行的上一行。

基本格式：

```shell
sed [选项] '[address]a\content' [文件]
sed [选项] '[address]i\content' [文件]
```

同样的，下面也是给出常见用法的说明。

- `sed 'anew line' test.txt`: 在test.txt文件的每一行 **下面** 都增加一行，且新增加行的内容为new line
- `sed 'inew line' test.txt`: 在test.txt文件的每一行 **上面** 都增加一行，且新增加行的内容为new line
- `sed '1anew line' test.txt`: 在test.txt文件的第1行 **下面** 都增加一行，且新增加行的内容为new line

> 如果你想添加多行文本，可以参考下面的代码。
>
> ```shell
> sed '1i\
> > newline1\
> > newline2' test.txt
> ```
>
> 注意在添加多行时，除最后一行外每行的结尾都是 `\`.
>

#### c 取代

`c` 的用法与 `a` 、 `i` 、 `d` 大致类似。

```shell
sed [选项] '[address]c\content' [文件]
```

- `sed '3c\changeline' test.txt`: 修改第3行为changeline

> 如果使用命令 `sed '1,3c\changeline' test.txt`，并不会把第1行到第3行的每一行都换为changeline，而是会把第1行到第3行整个替换为一个changeline。
>