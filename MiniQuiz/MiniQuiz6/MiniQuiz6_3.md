We’re given the data points

$$
(1,7),\quad (2,9),\quad (5,4),\quad (9,9)
$$

So

$$
x_0=1,; x_1=2,; x_2=5,; x_3=9, \qquad
f(x_0)=7,; f(x_1)=9,; f(x_2)=4,; f(x_3)=9.
$$

---

## 1. Divided difference table

### 0-th order (just the (y)-values)

$$
f[x_0]=7,\quad f[x_1]=9,\quad f[x_2]=4,\quad f[x_3]=9
$$

### 1st order divided differences

Formula:

$$
f[x_i,x_{i+1}] = \frac{f[x_{i+1}]-f[x_i]}{x_{i+1}-x_i}
$$

Do them one by one:

* Between (x_0=1) and (x_1=2):

$$
f[x_0,x_1] = \frac{9-7}{2-1} = \frac{2}{1} = 2
$$

* Between (x_1=2) and (x_2=5):

$$
f[x_1,x_2] = \frac{4-9}{5-2} = \frac{-5}{3} = -\frac53
$$

* Between (x_2=5) and (x_3=9):

$$
f[x_2,x_3] = \frac{9-4}{9-5} = \frac{5}{4}
$$

### 2nd order divided differences

Formula:

$$
f[x_i,x_{i+1},x_{i+2}]
= \frac{f[x_{i+1},x_{i+2}] - f[x_i,x_{i+1}]}{x_{i+2}-x_i}
$$

* Using (x_0,x_1,x_2):

$$
f[x_0,x_1,x_2]
= \frac{f[x_1,x_2]-f[x_0,x_1]}{x_2-x_0}
= \frac{-\frac53 - 2}{5-1}
= \frac{-\frac53 - \frac{6}{3}}{4}
= \frac{-\frac{11}{3}}{4}
= -\frac{11}{12}
$$

* Using (x_1,x_2,x_3):

$$
f[x_1,x_2,x_3]
= \frac{f[x_2,x_3]-f[x_1,x_2]}{x_3-x_1}
= \frac{\frac54 -\left(-\frac53\right)}{9-2}
= \frac{\frac54 + \frac53}{7}
= \frac{\frac{15}{12}+\frac{20}{12}}{7}
= \frac{\frac{35}{12}}{7}
= \frac{35}{12}\cdot\frac{1}{7}
= \frac{5}{12}
$$

### 3rd order divided difference

Formula:

$$
f[x_0,x_1,x_2,x_3]
= \frac{f[x_1,x_2,x_3]-f[x_0,x_1,x_2]}{x_3-x_0}
$$

$$
f[x_0,x_1,x_2,x_3]
= \frac{\frac{5}{12} -\left(-\frac{11}{12}\right)}{9-1}
= \frac{\frac{5}{12} + \frac{11}{12}}{8}
= \frac{\frac{16}{12}}{8}
= \frac{\frac{4}{3}}{8}
= \frac{4}{3}\cdot\frac{1}{8}
= \frac{1}{6}
$$

Now the table (just collecting values):

| (x_i) | (f[x_i]) |  1st order |        2nd order | 3rd order |
| ----- | -------: | ---------: | ---------------: | --------: |
| 1     |        7 |        2   | -11/12           | 1/6       |
| 2     |        9 | -5/3       |   5/12           |           |
| 5     |        4 |  5/4       |                  |           |
| 9     |        9 |            |                  |           |

The coefficients we need for the Newton form are the **top entries** of each column:

$$
a_0 = f[x_0] = 7,\quad
a_1 = f[x_0,x_1] = 2,\quad
a_2 = f[x_0,x_1,x_2] = -\frac{11}{12},\quad
a_3 = f[x_0,x_1,x_2,x_3] = \frac{1}{6}.
$$

---

## 2. Newton interpolating polynomial

Newton form (starting from (x_0)) is

$$
P(x) = a_0

* a_1(x-x_0)
* a_2(x-x_0)(x-x_1)
* a_3(x-x_0)(x-x_1)(x-x_2).
$$

Plug in the coefficients and points:

$$
\boxed{
P(x) = 7

* 2(x-1)
  -\frac{11}{12}(x-1)(x-2)
* \frac{1}{6}(x-1)(x-2)(x-5)
  }
$$

(That’s already in Newton form, so we’re done for part 2.)

---

## 3. Evaluate (P(4))

Now substitute (x = 4) into the Newton form.

First compute the differences:

$$
4-1 = 3,\quad
4-2 = 2,\quad
4-5 = -1.
$$

Term by term:

1. Constant term: (7)

2. First-order term:

$$
2(x-1) = 2\cdot 3 = 6
$$

3. Second-order term:

$$
-\frac{11}{12}(x-1)(x-2)
= -\frac{11}{12}\cdot 3 \cdot 2
= -\frac{11}{12}\cdot 6
= -\frac{66}{12}
= -\frac{11}{2}
$$

4. Third-order term:

$$
\frac{1}{6}(x-1)(x-2)(x-5)
= \frac{1}{6}\cdot 3 \cdot 2 \cdot (-1)
= \frac{1}{6}\cdot (-6)
= -1
$$

Now add them:

$$
P(4) = 7 + 6 - \frac{11}{2} - 1
$$

First (7 + 6 = 13), then (13 - 1 = 12):

$$
P(4) = 12 - \frac{11}{2}
= \frac{24}{2} - \frac{11}{2}
= \frac{13}{2}
$$

So

$$
\boxed{P(4) = \frac{13}{2} = 6.5}
$$
