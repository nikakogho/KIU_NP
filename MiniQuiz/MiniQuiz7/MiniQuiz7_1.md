We have points (0, 1), (2, 3) and (3, 2)

So
x0 = 0
y0 = 1

x1 = 2
y1 = 3

x2 = 3
y2 = 2

Intervals: [0, 2] and [2, 3]

h0 = x1 - x0 = 2
h1 = x2 - x1 = 1

M0 = S''(x0) = M2 = S''(x2) = 0

Must find M1 = S''(x1)

Standard equation at interior knot x1:
h0 * M0 + 2 * (h0 + h1) * M1 + h1 * M2 = 6 * ((y2-y1)/h1 - (y1-y0)/h0)

By plugging in the values we get
2 * 0 + 2 * 3 * M1 + 1 * 0 = 6 * (-1/1 - 2/2)
so
6 * M1 = 6 * (-1 - 1)
6 * M1 = 6 * (-2)
M1 = -2

Therefore we have
M0 = 0
M1 = -2
M2 = 0

Now we build spline on each interval:

For [0, 2]
S0 (x) = -x^3 / 6 + (2-x)/2 + 13x/6 = -x^3/6 + 5/3x + 1

For [2, 3]
S1 (x) = (x-3)^3 / 3 - 4/3x + 6