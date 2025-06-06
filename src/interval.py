
import torch
from typing import Union
import functools


class Interval:

    def __init__(self, lower: Union[torch.Tensor, int, float], upper: Union[torch.Tensor, int, float]):
        self.lower = lower if isinstance(lower, torch.Tensor) else torch.as_tensor(lower)
        self.upper = upper if isinstance(upper, torch.Tensor) else torch.as_tensor(upper)

        if (self.lower > self.upper).any():
            raise Exception("Invalid Interval!")

    def __repr__(self):
        return f"({self.lower}, {self.upper})"
    
    def __eq__(self, right_operand: 'Interval'):
        if isinstance(right_operand.lower, int | float):
            if self.lower == right_operand.lower and self.upper == right_operand.upper:
                return True
            return False
        if (self.lower == right_operand.lower).all() and (self.upper == right_operand.upper).all():
                return True
        return False
        
    def __add__(self, right_operand: Union[torch.Tensor, 'Interval']):
        if right_operand.__class__ != Interval:
            lb = self.lower + right_operand
            ub = self.upper + right_operand
            return Interval(lb, ub)
        lb = self.lower + right_operand.lower
        ub = self.upper + right_operand.upper
        return Interval(lb, ub)

    def __radd__(self, left_operand: Union[torch.Tensor, int, float]):
        lb = left_operand + self.lower
        ub = left_operand + self.upper
        return Interval(lb, ub)

    def __sub__(self, right_operand: Union[torch.Tensor, 'Interval']):
        if right_operand.__class__ != Interval:
            lb = self.lower - right_operand
            ub = self.upper - right_operand
            return Interval(lb, ub)
        lb = self.lower - right_operand.upper
        ub = self.upper - right_operand.lower
        return Interval(lb, ub)

    def __rsub__(self, left_operand: Union[torch.Tensor, int, float]):
        return Interval(left_operand, left_operand) - self
        
    def __mul__(self, right_operand: Union['Interval', torch.Tensor, int, float]):
        if right_operand.__class__ != Interval:
            p1 = self.lower * right_operand
            p2 = self.upper * right_operand
            lb = functools.reduce(torch.min, [p1, p2])
            ub = functools.reduce(torch.max, [p1, p2])
            return Interval(lb, ub)
    
        p1 = self.lower * right_operand.lower 
        p2 = self.lower * right_operand.upper 
        p3 = self.upper * right_operand.lower 
        p4 = self.upper * right_operand.upper
        lb = functools.reduce(torch.min, [p1, p2, p3, p4])
        ub = functools.reduce(torch.max, [p1, p2, p3, p4])
        return Interval(lb, ub)
 
    def __rmul__(self, left_operand: Union[torch.Tensor, int, float]):
        p1 = left_operand * self.lower
        p2 = left_operand * self.upper 
        lb = functools.reduce(torch.min, [p1, p2])
        ub = functools.reduce(torch.max, [p1, p2])
        return Interval(lb, ub)

    def __matmul__(self, right_operand: Union['Interval', torch.Tensor, int, float]):
        
        if right_operand.__class__ != Interval:
            b_pos, b_neg = self.positive_and_negative_parts(right_operand)
            lb = (self.lower @ b_pos) + (self.upper @ b_neg)
            ub = (self.upper @ b_pos) + (self.lower @ b_neg)
            return Interval(lb, ub)

        u, v = self.lower, self.upper
        w, x = right_operand.lower, right_operand.upper
        u_pos, u_neg = self.positive_and_negative_parts(u)
        v_pos, v_neg = self.positive_and_negative_parts(v)
        w_pos, w_neg = self.positive_and_negative_parts(w)
        x_pos, x_neg = self.positive_and_negative_parts(x)
    
        min_pairs = [(u_pos, w_pos), (v_pos, w_neg),
                    (u_neg, x_pos), (v_neg, x_neg)]
        min_vals = functools.reduce( torch.add, [x @ y for x, y in min_pairs])
        max_pairs = [(v_pos, x_pos), (v_neg, w_pos),
                    (u_pos, x_neg), (u_neg, w_neg)]
        max_vals = functools.reduce( torch.add, [x @ y for x, y in max_pairs])
        return Interval(min_vals, max_vals)
    
    def __rmatmul__(self, left_operand: Union[torch.Tensor, int, float] ):
        a_pos, a_neg = self.positive_and_negative_parts(left_operand)
        lb = a_pos @ self.lower + a_neg @ self.upper
        ub = a_pos @ self.upper + a_neg @ self.lower
        return Interval(lb, ub)
    
    def __gt__(self, right_operand: Union['Interval', torch.Tensor, int, float]):
        if right_operand.__class__ != Interval:
            lb = (self.lower > right_operand).float()
            ub = (self.upper > right_operand).float()
            return Interval(lb, ub)
        lb = (self.lower > right_operand.lower).float()
        ub = (self.upper > right_operand.upper).float()
        return Interval(lb, ub)

    def __lt__(self, right_operand: Union['Interval', torch.Tensor, int, float]):
        if right_operand.__class__ != Interval:
            lb = (self.lower < right_operand).float()
            ub = (self.upper < right_operand).float()
            return Interval(lb, ub)
        lb = (self.lower < right_operand.lower).float()
        ub = (self.upper < right_operand.upper).float()
        return Interval(lb, ub)
    
    
    @property
    def shape(self):
        return self.lower.shape
    
    @property
    def T(self):
        return Interval(self.lower.T, self.upper.T)
    
    
    @property
    def width(self):
        return abs(self.upper - self.lower)

    @staticmethod
    def positive_and_negative_parts(x):
        return torch.maximum(torch.tensor(0), x), torch.minimum(torch.tensor(0), x)
    
    @staticmethod
    def sigmoid(x):
        if x.__class__ != Interval:
            return torch.sigmoid(x)
        return Interval(torch.sigmoid(x.lower), torch.sigmoid(x.upper))
    
if __name__ == '__main__':
    
    x = 2 * torch.rand(10, 2) - 1
    y = x - 0.2
    ival = Interval(y, x)
    print(ival)
    if (ival.lower < 0).any() and (ival.upper > 0).any():
        print("Contains zero") 
    # x = Interval(np.array([2, 3]), np.array([5, 7]))
    # y = Interval(np.array([11, 13]), np.array([17, 19]))
    # z = Interval(61, 218)
    
    # x = Interval(torch.randn((1, 2)), torch.randn((1, 2)))
    # w1 = torch.randn((2, 5))
    # a1 = x @ w1 
    # w2 = torch.randn((5, 3))
    # a2 = a1 @ w2
    # w3 = torch.randn((3, 1))
    # y = a2 @ w3 

    # x1 = Interval(torch.randn((1, 2)), torch.randn((1, 2)))
    # x2 = Interval(torch.randn((2, 3)), torch.randn((2, 3)))
    # print(x1 @ x2)

    # x = Interval(torch.randn((5, 3)), torch.randn((5, 3)))
    # y = w1 @ x
    # print(y)
    # print(y.width)

    # x = torch.tensor([  [-2, 2, -2, -2, 2],
    #                     [-2, 2, -2, -2, 2]], dtype=torch.float32)
    # xIval = Interval(x-1, x+1)
    # xEqIval = Interval(x, x)

    # # print( x > xIval)
    # # print(xIval >= 0 )
    # # print(x > 0)
    # print(xIval * xIval) 
    # print((x-1) * (x-1))
    # print((x-1) * (x+1))
    # print((x+1) * (x+1))
