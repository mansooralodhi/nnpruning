


"""
f = R(R(R(x.w1 + b1).w2 + b2).w3 + b3)

forward-pass
------------
v1 = x
v2 = w1
v3 = x.w1
v4 = b1
v5 = v3 + v4
v6 = R
v7 = v6(v5)
v8 = w2
v9 = v7.v8
v10 = b2
v11 = v9 + v10
v13 = v6(v11)
v14 = w3
v15 = v13.v14
v16 = b3
v17 = v15 + v16
v18 = v6(v17)       
v19 = R(v18)        1

backward-pass
-------------

"""