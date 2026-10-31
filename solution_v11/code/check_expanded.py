from expanded_calculations import equation_features as f
assert f(r'$28 \text{sweaters} \times 4 \text{ balls/sweater} = 112 \text{balls}$')[2]==0
assert f(r'$28 \text{balls} \times 4 = 112 \text{balls}$')[1]>0
assert f(r'$28 \text{balls} \times 4 = 100 \text{balls}$')[2]>0
assert f(r'$2 \text{hours} = 120 \text{minutes}$')[2]==0
assert f(r'$\frac{1}{3} \approx 0.33$')[2]==0
assert f(r'$x+2=5$')[2]==0
assert f(r'$2+3=9$ and #### 9')[2]>0
print('passed: numeric arithmetic, unit conversion abstention, rounding and symbolic abstention')
