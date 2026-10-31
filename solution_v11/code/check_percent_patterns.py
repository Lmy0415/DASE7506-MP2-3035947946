from percent_patterns import hypotheses

def get(q,f):return [v for k,v in hypotheses(q) if k==f]
assert get('A haircut costs $18. The weekend price is 50% more expensive. How much would the haircut cost?', 'percent_increase')==['27']
assert get('A shirt costs $60. The store gives a 25% discount. How much will she pay?', 'percent_decrease')==['45']
assert not hypotheses('A shirt costs $60. The store gives a 25% discount. How much will she save?')
assert not hypotheses('A shirt costs $60. The store gives a 25% discount and 10% sales tax. How much will she pay?')
assert not hypotheses('A shirt costs $60. The store gives a 25% discount. Another shirt costs $20. How much will she pay?')
print('passed')
