from relation_constraints import hypotheses

def check(q,f,expected):
 h=[x for x in hypotheses(q) if x['family']==f];assert h and h[0]['expected']==str(expected),(q,h)
check('Joe spends 3 hours learning rules and 5 times that long practicing. How much time total?', 'time_stages',18)
check('Joe spends 3 hours learning rules and 5 times that long practicing. Then he spends 2 times as much as the combined time competing. How much total time?', 'time_stages',54)
check('Ana had 12 female ducks. They laid 4 eggs each. 1/2 of the eggs hatched. How many ducks does Ana have now?', 'hatching_stock',36)
check('Ana had 12 female ducks. They laid 4 eggs each. 50% of the eggs hatched. How many ducks does Ana have now?', 'hatching_stock',36)
check('The fish increased by 3 times the number initially there. There were 20 fish initially. How many fish now?', 'growth_by',80)
check('Joe had 12 apples. He bought 3 apples. He ate 2 apples. How many apples does Joe have now?', 'stock_events',13)
assert not hypotheses('Joe had 12 apples. Sam bought 3 apples. How many apples does Joe have now?')
assert not hypotheses('Joe had 12 apples. He bought 3 oranges. How many apples does Joe have now?')
assert not hypotheses('Ana had 12 ducks. They laid 4 eggs each. 1/2 of the eggs hatched. How many eggs hatched?')
assert not hypotheses('Joe spends 3 hours learning rules and 5 times that long practicing every day. How much time total?')
print('constraint checks passed')
