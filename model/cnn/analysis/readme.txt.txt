Some clearification on the filename:
1. cnn2.py runs in 3 inconsistent try but based on the formal(1g, 2g, 3g).
1g has 30 ieterations before terminate on day 1
2g has 60 ieterations before terminate on day 2, total of 30+60=90
3g has 60 ieterations before terminate on day 3, total of 90+60=150
So for example "cnn2-2g-48iter-best.csv" runs 48 iterations on second try, which is 48+30=78 iterations in total
2. "iter" means iteraion, and "valacc" means validate accuracy.