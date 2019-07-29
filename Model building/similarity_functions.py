from math import*
from decimal import Decimal
 
class Similarity():
 
    """ Five similarity measures function """
 
    def euclidean_distance(self,x,y):
 
        """ return euclidean distance between two lists """
 
        return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
 
    def manhattan_distance(self,x,y):
 
        """ return manhattan distance between two lists """
 
        return sum(abs(a-b) for a,b in zip(x,y))
 
    def minkowski_distance(self,x,y,p_value):
 
        """ return minkowski distance between two lists """
 
        return self.nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),
           p_value)
 
    def nth_root(self,value, n_root):
 
        """ returns the n_root of an value """
 
        root_value = 1/float(n_root)
        return round (Decimal(value) ** Decimal(root_value),3)
 
    def cosine_similarity(self,x,y):
 
        """ return cosine similarity between two lists """
 
        numerator = sum(a*b for a,b in zip(x,y))
        denominator = self.square_rooted(x)*self.square_rooted(y)
        return round(numerator/float(denominator),3)
 
    def square_rooted(self,x):
 
        """ return 3 rounded square rooted value """
 
        return round(sqrt(sum([a*a for a in x])),3)
 
    def jaccard_similarity(self,x,y):
 
        """ returns the jaccard similarity between two lists """
 
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)
 
    def exp_neg_manhattan_distance(self,x,y):
 
        """ return manhattan distance between two lists """
 
        return pow(2.7, -sum(abs(a-b) for a,b in zip(x,y)))


measure = Similarity()
eu_dist = []
cos_sim = []
man_dis = []
exp_man = []
for i in range(0, len(X_train_1['left'])):
	eu_dist.append(measure.euclidean_distance(X_train_1['left'][i], X_train_1['right'][i]))
	cos_sim.append(measure.cosine_similarity(X_train_1['left'][i], X_train_1['right'][i]))
	man_dis.append(measure.manhattan_distance(X_train_1['left'][i], X_train_1['right'][i]))
	exp_man.append(measure.exp_neg_manhattan_distance(X_train_1['left'][i], X_train_1['right'][i]))

d['eu_dist'] = eu_dist
d['cos_sim'] = cos_sim
d['man_dis'] = man_dis
d['exp_man'] = exp_man

