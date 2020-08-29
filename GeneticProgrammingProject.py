import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import matplotlib.pyplot as plt

# Zabezpieczone dzielenie
def div(left, right):
    if right == 0:
        return 1
    return left / right

def numFromStr(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def sqrt(a):
    if a <= 0:
        return 0
    return math.sqrt(a)

def root_find(f,a,b,step):

    result = []
    step2 = math.ceil((b-a)/step)
    print ("\nPrzedzial:[",a,",",b,"]","krok: ",step,"\n")
    for i in range(step2):
        if f(a)*f(round(a+step,3)) <= 0:
            result.append(a)
           #print("find\n")
        a += step
        a = round(a,3)
        #print (a,b,f(a),f(round(a+step,3)),round(a+step,3),f(a)*f(round(a+step,3)))
    return result

# Zbior funkcji i terminali
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(operator.abs, 1)
pset.addPrimitive(max, 2)
pset.addPrimitive(min, 2)
pset.addPrimitive(sqrt, 1)

pset.addTerminal(math.pi)
pset.addTerminal(math.e)
pset.addTerminal(math.tau)
for x in range(-10,10):
    pset.addTerminal(x)

pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))                     # obiekt funckji dopasowania, waga ujemna bo minimalizujemy
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)      # obiekt osobnika

# rejestrowanie parametrow dla procesu ewolucji
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)          # jak tworzyc osobnika- polowa GenFull (równa wysokość drzewa), a polowa GenGrow(rozna wysokosc drzewa)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)      # jak tworzyc populacje - poprzez powtarzanie tworzenia osobnika
toolbox.register("compile", gp.compile, pset=pset)

evaluate = input("Podaj funkcję (od argumentu x)\n    y =  ")
from1 = int(input("\nDyskretyzacja funkcji:\n\nPodaj początek przedziału: [int]\n"))
elem_num = int(input("\nPodaj liczbę elementów: [int]\n"))
step1 = float(input("\nPodaj krok: [float]\n"))


def evalSymbReg(individual, points):

    func = toolbox.compile(expr=individual)     # przeksztalca drzewo w funkcje

    sqerrors = ((func(x) - eval(evaluate))**2 for x in points) # sredni blad kwadratowy

    return math.fsum(sqerrors) / len(points),

toolbox.register("evaluate", evalSymbReg, points=[from1 + (x * step1) for x in range(0,elem_num+1)])     # rejestracja funkcji
# wypisanie zdyskretyzowanych wartosci
temp_points1=["{0:.2f}".format(from1 + (x * step1)) for x in range(0,elem_num+1)]
print("\nArgumenty:\n",temp_points1,"\n")
temp_points2=[]
for y in range(0,elem_num+1):
    x = from1 + (y * step1)
    temp_points2 += ["{0:.2f}".format(eval(evaluate))]
print("\nWartosci:\n",temp_points2,"\n")

toolbox.register("select", tools.selTournament, tournsize=3)    # selekcja turniejowa z rozmiarem turnieju
toolbox.register("mate", gp.cxOnePoint)                 # krzyzowanie w jednym punkcie
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)    # drzewo zbudowane na potrzeby mutacji przy pomocy GenFull(rowna wysokosc drzewa)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)     # mutacja z jednolitym prawdopodobienstwem 

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))       # ograniczenie wysokosci osobnika po krzyzowaniu
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))     # ograniczenie wysokosci osobnika po mutacji

toolbox.decorate("mate", gp.staticLimit(key=len, max_value=elem_num*3))       # ograniczenie rozmiaru osobnika po krzyzowaniu
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=elem_num*3))     # ograniczenie rozmiaru osobnika po mutacji

def main():
    random.seed(318)        # do identycznego losowania

    pop_size = int(input("Podaj wielkość populacji:  [int]\n")) 
    print("\n")

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    # obiekt obliczajacy statystyki: srednia, odchylenie standardowe, min i max dla funkcji dopasowania i rozmiaru generacji
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_depth = tools.Statistics(operator.attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, depth=stats_depth)
    mstats.register("avg", numpy.mean)
    #mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    gen_num = int(input("Podaj liczbę generacji:  [int]\n"))
    print("\n")

    # ewolucja na podstawie populacji, operatorow genetycznych, prawdop. krzyzowania i mutacji, liczby generacji, obiektu do aktualizacji statystyk
    pop, log = algorithms.eaSimple(pop, toolbox, 0.6, 0.3, gen_num, stats=mstats,
                                  halloffame=hof, verbose=True)         # halloffame - najlepszy osobnik z populacji, verbose - wypisywanie statystyk

    print("\nNajlepszy osobnik:\n\t y = ", hof[0], "\n")
    print("\nWartosc funkcji dopasowania:\t", evalSymbReg(hof[0],points=[from1 + (x * step1) for x in range(0,elem_num+1)])[0], "\n")

    func1 = toolbox.compile(hof[0])
    print("\n\tSZUKANIE MIEJSCA ZEROWEGO...\n")
    root = root_find(func1,math.floor(float(temp_points1[0])),math.ceil(float(temp_points1[len(temp_points1)-1])),0.03)
    print("\nMiejsca zerowe funkcji  +/- 0.003 :\n\n\t", numpy.around(root,decimals=4), "\n")

    x1 = numpy.linspace(from1-1,from1+elem_num*step1+1,100)
    y1 = []
    for i in x1:
        y1.append(func1(i))
    plt.plot([float(i) for i in temp_points1],[float(i) for i in temp_points2], label = "wejście")
    plt.plot(x1,y1, label = "wyjście")
    plt.legend(loc='lower right')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Regresja")
    plt.show()

    return pop, log, hof

if __name__ == "__main__":
    main()


