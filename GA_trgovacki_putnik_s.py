# -*- coding: utf-8 -*-

import sys
import os
import random
import math
import xml.etree.ElementTree as ET
import numpy as np
import csv
import statistics

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtChart import QLineSeries, QChart, QValueAxis, QChartView
from PyQt5.QtWidgets import QFileDialog
from sympy import Triangle, Point, Segment, intersection

from deap import base, creator, tools

#Change current working directory to this script path
import pathlib
os.chdir(pathlib.Path(__file__).parent.absolute())


####Global GA parameters####
IND_SIZE = 0 #Size of the individual (number of cities)
NGEN = 5000 #number of generations
POP_SIZE = 50  #population size
MUTPB = 0.05 #probability for mutating an individual
NELT = 5    #number of elite individuals
POPULATION_SIZES = [50, 100, 200, 400]
MUTATION_SIZES = [0.05, 0.1, 0.15, 0.2]
ELITE_SIZES = [5, 10, 15, 20]
#########################

####Other global variables####
stop_evolution = False
q_min_series = QLineSeries()
q_min_series.setName("MIN")
q_max_series = QLineSeries()
q_max_series.setName("MAX")
q_avg_series = QLineSeries()
q_avg_series.setName("AVG")
croatia_map_img = QImage("Croatia620.png")
gradovi = []
sirine = []
duzine = []
border_check = False
combination_series = []
best_fit_values = []
best_individual = []
parameter_name = "original"
##############################


#Load the list of cities
tree = ET.parse('gradovi.xml')
root = tree.getroot()
for child in root:
    gradovi.append(str(child.attrib['ime_grada']))
    sirine.append(float(child.attrib['sirina']))
    duzine.append(float(child.attrib['duzina']))

#Set the number of cities when they have been parsed
IND_SIZE = len(gradovi)


# Defining border lines -- Added by Denis Lazor
L1 = ([15.98, 44.9], [19, 44.9])
L2 = ([15.98, 44.9], [18.5, 42.7])


# Checking if 2 segments are intersecting -- https://kite.com/python/answers/how-to-check-if-two-line-segments-intersect-in-python
def orientation(p, q, r):
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val == 0:
        return 0
    return 1 if val > 0 else -1

def intersects(seg1, seg2):

    p1, q1 = seg1
    p2, q2 = seg2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True
    else: return False

# Define evaluation (fitness) function for individual (cromosome)
def evaluateInd(individual):
    fit_val = 0.0 #starting fitness is 0

    # Fitness function -- Added by Denis Lazor
    for idx, grad in enumerate(individual):

        if idx > 0:
            sirina = abs(sirine[grad] - sirine[prethodni]) * 110.64
            duzina = abs(duzine[grad] - duzine[prethodni]) * 78.85
            fit_val = fit_val + math.sqrt(pow(sirina, 2) + pow(duzina, 2))

            if border_check:
                path_line = ([duzine[grad], sirine[grad]], [duzine[prethodni], sirine[prethodni]])

                if intersects(L1, path_line) or intersects(L2, path_line):  #  Cheking if path is going over border line -- Added by Denis Lazor
                    fit_val = fit_val + 5000
        prethodni = grad

    return fit_val, #returning must be a tuple becos of posibility of optimization via multiple goal values (objectives)

def GlobToImgCoords(coord_x, coord_y):
    stupnjevi_1 = math.floor(coord_x)
    minute_1 = round((coord_x - math.floor(coord_x)) * 60)
    stupnjevi_2 = math.floor(coord_y)
    minute_2 = round((coord_y - math.floor(coord_y)) * 60)
    
    kor_x = 0
    kor_y = 0
    if stupnjevi_2 > 13:
        kor_x = ((stupnjevi_2 - (14)) * 60) + (minute_2 + 54)
    else:
        kor_x = minute_2 - 6

    if stupnjevi_1 < 46:
        kor_y = (((46 - (stupnjevi_1 + 1)) * 60) + (48 + (60 - minute_1)))
    else:
        kor_y = (48 - minute_1)

    kor_x = kor_x + math.floor(kor_x * 0.52)
    kor_y = (kor_y * 2) + math.floor(kor_y * 0.12)

    return kor_x, kor_y

def generateWorldImage(individual):
    #Create a transparent image
    img = QImage(620, 600, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    
    #Create a painter
    painter = QPainter(img)
    
    #Highlight first and last town
    g_first = individual[0]
    g_last = individual[IND_SIZE - 1]
    x1, y1 = GlobToImgCoords(sirine[g_first], duzine[g_first])
    x2, y2 = GlobToImgCoords(sirine[g_last], duzine[g_last])
    painter.setBrush(Qt.green)
    painter.drawEllipse(x1-10, y1-10, 15, 15)
    painter.setBrush(Qt.blue)
    painter.drawEllipse(x2-10, y2-10, 15, 15)
    
    #Drawing Path
    painter.setPen(QPen(Qt.black,  3, Qt.DashLine))
    for i in range(IND_SIZE - 1): #
        x1, y1 = GlobToImgCoords(sirine[individual[i]], duzine[individual[i]])
        x2, y2 = GlobToImgCoords(sirine[individual[i + 1]], duzine[individual[i + 1]])
        painter.drawLine(x1, y1, x2, y2)
    
    #Finish painter
    painter.end()
    
    #Return finished image
    return img


# Print function added by -- Denis Lazor
def print_results(p, m, e, best_fit, mean_fit,  NGEN):
    print("Combination: " + "  Population = " + str(p) + ", Mutation = " + str(m*100) + "%, Elitism = " + str(e) + ", Generations = " + str(NGEN))
    print("Mean fitness: " + str(mean_fit) + "\nBest solution:: " + str(best_fit) + "\n")


# Clearing all .csv files on program start -- Added by Denis Lazor
def clear_all_csv():
    directory = 'graphs_csv/'

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            f = open("graphs_csv/" + filename, 'w')
            f.truncate()
            continue
        else:
            continue


# Writing min_series for mean generation to .csv files. Added by -- Denis Lazor
def write_to_file(series, parameter_name):

    with open(("graphs_csv/" + parameter_name + ".csv"), "a") as file:
        writer = csv.writer(file, lineterminator='\n')
        iterations = []
        fitness_values = []
        for i in range(len(series)):
            iterations.append(i)
            fitness_values.append(series[i])
        writer.writerow(iterations)
        writer.writerow(fitness_values)

        

class MyQFrame(QtWidgets.QFrame):
    def paintEvent(self, event):
        painterWorld = QPainter(self)
        painterWorld.drawPixmap(self.rect(), self.img)
        painterWorld.end()

class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(850, 1080)
        self.setWindowTitle("GA - Queens")
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.frameWorld = MyQFrame(self.centralwidget)
        self.frameWorld.img = QPixmap(1000,1000)
        self.frameWorld.setGeometry(QtCore.QRect(10, 10, 620, 600))
        self.frameWorld.setFrameShape(QtWidgets.QFrame.Box)
        self.frameWorld.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frameWorld.setObjectName("frameWorld")
        self.frameChart = QChartView(self.centralwidget)
        self.frameChart.setGeometry(QtCore.QRect(10, 620, 620, 400))
        self.frameChart.setFrameShape(QtWidgets.QFrame.Box)
        self.frameChart.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frameChart.setRenderHint(QPainter.Antialiasing)
        self.frameChart.setObjectName("frameChart")
        self.gaParams = QtWidgets.QGroupBox(self.centralwidget)
        self.gaParams.setGeometry(QtCore.QRect(650, 10, 161, 145))
        self.gaParams.setObjectName("gaParams")
        self.gaParams.setTitle("GA parameters")
        self.label1 = QtWidgets.QLabel(self.gaParams)
        self.label1.setGeometry(QtCore.QRect(10, 20, 61, 16))
        self.label1.setObjectName("label1")
        self.label1.setText("Population:")
        self.label2 = QtWidgets.QLabel(self.gaParams)
        self.label2.setGeometry(QtCore.QRect(10, 50, 47, 16))
        self.label2.setObjectName("label2")
        self.label2.setText("Mutation:")
        self.label3 = QtWidgets.QLabel(self.gaParams)
        self.label3.setGeometry(QtCore.QRect(10, 80, 81, 16))
        self.label3.setObjectName("label3")
        self.label3.setText("Elite members:")
        self.label4 = QtWidgets.QLabel(self.gaParams)
        self.label4.setGeometry(QtCore.QRect(10, 110, 91, 16))
        self.label4.setObjectName("label4")
        self.label4.setText("No. generations:")
        self.tbxPopulation = QtWidgets.QLineEdit(self.gaParams)
        self.tbxPopulation.setGeometry(QtCore.QRect(100, 20, 51, 20))
        self.tbxPopulation.setObjectName("tbxPopulation")
        self.tbxMutation = QtWidgets.QLineEdit(self.gaParams)
        self.tbxMutation.setGeometry(QtCore.QRect(100, 50, 51, 20))
        self.tbxMutation.setObjectName("tbxMutation")
        self.tbxElite = QtWidgets.QLineEdit(self.gaParams)
        self.tbxElite.setGeometry(QtCore.QRect(100, 80, 51, 20))
        self.tbxElite.setObjectName("tbxElite")
        self.tbxGenerations = QtWidgets.QLineEdit(self.gaParams)
        self.tbxGenerations.setGeometry(QtCore.QRect(100, 110, 51, 20))
        self.tbxGenerations.setObjectName("tbxGenerations")
        self.cbxNoVis = QtWidgets.QCheckBox(self.centralwidget)
        self.cbxNoVis.setGeometry(QtCore.QRect(650, 170, 170, 17))
        self.cbxNoVis.setObjectName("cbxNoVis")
        self.cbxNoVis.setText("No visualization per generation")
        self.cbxBorder = QtWidgets.QCheckBox(self.centralwidget)
        self.cbxBorder.setGeometry(QtCore.QRect(650, 200, 100, 17))
        self.cbxBorder.setObjectName("cbxBorder")
        self.cbxBorder.setText("Border patrol")
        self.btnStart = QtWidgets.QPushButton(self.centralwidget)
        self.btnStart.setGeometry(QtCore.QRect(650, 230, 75, 23))
        self.btnStart.setObjectName("btnStart")
        self.btnStart.setText("Start")
        self.btnStop = QtWidgets.QPushButton(self.centralwidget)
        self.btnStop.setEnabled(False)
        self.btnStop.setGeometry(QtCore.QRect(730, 230, 75, 23))
        self.btnStop.setObjectName("btnStop")
        self.btnStop.setText("Stop")
        self.btnSaveWorld = QtWidgets.QPushButton(self.centralwidget)
        self.btnSaveWorld.setGeometry(QtCore.QRect(650, 570, 121, 41))
        self.btnSaveWorld.setObjectName("btnSaveWorld")
        self.btnSaveWorld.setText("Save world as image")
        self.btnSaveChart = QtWidgets.QPushButton(self.centralwidget)
        self.btnSaveChart.setGeometry(QtCore.QRect(650, 930, 121, 41))
        self.btnSaveChart.setObjectName("btnSaveChart")
        self.btnSaveChart.setText("Save chart as image")
        self.btnSaveChartSeries = QtWidgets.QPushButton(self.centralwidget)
        self.btnSaveChartSeries.setGeometry(QtCore.QRect(650, 980, 121, 41))
        self.btnSaveChartSeries.setObjectName("btnSaveChartSeries")
        self.btnSaveChartSeries.setText("Save chart as series")
        self.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(self)
        
        #Connect events
        self.btnStart.clicked.connect(self.btnStart_Click)
        self.btnStop.clicked.connect(self.btnStop_Click)
        self.btnSaveWorld.clicked.connect(self.btnSaveWorld_Click)
        self.btnSaveChart.clicked.connect(self.btnSaveChart_CLick)
        self.btnSaveChartSeries.clicked.connect(self.btnSaveChartSeries_Click)
        
        #Set default GA variables
        self.tbxGenerations.insert(str(NGEN))
        self.tbxPopulation.insert(str(POP_SIZE))
        self.tbxMutation.insert(str(MUTPB))
        self.tbxElite.insert(str(NELT))
        
        self.new_image = QPixmap(1000,1000)


    def btnStart_Click(self):

        global combination_series  # List of lists containing min_series of 5 correct results -- Added by Denis Lazor
        global parameter_name  # Name of parameter used for writing its data to .csv file -- Added by Denis Lazor
        global best_fit_values  # List containing best fitness values for every of 5 experiments per combination -- Added by Denis Lazor
        global best_individual  # Saving best individuals for drawing the best solution -- Added by Denis Lazor

        global ELITE_SIZES
        global POPULATION_SIZES
        global MUTATION_SIZES

        # Checking if files are empty or not -- Added by Denis Lazor
        csv_contains = os.stat("graphs_csv/original.csv").st_size != 0

        if csv_contains:
            clear_all_csv()

        n = 5000
        # Automation for all necessary combinations -- Added by Denis Lazor

        for p in POPULATION_SIZES:
            for m in MUTATION_SIZES:
                for e in ELITE_SIZES:
                    for i in range(5):


                        # Set global variables
                        global stop_evolution
                        global q_min_series
                        global q_max_series
                        global q_avg_series
                        stop_evolution = False
                        q_min_series.clear()
                        q_max_series.clear()
                        q_avg_series.clear()

                        # Set global variables from information on UI
                        global NGEN
                        global POP_SIZE
                        global MUTPB
                        global NELT
                        NGEN = n
                        POP_SIZE = p
                        MUTPB = m
                        NELT = e
                        global border_check
                        border_check = self.cbxBorder.isChecked()

                        # Loading Croatia map
                        self.img = QPixmap(620, 600)
                        self.img.load('Croatia620.png')
                        self.frameWorld.img = self.img
                        # Drawing towns
                        painter = QPainter(self.img)
                        painter.setPen(QPen(Qt.black, 10, Qt.SolidLine))
                        painter.setFont(QFont('Arial', 12))
                        for i in range(len(gradovi)):
                            x, y = GlobToImgCoords(sirine[i], duzine[i])
                            painter.drawPoint(x, y)
                            painter.drawText(x + 5, y + 5, gradovi[i])

                        painter.end()
                        # Redrawing frames
                        self.frameWorld.repaint()
                        app.processEvents()

                        ####Initialize deap GA objects####

                        # Make creator that minimize. If it would be 1.0 instead od -1.0 than it would be maxmize
                        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

                        # Create an individual (a blueprint for cromosomes) as a list with a specified fitness type
                        creator.create("Individual", list, fitness=creator.FitnessMin)

                        # Create base toolbox for finishing creation of a individual (cromosome)
                        self.toolbox = base.Toolbox()

                        # This is if we want a permutation coding of genes in the cromosome
                        self.toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)

                        # initIterate requires that the generator of genes (such as random.sample) generates an iterable (a list) variable
                        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices)

                        # Create a population of individuals (cromosomes). The population is then created by toolbox.population(n=300) where 'n' is the number of cromosomes in population
                        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

                        # Register evaluation function
                        self.toolbox.register("evaluate", evaluateInd)

                        # Register what genetic operators to use
                        self.toolbox.register("mate", tools.cxUniformPartialyMatched,
                                              indpb=0.2)  # Use uniform recombination for permutation coding

                        # Permutation coding
                        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)

                        self.toolbox.register("select", tools.selTournament, tournsize=3)  # Use tournament selection

                        ##################################

                        # Generate initial poplation. Will be a member variable so we can easely pass everything to new thread
                        self.pop = self.toolbox.population(n=POP_SIZE)

                        # Evaluate initial population, we map() the evaluation function to every individual and then assign their respective fitness, map runs evaluate function for each individual in pop
                        fitnesses = list(map(self.toolbox.evaluate, self.pop))
                        for ind, fit in zip(self.pop, fitnesses):
                            ind.fitness.values = fit  # Assign calcualted fitness value to individuals

                        # Extracting all the fitnesses of all individuals in a population so we can monitor and evovlve the algorithm until it reaches 0 or max number of generation is reached
                        self.fits = [ind.fitness.values[0] for ind in self.pop]

                        # Disable start and enable stop
                        self.btnStart.setEnabled(False)
                        self.btnStop.setEnabled(True)
                        self.gaParams.setEnabled(False)
                        self.cbxBorder.setEnabled(False)
                        self.cbxNoVis.setEnabled(False)

                        # Start evolution
                        self.evolve()



                    # Best fitness value -- Added by Denis Lazor
                    best_fit = np.array(min(best_fit_values))[0]
                    mean_fit = min(best_fit_values, key=lambda x: abs(x - statistics.mean((np.asarray(best_fit_values)).flatten())))[0]

                    # Index of best fitness value -- Added by Denis Lazor
                    best_fit_idx = best_fit_values.index(best_fit)

                    write_to_file(combination_series[best_fit_idx], parameter_name)

                    # First name will be "original", second one "elites" -- Added by Denis Lazor
                    parameter_name = "elites"

                    print_results(p, m, e, best_fit, mean_fit, NGEN)

                    # Clearing past lists  -- Added by Denis Lazor
                    combination_series = []
                    best_fit_values = []

                # Reducing number of combinations and changing .csv file for writing -- Added by Denis Lazor
                ELITE_SIZES = ELITE_SIZES[0:1]
                parameter_name = "mutation"

            MUTATION_SIZES = MUTATION_SIZES[0:1]
            parameter_name = "population"

        print("Best individual: " + str(best_individual))
        self.updateWorldFrame(generateWorldImage(best_individual))  # Drawing best solution -- Added by Denis Lazor

    def btnStop_Click(self):
        global stop_evolution
        stop_evolution = True
        #Disable stop and enable start
        self.btnStop.setEnabled(False)
        self.btnStart.setEnabled(True)
        self.gaParams.setEnabled(True)
        self.cbxBorder.setEnabled(True)
        self.cbxNoVis.setEnabled(True)
    
    #Function for GA evolution
    def evolve(self):
        global q_min_series
        global q_max_series
        global q_avg_series

        global best_fit_values
        global combination_series
        global best_individual

        combination_current_series = []  # Clearing fitness values series -- Added by Denis Lazor


        # Variable for keeping track of the number of generations
        curr_g = 0
        
        # Begin the evolution till goal is reached or max number of generation is reached
        while min(self.fits) != 0 and curr_g < NGEN:
            #Check if evolution and thread need to stop
            if stop_evolution:
                break #Break the evolution loop
            
            # A new generation
            curr_g = curr_g + 1
            #print("-- Generation %i --" % curr_g)
            
            # Select the next generation individuals
            #Select POP_SIZE - NELT number of individuals. Since recombination is between neigbours, not two naighbours should be the clone of the same individual
            offspring = []
            offspring.append(self.toolbox.select(self.pop, 1)[0])    #add first selected individual
            for i in range(POP_SIZE - NELT - 1):    # -1 because the first seleceted individual is already added
                while True:
                    new_o = self.toolbox.select(self.pop, 1)[0]
                    if new_o != offspring[len(offspring) - 1]:   #if it is different than the last inserted then add to offspring and break
                        offspring.append(new_o)
                        break
            
            # Clone the selected individuals because all of the changes are inplace
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover on the selected offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                self.toolbox.mate(child1, child2)    #inplace recombination
                #Invalidate new children fitness values
                del child1.fitness.values
                del child2.fitness.values
    
            #Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            #Add elite individuals #Is clonning needed?
            offspring.extend(list(map(self.toolbox.clone, tools.selBest(self.pop, NELT))))         
                    
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            #print("  Evaluated %i individuals" % len(invalid_ind))
            
            #Replace population with offspring
            self.pop[:] = offspring
            
            # Gather all the fitnesses in one list and print the stats
            self.fits = [ind.fitness.values[0] for ind in self.pop]
            
            length = len(self.pop)
            mean = sum(self.fits) / length
            sum2 = sum(x*x for x in self.fits)
            std = abs(sum2 / length - mean**2)**0.5

            q_min_series.append(curr_g, min(self.fits))
            q_max_series.append(curr_g, max(self.fits))
            q_avg_series.append(curr_g, mean)

            combination_current_series.append(min(self.fits))  # Saving min_series fitness values of an experiment -- Added by Denis Lazor

                      
            #print("  Min %s" % q_min_series.at(q_min_series.count()-1).y())
            #print("  Max %s" % q_max_series.at(q_max_series.count()-1).y())
            #print("  Avg %s" % mean)
            #print("  Std %s" % std)
            
            if self.cbxNoVis.isChecked():
                app.processEvents()
            else:
                self.chart = QChart()
                self.chart.addSeries(q_min_series)
                self.chart.addSeries(q_max_series)
                self.chart.addSeries(q_avg_series)
                self.chart.setTitle("Fitness value over time")
                self.chart.setAnimationOptions(QChart.NoAnimation)
                self.chart.createDefaultAxes()
                self.frameChart.setChart(self.chart)
                
                #Draw queen positions of best individual on a image
                best_ind = tools.selBest(self.pop, 1)[0]
                self.updateWorldFrame(generateWorldImage(best_ind))

                      
        #Printing best individual
        best_ind = tools.selBest(self.pop, 1)[0]
        #print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

        combination_series.append(combination_current_series)  # Saving best 5 results -- Added by Denis Lazor

        if not best_individual:
            best_individual = best_ind

        elif best_ind.fitness.values < best_individual.fitness.values:  # Saving best individual in all combinations -- Added by Denis Lazor
            best_individual = best_ind


        best_fit_values.append(best_ind.fitness.values)  # Adding best fitness value of experiment -- Added by Denis Lazor

        
        # #Visulaize final solution
        # if self.cbxNoVis.isChecked():
        #     self.chart = QChart()
        #     self.chart.addSeries(q_min_series)
        #     self.chart.addSeries(q_max_series)
        #     self.chart.addSeries(q_avg_series)
        #     self.chart.setTitle("Fitness value over time")
        #     self.chart.setAnimationOptions(QChart.NoAnimation)
        #     self.chart.createDefaultAxes()
        #     self.frameChart.setChart(self.chart)
        #
        #     #Draw queen positions of best individual on a image
        #     best_ind = tools.selBest(self.pop, 1)[0]
        #     self.updateWorldFrame(generateWorldImage(best_ind))
        
        #Disable stop and enable start
        self.btnStop.setEnabled(False)
        self.btnStart.setEnabled(True)
        self.gaParams.setEnabled(True)
        self.cbxBorder.setEnabled(True)
        self.cbxNoVis.setEnabled(True)
        
    def updateWorldFrame(self, best_individual_img):
        #new_image = QPixmap(1000,1000)
        self.new_image.fill() #White color is default
        painter = QPainter(self.new_image)
        #First draw the map with towns
        painter.drawPixmap(self.new_image.rect(), self.img)
        #Then draw the best individual
        painter.drawImage(self.new_image.rect(), best_individual_img)
        painter.end()
        #Set new image to the frame
        self.frameWorld.img = self.new_image
        #Redrawing frames
        self.frameWorld.repaint()
        self.frameChart.repaint()
        app.processEvents()
    
    def btnSaveWorld_Click(self):
        filename, _ = QFileDialog.getSaveFileName(None,"Save world as a image","","Image Files (*.png)")
        self.frameWorld.img.save(filename, "PNG")
        print ("World image saved to: ", filename)
    
    def btnSaveChart_CLick(self):
        p = self.frameChart.grab()
        filename, _ = QFileDialog.getSaveFileName(None,"Save series chart as a image","","Image Files (*.png)")
        p.save(filename, "PNG")
        print ("Chart series image saved to: ", filename)
    
    def btnSaveChartSeries_Click(self):
        global q_min_series
        global q_max_series
        global q_avg_series
        filename, _ = QFileDialog.getSaveFileName(None,"Save series to text file","","Text Files (*.txt, *.csv)")
        with open(filename, 'w') as dat:
            for i in range(q_min_series.count()):
                dat.write('%f,%f,%f\n' % (q_min_series.at(i).y(), q_avg_series.at(i).y(), q_max_series.at(i).y()))
        print ("Chart series saved to: ", filename)
    


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    ui = Ui_MainWindow()
    ui.setupUi()
    ui.show()
    sys.exit(app.exec_())
