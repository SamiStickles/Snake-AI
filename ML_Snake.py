import random
import os
import sys
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import pygame
from pygame.locals import *

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.losses import categorical_crossentropy

# test edit pls ignore

div = 20
size = 500
tileSize = size // div
fps = pygame.time.Clock()
difficulty = 15
mode = "on"

darkGreen = pygame.Color(50, 168, 82)
green = pygame.Color(35, 112, 56)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
gray = pygame.Color(180, 180, 180)
red = pygame.Color(255, 0, 0)

load_saved_pool = False
save_current_pool = False
current_pool = []
fitness = []
total_models = 20
generation = 1
highest_fitness = -1
best_weights = []

def save_pool():
    for i in range(total_models):
        current_pool[i].save_weights("SavedModels/model_new" + str(i) + ".keras")

# Inputs are food, wall, tail booleans for 1 square in front, right and left of the snake's head, so (3, 3)
# Output is an array of three probabilities, representing the 3 direction choices, highest is chosen as the move

# MODEL START

def makeModel():

    model = Sequential()
    model.add(Dense(9, input_shape=(3,), activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    adam = Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

    return model

def makePrediction(model_num):
    global current_pool
    # make variables for the left, center, right positions next to the head, the snakes "vision"
    check_left = [0, 0]
    check_center = [0, 0]
    check_right = [0, 0]

    if snake.dir == "north":
        check_left = [snake.head.pos[0] - 1, snake.head.pos[1]]
        check_center = [snake.head.pos[0], snake.head.pos[1] - 1]
        check_right = [snake.head.pos[0] + 1, snake.head.pos[1]]
    elif snake.dir == "south":
        check_left = [snake.head.pos[0] + 1, snake.head.pos[1]]
        check_center = [snake.head.pos[0], snake.head.pos[1] + 1]
        check_right = [snake.head.pos[0] - 1, snake.head.pos[1]]
    elif snake.dir == "west":
        check_left = [snake.head.pos[0], snake.head.pos[1] + 1]
        check_center = [snake.head.pos[0] - 1, snake.head.pos[1]]
        check_right = [snake.head.pos[0], snake.head.pos[1] - 1]
    elif snake.dir == "east":
        check_left = [snake.head.pos[0], snake.head.pos[1] - 1]
        check_center = [snake.head.pos[0] + 1, snake.head.pos[1]]
        check_right = [snake.head.pos[0], snake.head.pos[1] + 1]

    check_fruit = [False, False, False]
    # check if any position is the same as food's position
    if fruit.pos[0] == check_left[0] and fruit.pos[1] == check_left[1]:
        check_fruit[0] = True
    if fruit.pos[0] == check_center[0] and fruit.pos[1] == check_center[1]:
        check_fruit[1] = True
    if fruit.pos[0] == check_right[0] and fruit.pos[1] == check_right[1]:
        check_fruit[2] = True

    check_wall = [False, False, False]
    # check if any position is out of board bounds
    if check_left[0] >= div or check_left[0] < 0 or check_left[1] < 0 or check_left[1] >= div:
            check_wall[0] = True
    if check_center[0] >= div or check_center[0] < 0 or check_center[1] < 0 or check_center[1] >= div:
            check_wall[1] = True
    if check_right[0] >= div or check_right[0] < 0 or check_right[1] < 0 or check_right[1] >= div:
            check_wall[2] = True

    check_tail = [False, False, False]
    # loop through tail checking if positions are the same
    for seg in snake.tail:
        if seg.pos[0] == check_left[0] and seg.pos[1] == check_left[1]:
            check_tail[0] = True
        if seg.pos[0] == check_center[0] and seg.pos[1] == check_center[1]:
            check_tail[1] = True
        if seg.pos[0] == check_right[0] and seg.pos[1] == check_right[1]:
            check_tail[2] = True

    input = np.asarray([check_fruit, check_wall, check_tail])
    output = current_pool[model_num].predict(input, 1)
    print (output[0])

    if output[0][0] > output[0][1] and output[0][0] > output[0][2]:
        snake.moveDir = 'left'
    elif output[0][1] > output[0][0] and output[0][1] > output[0][2]:
        snake.moveDir = 'straight'
    elif output[0][2] > output[0][0] and output[0][2] > output[0][1]:
        snake.moveDir = 'right'
    else:
        snake.moveDir = random.choice(['left', 'straight', 'right'])

def crossover(parent1, parent2):
    global current_pool

    parent1_weight = current_pool[parent1].get_weights()
    parent2_weight = current_pool[parent2].get_weights()
    parent1_new_weight = parent1_weight
    parent2_new_weight = parent2_weight

    gene = random.randint(0, len(parent1_weight) - 1)

    parent1_new_weight[gene] = parent2_weight[gene]
    parent2_new_weight[gene] = parent1_weight[gene]

    return np.asarray([parent1_new_weight, parent2_new_weight])

def mutate(weights):
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if random.uniform(0, 1) >= .85:
                change = random.uniform(-0.5, 0.5)
                weights[i][j] += change
    return weights

def endGame():
    # this is where the snakes will be modified and mixed together to form the next generation
    global current_pool
    global generation
    global fitness
    global highest_fitness
    global best_weights

    new_weights = []
    new_highscore = False

    for model in range(total_models):
        # if one of the child snakes fitness is higher than the max, make it the new max
        if fitness[model] >= highest_fitness:
            new_highscore = True
            highest_fitness = fitness[model]
            best_weights = current_pool[model].get_weights()

    # Get random parents then compare with all other models by fitness to find best parents
    parent1 = random.randint(0,total_models-1)
    parent2 = random.randint(0,total_models-1)

    for model in range(total_models):
        if fitness[model] >= fitness[parent1]:
            parent1 = model

    for model in range(total_models):
        if model != parent1:
            if fitness[model] >= fitness[parent2]:
                parent2 = model

    for model in range(total_models // 2):

        # take the two parents and perform crossover and mutation on them to obtain two mixed children
        # do this 10 times to make the 20 models
        crossover_weights = crossover(parent1, parent2)

        # if new_highscore is False, then no snake was better than the previos generation, so we use last generation's best snake
        if new_highscore == False:
            crossover_weights[1] = best_weights

        mutated1 = mutate(crossover_weights[0])
        mutated2 = mutate(crossover_weights[1])

        new_weights.append(mutated1)
        new_weights.append(mutated2)

    for model in range(len(new_weights)):
        fitness[model] = 0
        current_pool[model].set_weights(new_weights[model])

    if save_current_pool == True:
        save_pool()

    generation += 1
    print (f"Gen: {generation}, Score: {len(snake.tail) - 1}")
    return

# Initialize all models
for i in range(total_models):
    model = makeModel()
    current_pool.append(model)
    fitness.append(0)

if load_saved_pool:
    for i in range(total_models):
        current_pool[i].load_weights("SavedModels/model_new"+str(i)+".keras")

# MODEL END

class Segment():
    def __init__(self, pos):
        self.pos = pos
        self.color = green

    def draw(self, surface, eyes=False):
        xPos = self.pos[0]
        yPos = self.pos[1]

        pygame.draw.rect(surface, self.color, (xPos * tileSize + 1, yPos * tileSize + 1, tileSize - 1, tileSize - 1))
        if eyes:
            radius = 2
            quarter = tileSize // 4
            threeQuarter = quarter * 3

            if snake.dir == 'south':
                leftEye = (xPos * tileSize + quarter + radius, yPos * tileSize + threeQuarter)
                rightEye = (xPos * tileSize + threeQuarter, yPos * tileSize + threeQuarter)
            elif snake.dir == 'east':
                leftEye = (xPos * tileSize + threeQuarter, yPos * tileSize + quarter + radius)
                rightEye = (xPos * tileSize + threeQuarter, yPos * tileSize + threeQuarter)
            elif snake.dir == 'west':
                leftEye = (xPos * tileSize + quarter, yPos * tileSize + threeQuarter)
                rightEye = (xPos * tileSize + quarter, yPos * tileSize + quarter + radius)
            else:
                leftEye = (xPos * tileSize + quarter + radius, yPos * tileSize + quarter)
                rightEye = (xPos * tileSize + threeQuarter, yPos * tileSize + quarter)

            pygame.draw.circle(surface, black, leftEye, radius)
            pygame.draw.circle(surface, black, rightEye, radius)

class Snake():
    def __init__(self):
        self.head = Segment([10, 10])
        self.tail = []
        self.tail.append(self.head)
        self.dir = 'north'
        self.moveDir = 'stop'
        self.dead = False
        self.winner = False
        self.color = green

    def reset(self):
        self.head = Segment([10, 10])
        self.tail = []
        self.tail.append(self.head)
        self.dir = 'north'
        self.moveDir = 'stop'
        self.dead = False
        self.winner = False

    def draw(self, surface):
        for seg in self.tail:
            if seg == self.tail[0]:
                seg.draw(surface, True)
            else:
                seg.draw(surface)

    def move(self):
        lastHeadPos = self.head.pos
        last = self.tail.pop()
        last.pos = [lastHeadPos[0], lastHeadPos[1]]
        self.tail.insert(1, last)

        if self.dir == 'north':
            if self.moveDir == 'left':
                self.head.pos[0] -= 1
                self.dir = 'west'
            if self.moveDir == 'straight':
                self.head.pos[1] -= 1
            if self.moveDir == 'right':
                self.head.pos[0] += 1
                self.dir = 'east'
        elif self.dir == 'south':
            if self.moveDir == 'left':
                self.head.pos[0] += 1
                self.dir = 'east'
            if self.moveDir == 'straight':
                self.head.pos[1] += 1
            if self.moveDir == 'right':
                self.head.pos[0] -= 1
                self.dir = 'west'
        elif self.dir == 'east':
            if self.moveDir == 'left':
                self.head.pos[1] -= 1
                self.dir = 'north'
            if self.moveDir == 'straight':
                self.head.pos[0] += 1
            if self.moveDir == 'right':
                self.head.pos[1] += 1
                self.dir = 'south'
        elif self.dir == 'west':
            if self.moveDir == 'left':
                self.head.pos[1] += 1
                self.dir = 'south'
            if self.moveDir == 'straight':
                self.head.pos[0] -= 1
            if self.moveDir == 'right':
                self.head.pos[1] -= 1
                self.dir = 'north'

    def checkDead(self):
        for seg in self.tail:
            if seg == self.head:
                pass
            else:
                if seg.pos[0] == self.head.pos[0] and seg.pos[1] == self.head.pos[1]:
                    fitness[0] -= 10
                    self.dead = True

        if self.head.pos[0] >= div or self.head.pos[0] < 0 or self.head.pos[1] < 0 or self.head.pos[1] >= div:
            fitness[0] -= 10
            self.dead = True

    def checkWin(self):
        if len(self.tail) >= div * div:
            self.dead = True
            self.winner = True

    def eatFruit(self):
        fruit.spawn()
        fitness[0] += 10

        if len(self.tail) == 1:
            if self.dir == 'north':
                firstX = self.head.pos[0]
                firstY = self.head.pos[1] + 1
            elif self.dir == 'south':
                firstX = self.head.pos[0]
                firstY = self.head.pos[1] - 1
            elif self.dir == 'west':
                firstX = self.head.pos[0] + 1
                firstY = self.head.pos[1]
            elif self.dir == 'east':
                firstX = self.head.pos[0] - 1
                firstY = self.head.pos[1]
            self.tail.append(Segment([firstX, firstY]))
        else:
            if self.tail[-1].pos[0] - 1 == self.tail[-2].pos[0]:
                newX = self.tail[-1].pos[0] + 1
                newY = self.tail[-1].pos[1]
            elif self.tail[-1].pos[0] + 1 == self.tail[-2].pos[0]:
                newX = self.tail[-1].pos[0] - 1
                newY = self.tail[-1].pos[1]
            elif self.tail[-1].pos[1] - 1 == self.tail[-2].pos[1]:
                newX = self.tail[-1].pos[0]
                newY = self.tail[-1].pos[1] + 1
            elif self.tail[-1].pos[1] + 1 == self.tail[-2].pos[1]:
                newX = self.tail[-1].pos[0]
                newY = self.tail[-1].pos[1] - 1
            self.tail.append(Segment([newX, newY]))

class Fruit():
    def __init__(self):
        self.pos = [-1, -1]
        self.color = red

    def spawn(self):
        while True:
            flag = False
            self.pos[0] = random.randint(0, div - 1)
            self.pos[1] = random.randint(0, div - 1)
            for seg in snake.tail:
                if seg.pos[0] == self.pos[0] and seg.pos[1] == self.pos[1]:
                    flag = True
            if flag:
                pass
            else:
                break

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.pos[0] * tileSize + 1, self.pos[1] * tileSize + 1, tileSize - 2, tileSize - 2))

def redrawWindow(surface):
    surface.fill((0,0,0))
    drawGrid(surface)
    snake.draw(surface)
    fruit.draw(surface)
    pygame.display.update()

def drawGrid(surface):
    x = 0
    y = 0
    for i in range(div):
        x += tileSize
        y += tileSize
        pygame.draw.line(surface, gray, (x, 0), (x, size))
        pygame.draw.line(surface, gray, (0, y), (size, y))

def main():
    pygame.init()
    pygame.display.set_caption("Snake")
    screen = pygame.display.set_mode((size, size))

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill(black)

    global snake, fruit
    snake = Snake()
    fruit = Fruit()
    fruit.spawn()

    makeModel()

    while True:
        fps.tick(difficulty)
        pygame.display.set_caption(f"Snake!   Score: {len(snake.tail) - 1}")

        makePrediction(1)
        snake.move()
        snake.checkDead()
        snake.checkWin()

        if not snake.dead:
            if snake.head.pos[0] == fruit.pos[0] and snake.head.pos[1] == fruit.pos[1]:
                snake.eatFruit()
            redrawWindow(screen)
        else:
            endGame()
            snake.reset()
            fruit.spawn()

if __name__ == '__main__': main()