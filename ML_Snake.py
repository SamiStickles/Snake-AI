import random
import os
import sys
import time
import numpy as np
import math

import pygame
from pygame.locals import *

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.losses import categorical_crossentropy

div = 20
size = 500
tileSize = size // div
fps = pygame.time.Clock()
difficulty = 80
mode = "on"

darkGreen = pygame.Color(50, 168, 82)
green = pygame.Color(35, 112, 56)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
gray = pygame.Color(180, 180, 180)
red = pygame.Color(255, 0, 0)

display_game = True
load_saved_pop = False
save_current_pop = False
current_pop = []
fitness = []
total_models = 50
generation = 1
highest_fitness = -1
best_weights = []
random_factor = 82

def save_pop():
    for i in range(total_models):
        current_pop[i].save_weights("Saved/model" + str(i) + ".keras")

# Inputs are food, wall, tail booleans for 1 square in front, right and left of the snake's head, so (3, 3)
# Output is an array of three probabilities, representing the 3 direction choices, highest is chosen as the move

def makeModel():

    model = Sequential()
    model.add(Dense(5, input_shape=(1,), activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    adam = Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

    return model

def makePrediction(model):
    global current_pop, div, random_factor

    # random factor decreases each generation, gen 1: 100%, gen 40: 0% -2% each gen
    # if random check is less than random factor, a random move is done instead of predicting one
    random_factor = 82 - 2 * generation
    random_check = random.randint(1, 100)
    if random_check < random_factor:
        snake.moveDir = random.choice(['left', 'straight', 'right'])
    else:
        # calc distance to fruit
        x = snake.head.pos[0] - fruit.pos[0]
        y = snake.head.pos[1] - fruit.pos[1]
        snake.fruit_distance = math.sqrt((x ** 2) + (y ** 2))

        # calc angle in radians to fruit
        if snake.dir == 'north' or snake.dir == 'south':
            snake.fruit_angle = math.asin(y / snake.fruit_distance)
        else:
            snake.fruit_angle = math.asin(x / snake.fruit_distance)

        # changes the negative radians to positive by adding 2pi
        if snake.fruit_angle < 0:
            snake.fruit_angle = snake.fruit_angle + 2*math.pi

        # calc distance to tail if it is in path, if not, distance to wall
        left_danger = []
        straight_danger = []
        right_danger = []

        # set danger vars to distance to wall, then check if any tail segments are in between
        # if so, set the corresponding danger var to the closest distance
        if snake.dir == 'north':
            straight_danger = [snake.head.pos[1]]
            left_danger = [snake.head.pos[0]]
            right_danger = [div - snake.head.pos[0]]

            for seg in snake.tail[1:]:

                for i in range(straight_danger[0]):
                    if seg.pos[0] == snake.head.pos[0] and seg.pos[1] == i:
                        straight_danger.append(snake.head.pos[1] - i)

                for i in range(left_danger[0]):
                    if seg.pos[1] == snake.head.pos[1] and seg.pos[0] == i:
                        left_danger.append(snake.head.pos[0] - i)

                for i in range(right_danger[0], div):
                    if seg.pos[1] == snake.head.pos[1] and seg.pos[0] == i:
                        right_danger.append(i - snake.head.pos[0])

        elif snake.dir == 'south':
            straight_danger = [div - snake.head.pos[1]]
            left_danger = [div - snake.head.pos[0]]
            right_danger = [snake.head.pos[0]]

            for seg in snake.tail[1:]:

                for i in range(straight_danger[0], div):
                    if seg.pos[0] == snake.head.pos[0] and seg.pos[1] == i:
                        straight_danger.append(i - snake.head.pos[1])

                for i in range(left_danger[0], div):
                    if seg.pos[1] == snake.head.pos[1] and seg.pos[0] == i:
                        left_danger.append(i - snake.head.pos[0])

                for i in range(right_danger[0]):
                    if seg.pos[1] == snake.head.pos[1] and seg.pos[0] == i:
                        right_danger.append(snake.head.pos[0] - i)

        elif snake.dir == 'east':
            straight_danger = [div - snake.head.pos[0]]
            left_danger = [snake.head.pos[1]]
            right_danger = [div - snake.head.pos[1]]

            for seg in snake.tail[1:]:

                for i in range(straight_danger[0], div):
                    if seg.pos[1] == snake.head.pos[1] and seg.pos[0] == i:
                        straight_danger.append(i - snake.head.pos[0])

                for i in range(left_danger[0]):
                    if seg.pos[0] == snake.head.pos[0] and seg.pos[1] == i:
                        left_danger.append(snake.head.pos[1] - i)

                for i in range(right_danger[0], div):
                    if seg.pos[0] == snake.head.pos[0] and seg.pos[1] == i:
                        right_danger.append(i - snake.head.pos[1])

        elif snake.dir == 'west':
            straight_danger = [snake.head.pos[0]]
            left_danger = [div - snake.head.pos[1]]
            right_danger = [snake.head.pos[1]]

            for seg in snake.tail[1:]:

                for i in range(straight_danger[0]):
                    if seg.pos[1] == snake.head.pos[1] and seg.pos[0] == i:
                        straight_danger.append(snake.head.pos[0] - i)

                for i in range(left_danger[0], div):
                    if seg.pos[0] == snake.head.pos[0] and seg.pos[1] == i:
                        left_danger.append(i - snake.head.pos[1])

                for i in range(right_danger[0]):
                    if seg.pos[0] == snake.head.pos[0] and seg.pos[1] == i:
                        right_danger.append(snake.head.pos[1] - i)

        # gets the min value, which is the closest danger, and sets danger var to it
        straight_danger = min(straight_danger)
        left_danger = min(left_danger)
        right_danger = min(right_danger)

        # normalizing inputs between 0-1
        straight_danger = straight_danger / 20
        left_danger = left_danger / 20
        right_danger = right_danger / 20
        snake.fruit_angle = snake.fruit_angle / math.radians(360)
        snake.fruit_distance = snake.fruit_distance / math.sqrt(800)

        # converts to numpy array because predict expects one
        input = np.asarray([left_danger, straight_danger, right_danger, snake.fruit_distance, snake.fruit_angle])
        output = current_pop[model].predict(input, 1)
        print (output[0])

        # moves the snake a direction depending on which has the highest "probability score"
        # meaning what the model "thinks" is the best move
        if output[0][0] > output[0][1] and output[0][0] > output[0][2]:
            snake.moveDir = 'left'
        elif output[0][1] > output[0][0] and output[0][1] > output[0][2]:
            snake.moveDir = 'straight'
        elif output[0][2] > output[0][0] and output[0][2] > output[0][1]:
            snake.moveDir = 'right'
        else:
            snake.moveDir = random.choice(['left', 'straight', 'right'])

# takes a random weight from each parent and swaps them, like swapping part of a chromosome
def crossover(parent1, parent2):
    global current_pop

    parent1_weight = current_pop[parent1].get_weights()
    parent2_weight = current_pop[parent2].get_weights()
    parent1_new_weight = parent1_weight
    parent2_new_weight = parent2_weight

    gene = random.randint(0, len(parent1_weight) - 1)

    parent1_new_weight[gene] = parent2_weight[gene]
    parent2_new_weight[gene] = parent1_weight[gene]

    return np.asarray([parent1_new_weight, parent2_new_weight])

# for each weight, it has a 15% chance of being altered by -0.5 to 0.5
# this introduces novel mutations that will be sometimes helpful and sometimes hurtful
def mutate(weights):
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if random.uniform(0, 1) >= .85:
                change = random.uniform(-0.5, 0.5)
                weights[i][j] += change
    return weights

# this is where the snakes will be modified and mixed together to form the next generation
def endGame():
    global current_pop
    global generation
    global fitness
    global highest_fitness
    global best_weights

    new_weights = []
    new_highscore = False

    if best_weights == []:
        best_weights = current_pop[0].get_weights()

    for model in range(total_models):
        # if one of the child snakes fitness is higher than the max, make it the new max
        if fitness[model] >= highest_fitness:
            new_highscore = True
            highest_fitness = fitness[model]
            best_weights = current_pop[model].get_weights()

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
        current_pop[model].set_weights(new_weights[model])

    if save_current_pop == True:
        save_pop()

    print (f"Gen: {generation}   Max Fitness: {fitness[parent1]}")
    generation += 1
    return

# Initialize all models
for i in range(total_models):
    model = makeModel()
    current_pop.append(model)
    fitness.append(0)

if load_saved_pop:
    for i in range(total_models):
        current_pop[i].load_weights("Saved/model"+str(i)+".keras")

# each tail segment including the head is an instance of this class
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

# one instance of this per game, it is reset and reused for each model and generation
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
        self.move_limit = 100
        self.fruit_distance = -1
        self.fruit_angle = -1

    def reset(self):
        self.head = Segment([10, 10])
        self.tail = []
        self.tail.append(self.head)
        self.dir = 'north'
        self.moveDir = 'stop'
        self.dead = False
        self.winner = False
        self.move_limit = 100
        self.fruit_distance = -1
        self.fruit_angle = -1

    def draw(self, surface):
        for seg in self.tail:
            if seg == self.tail[0]:
                seg.draw(surface, True)
            else:
                seg.draw(surface)

    # moves by advancing the head, and then popping last tail segment off end and moving to last head position
    # with this, there is no need to move every tail segment each tick
    def move(self, model):
        lastHeadPos = self.head.pos
        last = self.tail.pop()
        last.pos = [lastHeadPos[0], lastHeadPos[1]]
        self.tail.insert(1, last)
        self.move_limit -= 1

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

        x = snake.head.pos[0] - fruit.pos[0]
        y = snake.head.pos[1] - fruit.pos[1]
        new_fruit_dist = math.sqrt((x ** 2) + (y ** 2))

        if new_fruit_dist > self.fruit_distance:
            fitness[model] -= 1.5
        elif new_fruit_dist < self.fruit_distance:
            fitness[model] += 1

    # checks if the snake's head is touching the wall or it's tail, or if it has run out of moves
    def checkDead(self, model):
        for seg in self.tail[1:]:
            if seg.pos[0] == self.head.pos[0] and seg.pos[1] == self.head.pos[1]:
                fitness[model] -= 10
                self.dead = True

        if self.head.pos[0] >= div or self.head.pos[0] < 0 or self.head.pos[1] < 0 or self.head.pos[1] >= div:
            fitness[model] -= 10
            self.dead = True

        if self.move_limit <= 0:
            fitness[model] -= 10
            self.dead = True

    # only wins if every board tile is filled, meaning the length of the tail is the area of the whole board
    def checkWin(self):
        if len(self.tail) >= div * div:
            self.dead = True
            self.winner = True

    # when a fruit is eaten, it respawns, fitness is added to the model, and more moves are added
    # the move limit is to prevent the snake spinning in circles forever not exploring or looking for food
    def eatFruit(self, model):
        fruit.spawn()
        fitness[model] += 100
        self.move_limit += 50

        # determines where the next tail segment should go based on snake's direction
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

    # has a check to ensure the fruit never spawns inside the snake's tail
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

    global snake, fruit, total_models, generation
    snake = Snake()
    fruit = Fruit()
    fruit.spawn()

    makeModel()

    # loops runs for each model in the generation, for each generation up to the max
    # predicts, moves, checks dead or win each move, when snake dies, moves on to next in gen
    max_gen = 100
    for i in range(max_gen):
        for model in range(total_models):
            while True:
                fps.tick(difficulty)
                pygame.display.set_caption(f"Snake!   Score: {len(snake.tail) - 1}   Gen: {generation}   Model: {model}")

                makePrediction(model)
                snake.move(model)
                snake.checkDead(model)
                snake.checkWin()

                if not snake.dead:
                    if snake.head.pos[0] == fruit.pos[0] and snake.head.pos[1] == fruit.pos[1]:
                        snake.eatFruit(model)
                    redrawWindow(screen)
                else:
                    break
            # once snake dies, this is triggered so the snake is reset and the game restarts
            snake.reset()
            fruit.spawn()
        # once all snakes in a generation have died, this runs to change weights for next gen
        endGame()

if __name__ == '__main__': main()