import retro
import neat
import cv2
import numpy
import pickle

env = retro.make('PacManNamco-Nes', 'Level1')
imageArray = []
ob = env.reset()
def evaluate(genomes, config):
    for genomeID, genome in genomes:
        observation = env.reset()
        action = env.action_space.sample()
        inputX, inputY, inputC = env.observation_space.shape
        inputX = int(inputX/4)
        inputY = int(inputY/4)
        network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        currentFitness = 0
        done = False
        while not done:
            env.render()
            observation = cv2.resize(observation, (inputX, inputY))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = numpy.reshape(observation, (inputX, inputY))
            for x in observation:
                for y in x:
                    imageArray.append(y)
            networkOutput = network.activate(imageArray)
            observation, reward, done, info = env.step(networkOutput)
            imageArray.clear()
            currentFitness = info["score"]
            if int(info["lives"]) == 0:
                done = True
        genome.fitness = currentFitness
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config')
population = neat.Population(config)
winner = population.run(evaluate, 5)