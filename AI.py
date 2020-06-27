import retro
import neat

env = retro.make('PacManNamco-Nes', 'Level1')
dataArray = []
def evaluate(genomes, config):
    for genomeID, genome in genomes:
        observation = env.reset()
        action = env.action_space.sample()
        network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        currentFitness = 0
        done = False
        observation, reward, done, info = env.step(action)
        while not done:
            env.render()
            dataArray = [info["pacmanX"], info["pacmanY"], info["pinkyX"], info["pinkyY"], info["clydeX"], info["clydeY"], info["blinkyX"], info["blinkyY"], info["inkyX"], info["inkyY"], info["fruit"]]
            networkOutput = network.activate(dataArray)
            observation, reward, done, info = env.step(networkOutput)
            currentFitness = info["score"]
            if int(info["lives"]) == 0:
                done = True
        genome.fitness = currentFitness
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config')
population = neat.Population(config)
winner = population.run(evaluate, 5)