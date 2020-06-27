import retro
import neat

env = retro.make('PacManNamco-Nes', 'Level1')
generation = 0
def evaluate(genomes, config):
    global generation
    generation = generation + 1
    print("Generation: ", generation)
    specimin = 0
    for genomeID, genome in genomes:
        specimin = specimin + 1
        print("Specimin: ", specimin)
        observation = env.reset()
        action = env.action_space.sample()
        network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        currentFitness = 0
        done = False
        observation, reward, done, info = env.step(action)
        while not done:
            env.render()
            dataArray = [info["pacmanX"], info["pacmanY"], info["pinkyX"], info["pinkyY"], info["clydeX"], info["clydeY"], info["blinkyX"], info["blinkyY"], info["inkyX"], info["inkyY"], info["pellets"]]
            networkOutput = network.activate(dataArray)
            observation, reward, done, info = env.step(networkOutput)
            currentFitness = info["score"]
            if int(info["lives"]) == 0:
                done = True
        genome.fitness = currentFitness
        print("Fitness: ", currentFitness)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config')
population = neat.Population(config)
winner = population.run(evaluate)