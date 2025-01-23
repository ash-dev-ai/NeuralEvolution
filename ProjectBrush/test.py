import neat

def test_neat():
    config_path = "config/neat-config.ini"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    print("NEAT configured successfully!")

if __name__ == "__main__":
    test_neat()