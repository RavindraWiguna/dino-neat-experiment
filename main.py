from time import time
from typing import Counter
from PIL import ImageGrab
import cv2
import numpy as np
from neatUtils import visualize
import neat
import pickle
import pyautogui
import time
import os
# pyautogui.keyDown(pyautogui.KEYBOARD_KEYS)
# print(pyautogui.KEY_NAMES) # they are the same
# Global variable for game
goreg = cv2.imread('game_over.jpg', 0)
threshold = 0.8
x, y, w, h = 77, 306, 468, 88
x2, y2 = x+w, y+h
ss_region = (x, y, x2, y2)
inx, iny = w//6, h//6
print(inx*iny)
def game(genome, config):
    # Create the neural networks
    # net = neat.nn.FeedForwardNetwork.create(genome, config)
    net = neat.nn.RecurrentNetwork.create(genome, config)
    isRun = True
    # dict_div = {}
    # for i in range(256):
        # dict_div[i] = i/255
    
    last_dict = {0: -12, 1:0, 2:0}
    # prev_time = time.time()
    # cur_time = time.time()
    # click on the browser
    # restart key down
    pyautogui.keyUp('up')
    pyautogui.keyUp('down')
    # click on browser
    pyautogui.click(x+10, y+10)
    # pyautogui.keyDown('space')
    # pyautogui.keyUp('space')
    start_time = time.time()
    last_act = 0
    keysName = ('x', 'up', 'down')
    k_id = 0
    avoided = 0
    while isRun:        
        ss_img = ImageGrab.grab(ss_region)
        # print(type(ss_img))

        frame = np.array(ss_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        res = cv2.matchTemplate(frame[21:75, 200:260],goreg,cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= threshold)[::-1]
        loc = list(zip(*loc))

        frame = cv2.resize(frame, (inx, iny))


        frame = frame.astype('float64')
        frame/=255.0
        # frame.flatten()
        # frame = 
        # frame = np.vectorize(dict_div.get)(frame) # faster with divideer
        # print(frame)
        
        # cv2.imshow('test', frame[0:iny, 0:5])
        # print(frame[0:iny, 0:5])
                    #    0.3254902
        
        # print(np.in1d([0.3254902], frame[0:iny, 0:5]))
        # print(frame[0:iny, 0:5].flatten()*255)
        # print('=========')
        # print("don't close it with mouse!")
        k = cv2.waitKey(1) & 0xFF
        # check for esc pressed or died
        if(loc):
            isRun = False
            # cv2.destroyAllWindows()
            break
        
        avoided += 0
        # move the dang daino sour (after checking die, cause welp)
        output = net.activate(frame.flatten())
        comp_move = np.argmax(output)
        if(comp_move==1):
            if(k_id==2):
                pyautogui.keyUp('down')
            # say jump
            pyautogui.keyDown('up')
            # time.sleep(0.02)
            # pyautogui.keyUp('up')
            # print("shite")
        elif(comp_move==2):
            if(k_id==1):
                pyautogui.keyUp('up')
            pyautogui.keyDown('down')
            # time.sleep(0.02)
        else:
            if(k_id != 0):
                pyautogui.keyUp(keysName[k_id])
            # pyautogui.keyUp('down')
        last_act = comp_move
        k_id = comp_move
        # cur_time = time.time()
        # print(f'FPS: {1/(cur_time-prev_time)}')
        # prev_time = cur_time
        # time.sleep(0.5)
    # print(last_act)
    return time.time() - start_time + avoided*2 + last_dict[last_act]


def eval_genomes(genomes, config):
    print("training genomes")
    """
    Run each genome to play game
    """
    for genome_id, genome in genomes:
        genome.fitness = game(genome, config)
        print(f"genome:{genome_id}.fitness = {genome.fitness}")
        

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    print("create")
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-11')
    print("shees")
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Save the winner
    with open("best_genome.pickle", "wb") as saver:    
        pickle.dump(winner, saver, pickle.HIGHEST_PROTOCOL)
        print("WINNER IS SAVED on best_genome.pickle")
    
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    
    # Show output of the most fit genome against training data.
    # print('\nOutput:')
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # To load from last check point (in case the training is stopped syre)
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)

if __name__=="__main__":
    local_dir = os.path.dirname(__file__)
    # print(local_dir)
    config_path = os.path.join(local_dir, './neatUtils/config-feedforward')
    print("run")
    run(config_path)
    # game(1, 2) 
    # pass