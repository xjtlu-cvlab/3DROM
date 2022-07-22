import math
import random
import numpy as np


def generate_occlusion(base, initial_index, final_index, ocnum):

    start, stop = (initial_index, final_index)
    random_list = []

    while (len(random_list) != ocnum):
        flagd = 0
        ind = random.choice(range(start, stop))
        coord = base.get_worldcoord_from_pos(ind)
        if len(random_list) < 2:
            random_list.append(ind)
        else:
            for pi in range(len(random_list)):
                coord2 = base.get_worldcoord_from_pos(random_list[pi])
                dist = math.sqrt((coord[0] - coord2[0]) * (coord[0] - coord2[0]) +
                                 (coord[1] - coord2[1]) * (coord[1] - coord2[1]))
                if dist > base.unit:
                    continue
                else:
                    flagd = 1
                    break
            if flagd == 0:
                random_list.append(ind)

    random_list = np.array(random_list, dtype=int)
    bboxes = base.read_pom()

    return random_list, bboxes
