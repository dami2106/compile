import numpy as np
import random

# Define the available one-hot vectors
one_hot_vectors = [
    np.array([0, 0, 1]),
    np.array([1, 0, 0]),
    np.array([0, 1, 0])
]

truths_name = [
    "zero",
    "one",
    "two",
    "three"
]

for i in range(50):
    # Randomize the order in which sections are added
    random.shuffle(one_hot_vectors)

    sections = []
    for vec in one_hot_vectors:
        # Randomly choose the length of this section (between 1 and 3 rows)
        section_length = np.random.randint(1, 4)  # upper bound is exclusive
        # Create a section by repeating the one-hot vector for section_length rows
        section = np.tile(vec, (section_length, 1))
        sections.append(section)

    # Stack all sections vertically to form the final array
    final_array = np.vstack(sections)

    actions = []
    truths = []
    for r in final_array:
        ind = np.argmax(r)
        actions.append(ind)
        truths.append(truths_name[ind])

    actions = np.array(actions)

    np.save(f'Data/actions/simple_{i}.npy', actions)
    np.save(f'Data/features/simple_{i}.npy', final_array)

    #Save truths to a file where each truth is on a new line 
    with open(f'Data/groundTruth/simple_{i}', 'w') as f:
        f.write('\n'.join(truths))
            #Remove the last newline character
            # f.seek(f.tell() - 1)
            # f.truncate()
