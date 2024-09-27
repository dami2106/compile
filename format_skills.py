import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def get_latents(states, actions, model, args, device = 'cpu'):
    all_latents = []

    for i in range(len(states)):
        # Choose a single test input
        single_test_input = states[i:i + 1]  # Select the first trajectory for testing
        single_test_action = actions[i: i +1]  # Corresponding action sequence
        single_test_length = torch.tensor([args.max_steps]).to(device)

        # Convert to tensors and send to the appropriate device (CPU or GPU)
        single_test_input_tensor = torch.tensor(single_test_input).to(device)
        single_test_action_tensor = torch.tensor(single_test_action).to(device)
        single_test_inputs = (single_test_input_tensor, single_test_action_tensor)


        _, _, _, _, all_z = model.forward(single_test_inputs, single_test_length)

        for t in all_z['samples']:
            all_latents.append(t.detach().numpy()[0].tolist())

    all_latents = np.array(all_latents)
    return all_latents

def create_cluster_model_KM(latents, args):
    kmeans = KMeans(n_clusters=args.num_segments , random_state=args.random_seed, n_init='auto')
    kmeans.fit(latents)
    return kmeans

def create_GMM_model(latents, args):
    gmm = GaussianMixture(random_state=args.random_seed, n_components=args.num_segments)  # Specify the number of clusters
    gmm.fit(latents)
    return gmm


def predict_clusters(cluster_model, new_latents):

    cluster_to_skill = {
        0 : 'A',
        1 : 'B',
        2 : 'C',
        3 : 'D',
        4 : 'E',
        5 : 'F',
        6 : 'G',
    }

    clusters = []
    for l in new_latents:
        cluster = cluster_model.predict([l])[0]
        clusters.append(cluster_to_skill[cluster])
    return clusters


def extract_looking_for(state_set):
    colours = []
    for state in state_set:
        colours.append((state[4], state[7], state[10]))
    colours.append((1, 1, 1))
    return colours

def determine_objectives(state_set):
    trace = extract_looking_for(state_set)
   
    ind = [ 
        [0, "red"],
        [0, "green"],
        [0 , "blue"]
    ]

    for i in range(len(trace)):
        if trace[i][0] == 1:
            ind[0][0] = i
            break
    for i in range(len(trace)):
        if trace[i][1] == 1:
            ind[1][0] = i
            break    
    for i in range(len(trace)):
        if trace[i][2] == 1:
            ind[2][0] = i
            break      

    ind = sorted(ind, key=lambda x: x[0])
    
    first = (0, ind[0][0])
    second = (ind[0][0], ind[1][0])
    third = (ind[1][0], len(trace))
    colours = []

    for i in range(first[0], first[1]):
        colours.append(ind[0][1])
    for i in range(second[0], second[1]):
        colours.append(ind[1][1])
    for i in range(third[0], third[1]):
        colours.append(ind[2][1])

    return colours[:len(colours) - 1]

def get_boundaries(state_set):
    colours = determine_objectives(state_set)

    #Get the index of the start of each colour segment
    boundaries = []
    for i in range(1, len(colours)):
        if colours[i] != colours[i-1]:
            boundaries.append(i)
    return [0] + boundaries + [len(colours) - 1]


def calculate_metrics(true, predicted, tolerance=1):
    mse = np.mean((np.array(true) - np.array(predicted)) ** 2)

    correct_boundaries = np.sum(np.abs(np.array(true) - np.array(predicted)) <= tolerance)
    accuracy = correct_boundaries / len(true)

    return mse, accuracy


def skills_each_timestep(segments, clusters):
    assert len(clusters) == len(segments)
    skills = []

    for i in range(len(segments)):
        for _ in range(len(segments[i])):
            skills.append(clusters[i])
    
    return skills


def compare_skills_truth(states, segments, clusters):
    truth = determine_objectives(states)
    skills = skills_each_timestep(segments, clusters)

    print("Skills | Truth")
    for s, t in zip(skills, truth):
        print(f"{s:<8} {t}")


if __name__ == '__main__':
    print("here")

    test =np.array(\
    [[2,  3, -2,  2,  0,  1, -1,  0, -2,  0,  0],
    [ 3,  3, -1,  2,  0,  2, -1,  0, -1,  0,  0],
    [ 3,  2, -1,  1,  0,  2, -2,  0, -1, -1,  0],
    [ 4,  2,  0,  1,  0,  3, -2,  0,  0, -1,  0],

    [ 4,  1,  0,  0,  1,  3, -3,  0,  0, -2,  0],
    [ 4,  2,  0,  0,  1,  3, -2,  0,  0, -1,  0],

    [ 4,  3,  0,  0,  1,  3, -1,  0,  0,  0,  1],
    [ 3,  3,  0,  0,  1,  2, -1,  0,  0,  0,  1],
    [ 2,  3,  0,  0,  1,  1, -1,  0,  0,  0,  1]]
    )

    print(determine_objectives(test))