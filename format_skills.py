import torch
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import pandas as pd
import itertools
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os 

"""
TODO SIMPLIFY THE FUNCTION
A function to get the latents for each trajectory in the given dataset
Creates a numpy array from the latents where each latent is a new row
@param states: The states of the dataset
@param actions: The actions of the dataset
@param model: The trained model
@param args: The arguments for the model
@param device: The device to run the model on
@return: A numpy array of the latents for each trajectory
"""
def get_latents(states, actions, model, args, device = 'cuda'):
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
            all_latents.append(t.detach().cpu().numpy()[0].tolist())

    all_latents = np.array(all_latents)
    return all_latents


"""
A function to create a K MEANS clustering model based on the given latents
@param latents: The latents to create the model from
@param args: The arguments for the model
@return: The clustering model
"""
def create_KM_model(latents, args):
    kmeans = KMeans(n_clusters=args.num_segments , random_state=args.random_seed, n_init='auto')
    kmeans.fit(latents)
    return kmeans

"""
A function to create a Gaussian Mixture Model based on the given latents
@param latents: The latents to create the model from
@param args: The arguments for the model
@return: The clustering model
"""
def create_GMM_model(latents, args, clusters = 10):
    gmm = GaussianMixture(random_state=args.random_seed, n_components=clusters)  # Specify the number of clusters
    gmm.fit(latents)
    return gmm


"""
A function to predict a cluster for each latent within the latent array. Assigns a letter to each cluster
@param cluster_model: The clustering model to use
@param new_latents: The latents to predict the clusters for
@return: A list of the predicted clusters
"""
def predict_clusters(cluster_model, new_latents):
    clusters = []
    for l in new_latents:
        cluster = cluster_model.predict([l])[0]
        clusters.append(chr(65 + cluster))
    return clusters


"""
A function to create an Nx3 numpy array of the colours that the agent is looking for at each time step
@param state_set: The states of the trace
@return: An Nx3 numpy array of the colours the agent is looking for at each time step
"""
def extract_looking_for(state_set):
    colours = []
    for state in state_set:
        colours.append((state[4], state[7], state[10]))
    colours.append((1, 1, 1))
    return colours


"""
A function to determine the objectives of the agent at each time step
@param state_set: The states of the trace
@return: A list of the colours the agent is looking for at each time step
"""
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
    third = (ind[1][0], len(trace) - 1)
    colours = []

    for i in range(first[0], first[1]):
        colours.append(ind[0][1])
    for i in range(second[0], second[1]):
        colours.append(ind[1][1])
    for i in range(third[0], third[1]):
        colours.append(ind[2][1])

    return colours

"""
A function to get the boundaries of the colour segments in the state set
@param state_set: The states of the trace
@return: A list of the boundaries of the colour segments
"""
def get_boundaries(state_set):
    colours = determine_objectives(state_set)

    #Get the index of the start of each colour segment
    boundaries = []
    for i in range(1, len(colours)):
        if colours[i] != colours[i-1]:
            boundaries.append(i)
    return [0] + boundaries + [len(colours) - 1]


"""
A function to calculate the metrics for the predicted boundaries against the true boundaries
@param true_boundaries_list: A list of true boundaries for each trajectory
@param predicted_boundaries_list: A list of predicted boundaries for each trajectory
@param tolerance: The tolerance for a predicted boundary to be considered correct (how far it can be from the true boundary)
@return: A tuple of the overall MSE, overall L2 distance, accuracy, precision, recall, and F1 score
"""
def calculate_metrics(true_boundaries_list, predicted_boundaries_list, tolerance=1):
    mse_list = []
    l2_distance_list = []
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_boundaries = 0
    total_correct_boundaries = 0

    for true_boundaries, predicted_boundaries in zip(true_boundaries_list, predicted_boundaries_list):
        true_boundaries = np.array(true_boundaries)
        predicted_boundaries = np.array(predicted_boundaries)

        # Calculate MSE for this particular datapoint
        mse = np.mean((true_boundaries - predicted_boundaries) ** 2)
        mse_list.append(mse)

        # Calculate L2 distance for this particular datapoint
        l2_distance = np.sqrt(np.sum((true_boundaries - predicted_boundaries) ** 2))
        l2_distance_list.append(l2_distance)

        # Calculate True Positives, False Positives, False Negatives
        for pred in predicted_boundaries:
            if any(np.abs(true_boundaries - pred) <= tolerance):
                total_true_positives += 1
            else:
                total_false_positives += 1
        
        for true in true_boundaries:
            if not any(np.abs(predicted_boundaries - true) <= tolerance):
                total_false_negatives += 1

        # Calculate correct boundaries for accuracy
        correct_boundaries = np.sum(np.abs(true_boundaries - predicted_boundaries) <= tolerance)
        total_correct_boundaries += correct_boundaries
        total_boundaries += len(true_boundaries)

    # Overall MSE
    overall_mse = np.mean(mse_list)

    # Overall L2 distance
    overall_l2_distance = np.mean(l2_distance_list)

    # Accuracy
    accuracy = total_correct_boundaries / total_boundaries

    # Precision: True Positives / (True Positives + False Positives)
    precision = total_true_positives / (total_true_positives + total_false_positives) if total_true_positives + total_false_positives > 0 else 0

    # Recall: True Positives / (True Positives + False Negatives)
    recall = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0

    # F1 Score: Harmonic mean of Precision and Recall
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return overall_mse, overall_l2_distance, accuracy, precision, recall, f1_score



#TODO FIX THIS
def skills_each_timestep(segments, clusters):

    assert len(clusters) == len(segments)
    skills = []

    for i in range(len(segments)):
        for _ in range(len(segments[i])):
            skills.append(clusters[i])
    
    return skills


def print_skills_against_truth(states, segments, clusters):
    truth = determine_objectives(states)
    skills = skills_each_timestep(segments, clusters)

    print("Prediction | Truth")
    for s, t in zip(skills, truth):
        print(f"{s:<8} {t}")

def get_skill_dict(states, segments, clusters):
    truth = determine_objectives(states)
    skills = skills_each_timestep(segments, clusters)

    if len(truth) != len(skills):

        # print(truth)
        # print(skills)
        raise ValueError("Length of truth and skills do not match")

    skill_dict = {
        "Prediction" : skills,
        "Truth" : truth
    }
    
    return skill_dict


def print_directions_against_truth(directional_truth, segments, clusters):
    # truth = determine_objectives(states)
    skills = skills_each_timestep(segments, clusters)

    print("Prediction | Truth")
    for s, t in zip(skills, directional_truth):
        print(f"{s:<8} {t}")

def get_directional_dict(directional_truth, segments, clusters):
    # truth = determine_objectives(states)
    skills = skills_each_timestep(segments, clusters)

    if len(directional_truth) != len(skills):

        # print(truth)
        # print(skills)
        raise ValueError("Length of truth and skills do not match")

    skill_dict = {
        "Prediction" : skills,
        "Truth" : directional_truth
    }
    
    return skill_dict


def get_skill_dict_treasure(truth, predicted_boundaries, clusters): 
    skills = []
    for i in range(0, len(predicted_boundaries) - 1):
        cluster_len = (predicted_boundaries[i+1] - predicted_boundaries[i])
        if predicted_boundaries[i] == 0 :
            cluster_len += 1
        for _ in range(cluster_len):
            skills.append(clusters[i])

    new_truth = [str(x) for x in truth]

    skill_dict = {
        "Prediction" : skills,
        "Truth" : new_truth
    }


    return skill_dict


#Takes in a list of dataframes
def get_skill_accuracy(skill_dict_list, cluster_num = 3, type = "direction"):
    df_new_all = pd.concat(skill_dict_list)
 
    if type == "direction":
        truth_labels = ['bottom', 'top', 'middle']
    else:
        truth_labels = ['red', 'green', 'blue']

    # truth_labels = [str(x) for x in range(cluster_num)]
    prediction_labels = [chr(65 + x) for x in range(cluster_num)]

    # Generate all permutations of truth labels
    permutations = list(itertools.permutations(prediction_labels, cluster_num))

    # Calculate total number of predictions
    new_total_predictions = len(df_new_all)

    # Create a dictionary to store accuracy for each permutation
    accuracy_results = {}


    # Iterate over each permutation, create the mapping, and calculate accuracy
    for perm in permutations:
        # Create the mapping for this permutation
        label_mapping_perm = dict(zip(perm, truth_labels))
        
        # Apply the mapping to predictions
        df_new_all['Mapped_Prediction'] = df_new_all['Prediction'].map(label_mapping_perm)
        
        # Count correct matches (where mapped predictions match the truth)
        correct_matches_perm = (df_new_all['Mapped_Prediction'] == df_new_all['Truth']).sum()


        
        # Calculate accuracy for this permutation
        accuracy_perm = correct_matches_perm / new_total_predictions
        
        # Store the accuracy in the dictionary
        accuracy_results[str(label_mapping_perm)] = accuracy_perm

    # Display all permutation results sorted by accuracy
    sorted_accuracy_results = sorted(accuracy_results.items(), key=lambda x: x[1], reverse=True)

    # Output the sorted results
    return sorted_accuracy_results


def layered_to_vector(state):
    has_red = 1 in state[1]
    has_green = 1 in state[2]
    has_blue = 1 in state[3] 

    agent = get_coords(state[0], 1)
    red = get_coords(state[1], 1) if has_red else [-1, -1]
    green = get_coords(state[2], 1) if has_green else [-1, -1]
    blue = get_coords(state[3], 1) if has_blue else [-1, -1]

    state = np.zeros(11)

    state[0] = agent[0] #agent x 
    state[1] = agent[1] #agent y 

    state[2] = agent[0] - red[0] if has_red else 0  #distance red x 
    state[3] = agent[1] - red[1] if has_red else 0 #distance red y 
    state[4] = 0 if has_red else 1 #If have red in state

    state[5] = agent[0] - green[0] if has_green else 0  #distance green x 
    state[6] = agent[1] - green[1] if has_green else 0  #distance green y 
    state[7] = 0 if has_green else 1 #If have green in state

    state[8] = agent[0] - blue[0] if has_blue else 0  #distance blue x 
    state[9] = agent[1] - blue[1] if has_blue else 0  #distance blue y 
    state[10] = 0 if has_blue else 1 #If have blue in state

    return state
    

def PCA_cluster_plot(clusters, latents, title = "PCA Cluster Plot", directory = ""):
    pca = PCA(n_components=3, random_state=42)
    latents_3d = pca.fit_transform(latents)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    label_encoder = LabelEncoder()
    numeric_clusters = label_encoder.fit_transform(clusters)

    # Plot the 3D scatter with numeric clusters
    scatter = ax.scatter(latents_3d[:, 0], latents_3d[:, 1], latents_3d[:, 2], 
                        c=numeric_clusters, cmap='viridis', s=50)
        
    ax.set_title(title)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')

    # Adding color bar to show the different clusters
    plt.colorbar(scatter, ax=ax, label='Cluster Labels')

    plt.savefig(os.path.join(directory, title + '.png'))

def get_coords(obs, search):
    pos = np.where(obs == search)   
    return [pos[0][0], pos[1][0]]

def get_simple_obs(obs):
    # [agent_x, agent_y, dis_r_x, dis_r_y, has_red, dis_g_x,\\
    #  dis_g_y, has_green, dis_b_x, dis_r_x, has_blue ] 

    has_red = 2 in obs
    has_green = 3 in obs 
    has_blue = 4 in obs 

    agent = get_coords(obs, 1)
    red = get_coords(obs, 2) if has_red else [-1, -1]
    green = get_coords(obs, 3) if has_green else [-1, -1]
    blue = get_coords(obs, 4) if has_blue else [-1, -1]

    state = np.zeros(11)

    state[0] = agent[0] #agent x 
    state[1] = agent[1] #agent y 

    state[2] = agent[0] - red[0] if has_red else 0  #distance red x 
    state[3] = agent[1] - red[1] if has_red else 0 #distance red y 
    state[4] = 0 if has_red else 1 #If have red in state

    state[5] = agent[0] - green[0] if has_green else 0  #distance green x 
    state[6] = agent[1] - green[1] if has_green else 0  #distance green y 
    state[7] = 0 if has_green else 1 #If have green in state

    state[8] = agent[0] - blue[0] if has_blue else 0  #distance blue x 
    state[9] = agent[1] - blue[1] if has_blue else 0  #distance blue y 
    state[10] = 0 if has_blue else 1 #If have blue in state

    return state

def get_simple_obs_list(obs_list):
    new_obs_list = []
    for obs in obs_list:
        s_obs = np.array(obs).reshape(5, 5)
        new_obs_list.append(get_simple_obs(s_obs))
    return new_obs_list

def get_simple_obs_list_from_layers(obs_list):
    new_obs_list = []
    for obs in obs_list:
        s_obs = np.array(obs).reshape(4, 5, 5)
        new_obs_list.append(layered_to_vector(s_obs))
    return new_obs_list


"""
A function to get the boundaries of the treasure segments in the state set
@param state_set: The states of the trace
@return: A list of the boundaries of the colour segments
"""
def get_boundaries_treasure(ground_truth):
    boundaries = []
    for i in range(1, len(ground_truth)):
        if ground_truth[i] != ground_truth[i-1]:
            boundaries.append(i - 1)
    return [0] + boundaries + [len(ground_truth) - 1]


#@TODO WRite comments

def reverse_3d_obs(new_obs):
    size = new_obs.shape[1]  # Assuming new_obs is of shape (4, size, size)
    obs = np.zeros((size, size), dtype=np.uint8)

    # Map objects from new_obs back into obs
    for layer, value in enumerate([1, 2, 3, 4]):  # 1: agent, 2: red, 3: green, 4: blue
        coords = np.argwhere(new_obs[layer] == 1)
        if coords.size > 0:  # Check if there's a non-zero entry
            x, y = coords[0]  # Get the coordinates (only one per layer)
            obs[x, y] = value

    return obs

def classify_positions(color_coords):
    # Sort colors by the row index of their coordinates
    sorted_colors = sorted(color_coords.items(), key=lambda item: item[1][0])
    
    # Assign top, middle, and bottom based on sorted order
    classifications = {
        sorted_colors[0][0]: "top",
        sorted_colors[1][0]: "middle",
        sorted_colors[2][0]: "bottom"
    }
    
    return classifications

def analyze_pickups(layered_states):
    flat_states = np.array([reverse_3d_obs(obs) for obs in layered_states])

    positions = {
        "red": np.argwhere(flat_states[0] == 2)[0],
        "green": np.argwhere(flat_states[0] == 3)[0],
        "blue": np.argwhere(flat_states[0] == 4)[0]
    }

    classifications = classify_positions(positions)

    simple_states = []
    for trace in flat_states:
        simple_states.append(get_simple_obs(trace))

    simple_states = np.array(simple_states)

    objectives = determine_objectives(simple_states)

    directions = []

    for obj in objectives:
        directions.append(classifications[obj])
    
    return directions
