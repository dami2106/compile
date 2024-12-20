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
    model.eval()
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
def create_GMM_model(latents, args, components = 3):
    gmm = GaussianMixture(random_state=args.random_seed, n_components=components)  # Specify the number of clusters
    gmm.fit(latents)
    return gmm

"""
A function to generate a list of AIC, BIC for a range of components on a GMM model
Useful to plot an Elbow chart to see the number of skills
@param latents: The latents to create the model from
@param args: The arguments for the model
@param n_components: The range of components to test
@return: A tuple of the n_components, aic, bic
"""
def generate_elbow_plot(latents, args, n_components = np.arange(1, 12)):
    aic = []
    bic = []
    for n in n_components:
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(latents)
        aic.append(gmm.aic(latents))
        bic.append(gmm.bic(latents))

    return n_components, aic, bic



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
        clusters.append(cluster)
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

    # return colours

    #Convert the list of strings to a list of integers
    colour_dict = {
        "red": 0,
        "green": 1,
        "blue": 2
    }

    return [colour_dict[colour] for colour in colours]

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
def calculate_metrics(true_boundaries_list, predicted_boundaries_list, tolerance=0):
    mse_list = []
    l2_distance_list = []
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_boundaries = 0
    total_correct_boundaries = 0

    for true_boundaries, predicted_boundaries in zip(true_boundaries_list, predicted_boundaries_list):
        try:
            true_boundaries = np.array(true_boundaries)
            predicted_boundaries = np.array(predicted_boundaries)

            # Calculate MSE for this particular datapoint
            # print(true_boundaries, predicted_boundaries)
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
        except:
            pass

    # Overall MSE
    overall_mse = np.mean(mse_list)

    # Overall L2 distance
    overall_l2_distance = np.mean(l2_distance_list)

    # Accuracy
    # accuracy = total_correct_boundaries / total_boundaries

    # Precision: True Positives / (True Positives + False Positives)
    precision = total_true_positives / (total_true_positives + total_false_positives) if total_true_positives + total_false_positives > 0 else 0

    # Recall: True Positives / (True Positives + False Negatives)
    recall = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0

    # F1 Score: Harmonic mean of Precision and Recall
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return overall_mse, overall_l2_distance, 0, precision, recall, f1_score



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


def convert_dict_to_sota(skill_dict_list):
    torch_segs, np_segs, torch_truth, np_truth = [], [], [], []

    for d in skill_dict_list:
        preds = d["Prediction"]
        truths = d["Truth"]

        torch_segs.append(torch.tensor(preds))
        np_segs.append(np.array(preds))
        torch_truth.append(torch.tensor(truths))
        np_truth.append(np.array(truths))

    return torch_segs, np_segs, torch_truth, np_truth

#Takes in a list of dataframes
def get_skill_accuracy(skill_dict_list, cluster_num = 4):
    df_new_all = pd.concat(skill_dict_list)
 
    truth_labels = ['red', 'green', 'blue']
    prediction_labels = [chr(65 + x) for x in range(cluster_num)]

    # Generate all permutations of truth labels
    permutations = list(itertools.permutations(prediction_labels, 3))

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

    state[2] = red[0]  #distance red x 
    state[3] = red[1] #distance red y 
    state[4] = 0 if has_red else 1 #If have red in state

    state[5] = green[0]  #distance green x 
    state[6] = green[1]  #distance green y 
    state[7] = 0 if has_green else 1 #If have green in state

    state[8] = blue[0]  #distance blue x 
    state[9] = blue[1]  #distance blue y 
    state[10] = 0 if has_blue else 1 #If have blue in state

    return state

def get_simple_obs_list(obs_list):
    simple_obs = []
    for obs in obs_list:
        simple_obs.append(get_simple_obs(obs.reshape(5, 5)))    
    return np.array(simple_obs)