import torch
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import pandas as pd
import itertools

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

def create_KM_model(latents, args):
    kmeans = KMeans(n_clusters=args.num_segments , random_state=args.random_seed, n_init='auto')
    kmeans.fit(latents)
    return kmeans

def create_GMM_model(latents, args):
    gmm = GaussianMixture(random_state=args.random_seed, n_components=args.num_segments)  # Specify the number of clusters
    gmm.fit(latents)
    return gmm


def create_DBSCAN_model(latents, eps= 1.2, min_samples = 50):
    # Create the DBSCAN model with specified parameters (eps and min_samples)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(latents)
    return dbscan

def predict_clusters(cluster_model, new_latents):

    clusters = []
    for l in new_latents:
        cluster = cluster_model.predict([l])[0]
        clusters.append(chr(65 + cluster))
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

    return colours[:len(colours) - 2]

def get_boundaries(state_set):
    colours = determine_objectives(state_set)

    #Get the index of the start of each colour segment
    boundaries = []
    for i in range(1, len(colours)):
        if colours[i] != colours[i-1]:
            boundaries.append(i)
    return [0] + boundaries + [len(colours) - 1]


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

    skill_dict = {
        "Prediction" : skills,
        "Truth" : truth
    }
    
    return skill_dict


#Takes in a list of dataframes
def get_skill_accuracy(skill_dict_list):
    df_new_all = pd.concat(skill_dict_list)
    # Define the possible labels in predictions and truth values
    prediction_labels = ['A', 'B', 'C']
    truth_labels = ['red', 'green', 'blue']

    # Generate all permutations of truth labels
    permutations = list(itertools.permutations(truth_labels))

    # Calculate total number of predictions
    new_total_predictions = len(df_new_all)

    # Create a dictionary to store accuracy for each permutation
    accuracy_results = {}

    # Iterate over each permutation, create the mapping, and calculate accuracy
    for perm in permutations:
        # Create the mapping for this permutation
        label_mapping_perm = dict(zip(prediction_labels, perm))
        
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