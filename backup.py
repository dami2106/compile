# # Perform K-Means clustering on the latent space
# n_clusters = min(args.num_segments, len(latent_samples))  # Ensure we do not exceed the number of segments
# kmeans = KMeans(n_clusters=n_clusters)
# task_labels = kmeans.fit_predict(latent_samples)

# # Print the original sequence and its corresponding task labels
# print('Original sequence:')
# print(sequence)
# print('Task labels:')
# print(task_labels)

# # Identify boundary positions
# boundary_positions = [torch.argmax(b, dim=1)[0].item() for b in all_b['samples']]
# boundary_positions = [0] + boundary_positions + [len(sequence)]

# # Ensure the number of boundary positions matches the number of task labels
# segmented_tasks = []
# for i in range(len(boundary_positions) - 1):
#     start_idx = boundary_positions[i]
#     end_idx = boundary_positions[i + 1]
    
#     # Ensure index i is within the bounds of task_labels
#     if i < len(task_labels):
#         segmented_tasks.extend([task_labels[i]] * (end_idx - start_idx))
#     else:
#         # If there's a mismatch, assign a default or last task label
#         segmented_tasks.extend([task_labels[-1]] * (end_idx - start_idx))

# print('Segmented tasks:')
# print(segmented_tasks)


# model.eval()
# # Manually create the specific input sequence: 111222111333
# specific_input_sequence = [1, 1, 1, 2, 2, 2, 1, 1, 1, 3, 3, 3]

# # Convert the specific input sequence to a tensor and prepare it for the model
# specific_input = torch.tensor(specific_input_sequence).unsqueeze(0).to(device)  # Add batch dimension
# lengths = torch.tensor([len(specific_input_sequence)]).to(device)  # Lengths for the specific input

# # Run the model on the specific input
# all_encs, all_recs, all_masks, all_b, all_z = model.forward(specific_input, lengths)

# print(torch.stack(all_z['samples'], dim=1).squeeze(0).cpu().detach().numpy())

# # Convert to numpy for easier slicing and printing
# input_sample = specific_input[0].cpu().numpy()
# boundary_positions = [torch.argmax(b, dim=1)[0].item() for b in all_b['samples']]

# # Include the start and end positions to divide the input properly
# boundary_positions = [0] + boundary_positions + [len(input_sample)]

# # Print the input sample
# print('Input sample:')
# print(input_sample)

# # Divide the input into segments based on boundary positions
# segments = []
# segment_indices = []
# for i in range(len(boundary_positions) - 1):
#     start_idx = boundary_positions[i]
#     end_idx = boundary_positions[i + 1]
#     segments.append(input_sample[start_idx:end_idx])
#     segment_indices.append((start_idx, end_idx))

# # Print the segments and their corresponding indices
# print('\nSegments based on boundary positions:')
# for i, (segment, (start_idx, end_idx)) in enumerate(zip(segments, segment_indices)):
#     print(f'Segment {i + 1}: {segment} (Indices: {start_idx} to {end_idx-1})')




# ---------------------------
# print('\nAnalysis of a given input on the trained model:')
# model.eval()  # Switch to evaluation mode

# # Select a specific input for analysis
# specific_input = utils.generate_toy_data(
#     num_symbols=args.num_symbols,
#     num_segments=args.num_segments)
# lengths = torch.tensor([len(specific_input)])
# specific_input = specific_input.unsqueeze(0).to(device)  # Add batch dimension

# # Run the model on the specific input
# all_encs, all_recs, all_masks, all_b, all_z = model.forward(specific_input, lengths)

# # Convert to numpy for easier slicing and printing
# input_sample = specific_input[0].cpu().numpy()
# boundary_positions = [torch.argmax(b, dim=1)[0].item() for b in all_b['samples']]

# # Include the start and end positions to divide the input properly
# boundary_positions = [0] + boundary_positions + [len(input_sample)]

# # Print the input sample
# print('Input sample:')
# print(input_sample)

# # Divide the input into segments based on boundary positions
# segments = []
# segment_indices = []
# for i in range(len(boundary_positions) - 1):
#     start_idx = boundary_positions[i]
#     end_idx = boundary_positions[i + 1]
#     segments.append(input_sample[start_idx:end_idx])
#     segment_indices.append((start_idx, end_idx))

# # Print the segments and their corresponding indices
# print('\nSegments based on boundary positions:')
# for i, (segment, (start_idx, end_idx)) in enumerate(zip(segments, segment_indices)):
#     print(f'Segment {i + 1}: {segment} (Indices: {start_idx} to {end_idx-1})')


import torch 
import numpy as np


def generate_toy_data(num_symbols=5, num_segments=3, max_segment_len=5):
    """Generate toy data sample with repetition of symbols (EOS symbol: 0)."""
    seq = []
    symbols = np.random.choice(
        np.arange(1, num_symbols + 1), num_segments, replace=False)
    for seg_id in range(num_segments - 1):
        segment_len = np.random.choice(np.arange(1, max_segment_len))
        seq += [symbols[seg_id]] * segment_len

    segment_len = np.random.choice(np.arange(1, max_segment_len))
    seq += [symbols[0]] * segment_len

    seq += [0]
    return torch.tensor(seq, dtype=torch.int64)

print(generate_toy_data())



def post_process_boundaries(boundary_samples):
    """
    Convert boundary samples to actual segment indices.
    
    Args:
        boundary_samples: List of tensors with boundary samples from each segment.
    
    Returns:
        A list of indices representing the start of each segment.
    """
    segment_indices = []
    for sample_b in boundary_samples:
        # Convert the boundary samples to the actual positions.
        boundary_positions = torch.argmax(sample_b, dim=1).cpu().numpy()
        segment_indices.append(boundary_positions)

    # Flatten the list and remove duplicates
    flat_indices = sorted(set(np.concatenate(segment_indices)))
    
    # Ensure the indices are unique and sorted
    return flat_indices

def post_process_boundaries_2(boundary_samples):
    """
    Convert boundary samples to actual segment indices.
    
    Args:
        boundary_samples: List of tensors with boundary samples from each segment.
    
    Returns:
        A list where each entry corresponds to the start index of a segment.
    """
    segment_indices = []
    for i, sample_b in enumerate(boundary_samples):
        boundary_positions = torch.argmax(sample_b, dim=1)
        segment_indices.append(boundary_positions.cpu().numpy())
    return segment_indices


def extract_tasks(sequence, segment_indices):
    """
    Extract tasks from the sequence based on the boundary indices.
    
    Args:
        sequence: The input sequence (as a numpy array).
        segment_indices: List of segment start indices (as integers).
    
    Returns:
        A list of task segments (as numpy arrays).
    """
    tasks = []
    start_idx = 0
    
    for idx in segment_indices:
        if idx > start_idx:  # Ensure no empty segments
            task = sequence[start_idx:idx]
            tasks.append(task)
        start_idx = idx
    
    # Append the last segment
    if start_idx < len(sequence):
        tasks.append(sequence[start_idx:])
    
    return tasks

def classify_tasks(latent_samples):
    """
    Simple classification of tasks based on latent variables.
    
    Args:
        latent_samples: List of latent variables for each segment.
    
    Returns:
        A list of task classifications (e.g., task IDs).
    """
    task_ids = []
    for z in latent_samples:
        # In this simple case, assume the task ID is the argmax of the latent vector
        task_id = torch.argmax(z, dim=1).cpu().numpy()
        task_ids.append(task_id)
    return task_ids

# # Define the input sequence
# sequence = [1, 1, 1, 2, 2, 2, 1, 1, 1, 3, 3, 3]
# input_sequence = torch.tensor(sequence).unsqueeze(0).to(device)
# lengths = torch.tensor([len(sequence)]).to(device)

sequence = torch.tensor([[2, 2, 2, 3, 3, 3, 1]])
lengths = torch.tensor([7])  # Length of the sequence

# Run the model on the input sequence
model.eval()
all_encs, all_recs, all_masks, all_b, all_z = model.forward(sequence, lengths)



# Post-process the boundary samples to get the segment indices.
segment_indices = post_process_boundaries(all_b['samples'])
segment_indices2 = post_process_boundaries_2(all_b['samples'])

# print()
# print(sequence)
# print(segment_indices)
# print(segment_indices2)

boundary_positions = [torch.argmax(b, dim=1)[0].item() for b in all_b['samples']]

# Include the start and end positions to divide the input properly
boundary_positions = [0] + boundary_positions + [len(sequence)]


# Divide the input into segments based on boundary positions
segments = []
segment_indices = []
for i in range(len(boundary_positions) - 1):
    start_idx = boundary_positions[i]
    end_idx = boundary_positions[i + 1]
    segments.append(sequence[start_idx:end_idx])
    segment_indices.append((start_idx, end_idx))

# Print the segments and their corresponding indices
print('\nSegments based on boundary positions:')
for i, (segment, (start_idx, end_idx)) in enumerate(zip(segments, segment_indices)):
    print(f'Segment {i + 1}: {segment} (Indices: {start_idx} to {end_idx-1})')


# # Extract the tasks from the original sequence using the segment indices.
# sequence_np = sequence.cpu().numpy().flatten()
# tasks = extract_tasks(sequence_np, segment_indices)

# # Classify each task using the latent samples.
# task_classifications = classify_tasks(all_z['samples'])

# # Display the results
# print("Segment Indices (Boundaries):", segment_indices)
# print("Tasks Extracted:", tasks)
# print("Task Classifications:", task_classifications)



# # Convert to numpy for easier slicing and printing
# input_sample = specific_input[0].cpu().numpy()
# boundary_positions = [torch.argmax(b, dim=1)[0].item() for b in all_b['samples']]
# boundary_positions = [0] + boundary_positions + [len(input_sample)]

# # Print the input sample
# print('Input sample:')
# print(input_sample)

# # Divide the input into segments based on boundary positions
# segments = []
# segment_indices = []
# for i in range(len(boundary_positions) - 1):
#     start_idx = boundary_positions[i]
#     end_idx = boundary_positions[i + 1]
#     segments.append(input_sample[start_idx:end_idx])
#     segment_indices.append((start_idx, end_idx))

# # Print the segments and their corresponding indices
# print('\nSegments based on boundary positions:')
# for i, (segment, (start_idx, end_idx)) in enumerate(zip(segments, segment_indices)):
#     print(f'Segment {i + 1}: {segment} (Indices: {start_idx} to {end_idx-1})')