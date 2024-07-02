import json
import os
import numpy as np
from scipy.ndimage import label, find_objects
import hashlib
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors

import arckit
import arckit.vis as vis

def plot_shape(shape_grid, ax, shape_name, count, color):
    if isinstance(color, int):
        cmap = plt.get_cmap('tab20')
        rgba_color = cmap(color % 20)
    elif color == 'N/A':
        rgba_color = (0.7, 0.7, 0.7, 1)  # Gray color for N/A
    else:
        try:
            rgba_color = mcolors.to_rgba(color)
        except ValueError:
            rgba_color = (0.7, 0.7, 0.7, 1)  # Default to gray if color is invalid

    ax.imshow(shape_grid, cmap=mcolors.ListedColormap([rgba_color]))
    ax.set_title(f"{shape_name}\nCount: {count}", fontsize=8)
    ax.axis('off')

def load_data(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def identify_shapes(grid):
    shapes = {}
    shapes.update(identify_contiguous_shapes(grid))
    shapes.update(identify_same_color_shapes(grid))
    shapes.update(identify_relationship_shapes(grid))
    shapes.update(identify_specific_shapes(grid))
    shapes.update(identify_primitives(grid))
    return shapes

def get_shapes_for_matrix(grid):
    shapes = identify_shapes(grid)
    return shapes

def process_and_visualize_examples(file_paths, num_examples=3):
    for file_path in file_paths[:num_examples]:
        data = load_data(file_path)
        if not data:
            continue
        for item in data['train'][:1]:  # Process only the first train example from each file
            input_grid = np.array(item['input'])
            output_grid = np.array(item['output'])
            
            input_shapes = get_shapes_for_matrix(input_grid)
            output_shapes = get_shapes_for_matrix(output_grid)
            
            visualize_shapes(input_grid, input_shapes, f'Input Shapes - {os.path.basename(file_path)}')
            visualize_shapes(output_grid, output_shapes, f'Output Shapes - {os.path.basename(file_path)}')
            
            print(f"Shapes in input of {os.path.basename(file_path)}:")
            for shape, details in input_shapes.items():
                print(f"  {shape}: {details['details']['type']}, Count: {details['count']}, Color: {details['details'].get('color', 'N/A')}")
            print(f"Shapes in output of {os.path.basename(file_path)}:")
            for shape, details in output_shapes.items():
                print(f"  {shape}: {details['details']['type']}, Count: {details['count']}, Color: {details['details'].get('color', 'N/A')}")
            print("\n")

def detect_square(grid, x, y, visited, current_color):
    max_size = min(len(grid) - x, len(grid[0]) - y)
    for size in range(1, max_size + 1):
        if all(grid[i][j] == current_color and (i, j) not in visited
               for i in range(x, x + size) for j in range(y, y + size)):
            for i in range(x, x + size):
                for j in range(y, y + size):
                    visited.add((i, j))
            return "square", size
    return None

def detect_rectangle(grid, x, y, visited, current_color):
    max_height = len(grid) - x
    max_width = len(grid[0]) - y
    for height in range(1, max_height + 1):
        for width in range(1, max_width + 1):
            if all(grid[i][j] == current_color and (i, j) not in visited
                   for i in range(x, x + height) for j in range(y, y + width)):
                for i in range(x, x + height):
                    for j in range(y, y + width):
                        visited.add((i, j))
                return "rectangle", (height, width)
    return None

def detect_bar(grid, x, y, visited, current_color, orientation):
    dx, dy = orientation
    length = 0
    i, j = x, y
    while 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == current_color and (i, j) not in visited:
        visited.add((i, j))
        length += 1
        i += dx
        j += dy
    return f"bar-{orientation}", length if length > 1 else None

def identify_same_color_shapes(grid):
    shapes = {}
    colors = set(cell for row in grid for cell in row if cell != 0)
    for color in colors:
        color_mask = [[1 if cell == color else 0 for cell in row] for row in grid]
        labeled, num_features = label(color_mask)
        for i in range(1, num_features + 1):
            shape_mask = labeled == i
            shape_size = np.sum(shape_mask)
            if shape_size == 1:
                continue  # Filter out 1x1 shapes
            shape_hash = hash_shape("same_color", shape_size, shape_size, color)
            if shape_hash in shapes:
                shapes[shape_hash]['count'] += 1
            else:
                shapes[shape_hash] = {'count': 1, 'details': {"type": "same_color", "color": color, "size": shape_size}}
    return shapes

def identify_specific_shapes(grid):
    shapes = {}
    shapes.update(identify_rings(grid))
    shapes.update(identify_filled_shapes(grid))
    shapes.update(identify_lined_shapes(grid))
    shapes.update(identify_corner_pointed_shapes(grid))
    return shapes

def hash_shape(shape_type, width, height, color):
    if shape_type == 'primitive':
        return f"primitive_{width}_{height}"
    shape_str = f"{shape_type}_{width}_{height}_{color}"
    return f"{hashlib.md5(shape_str.encode()).hexdigest()}_{width}_{height}"

def identify_same_color_shapes(grid):
    shapes = {}
    colors = set(cell for row in grid for cell in row if cell != 0)
    for color in colors:
        color_mask = [[1 if cell == color else 0 for cell in row] for row in grid]
        labeled, num_features = label(color_mask)
        for i in range(1, num_features + 1):
            shape_mask = labeled == i
            shape_size = np.sum(shape_mask)
            if shape_size == 1:
                continue  # Filter out 1x1 shapes
            shape_name = hash_shape("same_color", shape_mask.shape[1], shape_mask.shape[0], color)
            if shape_name in shapes:
                shapes[shape_name]['count'] += 1
            else:
                shapes[shape_name] = {'count': 1, 'details': {"type": "same_color", "color": color, "size": shape_size, "width": shape_mask.shape[1], "height": shape_mask.shape[0]}}
    return shapes

def identify_relationship_shapes(grid):
    shapes = {}
    # Identify four points forming a square
    for i in range(len(grid) - 1):
        for j in range(len(grid[0]) - 1):
            if grid[i][j] != 0 and grid[i][j+1] != 0 and grid[i+1][j] != 0 and grid[i+1][j+1] != 0:
                shape_name = hash_shape("four_point_square", 2, 2, None)
                if shape_name in shapes:
                    shapes[shape_name]['count'] += 1
                else:
                    shapes[shape_name] = {'count': 1, 'details': {"type": "four_point_square", "start": (i, j), "size": 2}}
    return shapes

def identify_rings(grid):
    shapes = {}
    for i in range(1, len(grid) - 1):
        for j in range(1, len(grid[0]) - 1):
            if grid[i][j] == 0 and all(grid[i+di][j+dj] != 0 for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]):
                shape_name = hash_shape("ring", 3, 3, None)
                if shape_name in shapes:
                    shapes[shape_name]['count'] += 1
                else:
                    shapes[shape_name] = {'count': 1, 'details': {"type": "ring", "center": (i, j), "size": 3}}
    return shapes

def identify_contiguous_shapes(grid):
    shapes = {}
    visited = set()
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if (i, j) not in visited:
                current_color = grid[i][j]
                if current_color == 0:  # Skip background color
                    continue
                shape = (detect_square(grid, i, j, visited, current_color) or
                         detect_rectangle(grid, i, j, visited, current_color) or
                         detect_bar(grid, i, j, visited, current_color, (1, 0)) or
                         detect_bar(grid, i, j, visited, current_color, (0, 1)) or
                         detect_bar(grid, i, j, visited, current_color, (1, 1)))
                if shape:
                    shape_type, shape_size = shape
                    if shape_type == 'square' and shape_size == 1:
                        continue  # Filter out 1x1 shapes
                    if shape_type == 'square':
                        shape_name = hash_shape(shape_type, shape_size, shape_size, current_color)
                    elif shape_type == 'rectangle':
                        height, width = shape_size
                        shape_name = hash_shape(shape_type, width, height, current_color)
                    elif shape_type.startswith('bar'):
                        shape_name = hash_shape(shape_type, shape_size, 1, current_color)
                    else:
                        continue  # Skip unknown shape types
                    if shape_name in shapes:
                        shapes[shape_name]['count'] += 1
                    else:
                        shapes[shape_name] = {'count': 1, 'details': {"type": shape_type, "start": (i, j), "size": shape_size, "color": current_color}}
    return shapes


def identify_filled_shapes(grid):
    # Placeholder for filled shapes identification
    return {}

def identify_lined_shapes(grid):
    # Placeholder for lined shapes identification
    return {}

def identify_corner_pointed_shapes(grid):
    # Placeholder for corner-pointed shapes identification
    return {}

import json
import os
import numpy as np
from scipy.ndimage import label, find_objects
import hashlib
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors

def plot_shape(shape_grid, ax, shape_name, count, color):
    if isinstance(color, int):
        cmap = plt.get_cmap('tab20')
        rgba_color = cmap(color % 20)
    elif color == 'N/A':
        rgba_color = (0.7, 0.7, 0.7, 1)  # Gray color for N/A
    else:
        try:
            rgba_color = mcolors.to_rgba(color)
        except ValueError:
            rgba_color = (0.7, 0.7, 0.7, 1)  # Default to gray if color is invalid

    ax.imshow(shape_grid, cmap=mcolors.ListedColormap([rgba_color]))
    ax.set_title(f"{shape_name}\nCount: {count}", fontsize=8)
    ax.axis('off')

def load_data(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def identify_shapes(grid):
    shapes = {}
    shapes.update(identify_contiguous_shapes(grid))
    shapes.update(identify_same_color_shapes(grid))
    shapes.update(identify_relationship_shapes(grid))
    shapes.update(identify_specific_shapes(grid))
    shapes.update(identify_primitives(grid))
    return shapes

def get_shapes_for_matrix(grid):
    shapes = identify_shapes(grid)
    return shapes

def grid_to_ascii(grid):
    return "\n".join("".join(str(cell) for cell in row) for row in grid)

def process_and_visualize_examples(file_paths, num_examples=3):
    for file_path in file_paths[:num_examples]:
        data = load_data(file_path)
        if not data:
            continue
        for item_index, item in enumerate(data['train']):
            input_grid = np.array(item['input'])
            output_grid = np.array(item['output'])
            
            print(f"Example {item_index + 1} ASCII Visualization:")
            print("Input:")
            print(grid_to_ascii(input_grid))
            print("Output:")
            print(grid_to_ascii(output_grid))
            # Additional shape processing and visualization here


def detect_square(grid, x, y, visited, current_color):
    max_size = min(len(grid) - x, len(grid[0]) - y)
    for size in range(1, max_size + 1):
        if all(grid[i][j] == current_color and (i, j) not in visited
               for i in range(x, x + size) for j in range(y, y + size)):
            for i in range(x, x + size):
                for j in range(y, y + size):
                    visited.add((i, j))
            return "square", size
    return None

def detect_rectangle(grid, x, y, visited, current_color):
    max_height = len(grid) - x
    max_width = len(grid[0]) - y
    for height in range(1, max_height + 1):
        for width in range(1, max_width + 1):
            if all(grid[i][j] == current_color and (i, j) not in visited
                   for i in range(x, x + height) for j in range(y, y + width)):
                for i in range(x, x + height):
                    for j in range(y, y + width):
                        visited.add((i, j))
                return "rectangle", (height, width)
    return None

def detect_bar(grid, x, y, visited, current_color, orientation):
    dx, dy = orientation
    length = 0
    i, j = x, y
    while 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == current_color and (i, j) not in visited:
        visited.add((i, j))
        length += 1
        i += dx
        j += dy
    return f"bar-{orientation}", length if length > 1 else None

def identify_same_color_shapes(grid):
    shapes = {}
    colors = set(cell for row in grid for cell in row if cell != 0)
    for color in colors:
        color_mask = [[1 if cell == color else 0 for cell in row] for row in grid]
        labeled, num_features = label(color_mask)
        for i in range(1, num_features + 1):
            shape_mask = labeled == i
            shape_size = np.sum(shape_mask)
            if shape_size == 1:
                continue  # Filter out 1x1 shapes
            shape_hash = hash_shape("same_color", shape_size, shape_size, color)
            if shape_hash in shapes:
                shapes[shape_hash]['count'] += 1
            else:
                shapes[shape_hash] = {'count': 1, 'details': {"type": "same_color", "color": color, "size": shape_size}}
    return shapes

def identify_specific_shapes(grid):
    shapes = {}
    shapes.update(identify_rings(grid))
    shapes.update(identify_filled_shapes(grid))
    shapes.update(identify_lined_shapes(grid))
    shapes.update(identify_corner_pointed_shapes(grid))
    return shapes

def hash_shape(shape_type, width, height, color):
    if shape_type == 'primitive':
        return f"primitive_{width}_{height}"
    shape_str = f"{shape_type}_{width}_{height}_{color}"
    return f"{hashlib.md5(shape_str.encode()).hexdigest()}_{width}_{height}"

def identify_same_color_shapes(grid):
    shapes = {}
    colors = set(cell for row in grid for cell in row if cell != 0)
    for color in colors:
        color_mask = [[1 if cell == color else 0 for cell in row] for row in grid]
        labeled, num_features = label(color_mask)
        for i in range(1, num_features + 1):
            shape_mask = labeled == i
            shape_size = np.sum(shape_mask)
            if shape_size == 1:
                continue  # Filter out 1x1 shapes
            shape_name = hash_shape("same_color", shape_mask.shape[1], shape_mask.shape[0], color)
            if shape_name in shapes:
                shapes[shape_name]['count'] += 1
            else:
                shapes[shape_name] = {'count': 1, 'details': {"type": "same_color", "color": color, "size": shape_size, "width": shape_mask.shape[1], "height": shape_mask.shape[0]}}
    return shapes

def identify_relationship_shapes(grid):
    shapes = {}
    # Identify four points forming a square
    for i in range(len(grid) - 1):
        for j in range(len(grid[0]) - 1):
            if grid[i][j] != 0 and grid[i][j+1] != 0 and grid[i+1][j] != 0 and grid[i+1][j+1] != 0:
                shape_name = hash_shape("four_point_square", 2, 2, None)
                if shape_name in shapes:
                    shapes[shape_name]['count'] += 1
                else:
                    shapes[shape_name] = {'count': 1, 'details': {"type": "four_point_square", "start": (i, j), "size": 2}}
    return shapes

def identify_rings(grid):
    shapes = {}
    for i in range(1, len(grid) - 1):
        for j in range(1, len(grid[0]) - 1):
            if grid[i][j] == 0 and all(grid[i+di][j+dj] != 0 for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]):
                shape_name = hash_shape("ring", 3, 3, None)
                if shape_name in shapes:
                    shapes[shape_name]['count'] += 1
                else:
                    shapes[shape_name] = {'count': 1, 'details': {"type": "ring", "center": (i, j), "size": 3}}
    return shapes

def identify_contiguous_shapes(grid):
    shapes = {}
    visited = set()
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if (i, j) not in visited:
                current_color = grid[i][j]
                if current_color == 0:  # Skip background color
                    continue
                shape = (detect_square(grid, i, j, visited, current_color) or
                         detect_rectangle(grid, i, j, visited, current_color) or
                         detect_bar(grid, i, j, visited, current_color, (1, 0)) or
                         detect_bar(grid, i, j, visited, current_color, (0, 1)) or
                         detect_bar(grid, i, j, visited, current_color, (1, 1)))
                if shape:
                    shape_type, shape_size = shape
                    if shape_type == 'square' and shape_size == 1:
                        continue  # Filter out 1x1 shapes
                    if shape_type == 'square':
                        shape_name = hash_shape(shape_type, shape_size, shape_size, current_color)
                    elif shape_type == 'rectangle':
                        height, width = shape_size
                        shape_name = hash_shape(shape_type, width, height, current_color)
                    elif shape_type.startswith('bar'):
                        shape_name = hash_shape(shape_type, shape_size, 1, current_color)
                    else:
                        continue  # Skip unknown shape types
                    if shape_name in shapes:
                        shapes[shape_name]['count'] += 1
                    else:
                        shapes[shape_name] = {'count': 1, 'details': {"type": shape_type, "start": (i, j), "size": shape_size, "color": current_color}}
    return shapes

def identify_filled_shapes(grid):
    # Placeholder for filled shapes identification
    return {}

def identify_lined_shapes(grid):
    # Placeholder for lined shapes identification
    return {}

def identify_corner_pointed_shapes(grid):
    # Placeholder for corner-pointed shapes identification
    return {}

def identify_primitives(grid):
    primitives = {}
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] != 0:
                primitive = hash_shape("primitive", 1, 1, grid[i][j])
                if primitive in primitives:
                    primitives[primitive]['count'] += 1
                else:
                    primitives[primitive] = {'count': 1, 'details': {"type": "primitive", "color": grid[i][j]}}
    return primitives

def compare_shapes(shape1, shape2):
    if shape1[0] != shape2[0]:  # Compare shape types
        return False
    
    shape_type = shape1[0]
    if shape_type in ['square', 'rectangle']:
        return compare_rectangles(shape1[1], shape2[1])
    elif shape_type.startswith('bar'):
        return compare_bars(shape1[1], shape2[1])
    elif shape_type == 'same_color':
        return shape1[1] == shape2[1] and compare_sizes(shape1[2], shape2[2])
    elif shape_type in ['four_point_square', 'ring']:
        return shape1[1] == shape2[1]
    elif shape_type == 'primitive':
        return shape1[1] == shape2[1]
    return False

def compare_sizes(size1, size2):
    return size1 == size2 or size1 == size2 * 2 or size1 * 2 == size2

def compare_rectangles(size1, size2):
    if isinstance(size1, int) and isinstance(size2, int):  # square
        return compare_sizes(size1, size2)
    elif isinstance(size1, tuple) and isinstance(size2, tuple):  # rectangle
        h1, w1 = size1
        h2, w2 = size2
        return ((h1 == h2 and w1 == w2) or 
                (h1 == w2 and w1 == h2) or 
                (h1 == h2*2 and w1 == w2*2) or 
                (h1*2 == h2 and w1*2 == w2))
    return False

def compare_bars(size1, size2):
    return compare_sizes(size1, size2)

def store_shapes(shapes, shape_storage, compare_func):
    for shape, details in shapes.items():
        matched = False
        for stored_shape in list(shape_storage.keys()):  # Use list() to avoid runtime error
            try:
                if compare_func(shape, stored_shape):
                    shape_storage[stored_shape]['count'] += details['count']
                    matched = True
                    break
            except Exception as e:
                print(f"Error comparing shapes: {shape} and {stored_shape}")
                print(f"Error details: {str(e)}")
        if not matched:
            shape_storage[shape] = details

def filter_shapes(shape_storage):
    # Only filter out shapes with fewer than 2 occurrences
    filtered_shapes = {shape: details for shape, details in shape_storage.items() if details['count'] >= 2}
    return filtered_shapes


def decompose_shape(shape_details):
    sub_shapes = {}
    shape_type = shape_details['type']
    color = shape_details.get('color')
    
    if shape_type == 'square':
        size = shape_details['size']
        x, y = shape_details['start']
        primitive = hash_shape("primitive", 1, 1, color)
        sub_shapes[primitive] = {'count': size * size, 'details': {"type": "primitive", "color": color, "positions": [(x+i, y+j) for i in range(size) for j in range(size)]}}
    elif shape_type == 'rectangle':
        height, width = shape_details['size']
        x, y = shape_details['start']
        primitive = hash_shape("primitive", 1, 1, color)
        sub_shapes[primitive] = {'count': height * width, 'details': {"type": "primitive", "color": color, "positions": [(x+i, y+j) for i in range(height) for j in range(width)]}}
    elif shape_type.startswith('bar'):
        length = shape_details['size']
        x, y = shape_details['start']
        dx, dy = eval(shape_type.split('-')[1])
        primitive = hash_shape("primitive", 1, 1, color)
        sub_shapes[primitive] = {'count': length, 'details': {"type": "primitive", "color": color, "positions": [(x+i*dx, y+i*dy) for i in range(length)]}}
    elif shape_type == 'same_color':
        primitive = hash_shape("primitive", 1, 1, color)
        sub_shapes[primitive] = {'count': shape_details['size'], 'details': {"type": "primitive", "color": color}}
    elif shape_type in ['filled_shape', 'lined_shape', 'corner_pointed_shape']:
        primitive = hash_shape("primitive", 1, 1, color)
        sub_shapes[primitive] = {'count': shape_details['size'] * shape_details['size'], 'details': {"type": "primitive", "color": color}}
    elif shape_type == 'primitive':
        # Try to decompose primitive into smaller primitives
        width, height = shape_details.get('width', 1), shape_details.get('height', 1)
        if width > 1 or height > 1:
            primitive = hash_shape("primitive", 1, 1, color)
            sub_shapes[primitive] = {'count': width * height, 'details': {"type": "primitive", "color": color}}
        else:
            primitive = hash_shape("primitive", width, height, color)
            sub_shapes[primitive] = {'count': 1, 'details': {"type": "primitive", "color": color}}
    
    return sub_shapes


def process_datasets(file_paths):
    shape_storage = {}
    file_occurrence = {}
    for file_path in file_paths:
        data = load_data(file_path)
        if not data:
            continue
        file_shapes = set()
        for item in data['train']:
            grid = item['input']
            shapes = identify_shapes(grid)
            for shape, details in shapes.items():
                if shape not in file_shapes:
                    file_shapes.add(shape)
                    if shape in file_occurrence:
                        file_occurrence[shape] += 1
                    else:
                        file_occurrence[shape] = 1
                if shape in shape_storage:
                    shape_storage[shape]['count'] += details['count']
                else:
                    shape_storage[shape] = details
    
    # Decompose all shapes
    decomposed_shapes = {}
    for shape, details in shape_storage.items():
        sub_shapes = decompose_shape(details['details'])
        for sub_shape, sub_details in sub_shapes.items():
            if sub_shape in decomposed_shapes:
                decomposed_shapes[sub_shape]['count'] += sub_details['count']
            else:
                decomposed_shapes[sub_shape] = sub_details
    
    # Remove shapes with count < 5
    filtered_shapes = {shape: details for shape, details in decomposed_shapes.items() if details['count'] >= 5}
    
    return filtered_shapes


def visualize_shape(shape_grid, shape_name, count, color):
    # Convert float grid to int grid
    int_grid = shape_grid.astype(int)
    if color != 'N/A':
        int_grid = int_grid * int(color)
    
    print(f"\n{shape_name} (Count: {count})")
    print(int_grid)  # Simple print for now, we'll improve this

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def color_pixel_count(grid):
    colors = {}
    for row in grid:
        for pixel in row:
            if pixel in colors:
                colors[pixel] += 1
            else:
                colors[pixel] = 1
    return colors

def process_task(task):
    # Initial processing and shape detection
    input_color_count = color_pixel_count(task['input'])
    output_color_count = color_pixel_count(task['output'])

    result['example_text'] = {
        'input_color_count': input_color_count,
        'output_color_count': output_color_count,
        'shapes': identified_shapes  # Include specific shape data as previously processed
    }

def main():
    # Load the training set
    train_set, _ = arckit.load_data()

    # Process all tasks
    process_all_tasks(train_set)

    # Get a specific task (e.g., the first one)
    task = train_set[0]  # You can change this index to get a different task

    print("Task:")
    task.show()  # This will show all train and test examples

    # Identify shapes for the first train example
    input_grid, _ = task.train[0]

    # Identify shapes
    shapes = identify_shapes(input_grid)

    print("\nIdentified Shapes:")
    for shape_name, shape_info in shapes.items():
        print(f"  {shape_name}: {shape_info['details']['type']}, Count: {shape_info['count']}, Color: {shape_info['details'].get('color', 'N/A')}")

    print("\nVisualization of Shapes and Subshapes:")
    for shape_name, shape_info in shapes.items():
        shape_type = shape_info['details']['type']
        count = shape_info['count']
        color = shape_info['details'].get('color', 'N/A')
        size = shape_info['details'].get('size', (1, 1))

        # Create shape grid
        if isinstance(size, int):
            shape_grid = np.ones((size, size), dtype=int)
        elif isinstance(size, tuple):
            shape_grid = np.ones(size, dtype=int)
        else:
            shape_grid = np.ones((1, 1), dtype=int)

        print(f"\nShape: {shape_name}")
        visualize_shape(shape_grid, shape_name, count, color)

        # Visualize subshapes
        sub_shapes = decompose_shape(shape_info['details'])
        for sub_name, sub_info in sub_shapes.items():
            sub_type = sub_info['details']['type']
            sub_count = sub_info['count']
            sub_color = sub_info['details'].get('color', 'N/A')
            sub_size = sub_info['details'].get('size', (1, 1))

            if isinstance(sub_size, int):
                sub_grid = np.ones((sub_size, sub_size), dtype=int)
            elif isinstance(sub_size, tuple):
                sub_grid = np.ones(sub_size, dtype=int)
            else:
                sub_grid = np.ones((1, 1), dtype=int)

            print(f"Subshape: {sub_name}")
            visualize_shape(sub_grid, sub_name, sub_count, sub_color)

if __name__ == "__main__":
    main()
