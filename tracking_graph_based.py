import os
import glob
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class SingleTrack:
    positions: List[Tuple[int, int]]
    areas: List[float]
    frame_indices: List[int]
    motion_pattern: str = "unknown"  # "linear", "sinusoidal", etc.

    @property
    def is_active(self) -> bool:
        return len(self.positions) > 0

    @property
    def last_position(self) -> Tuple[int, int]:
        return self.positions[-1]

    @property
    def last_frame(self) -> int:
        return self.frame_indices[-1]

    @property
    def velocity(self) -> Tuple[float, float]:
        if len(self.positions) < 2:
            return (0, 0)
        return (
            self.positions[-1][0] - self.positions[-2][0],
            self.positions[-1][1] - self.positions[-2][1]
        )

    def predict_next_position(self) -> Tuple[int, int]:
        # For sinusoidal motion, look at more history
        if len(self.positions) < 3:
            return self.last_position
            
        # Calculate acceleration (change in velocity)
        last_vel = (self.positions[-1][0] - self.positions[-2][0],
                self.positions[-1][1] - self.positions[-2][1])
        prev_vel = (self.positions[-2][0] - self.positions[-3][0],
                self.positions[-2][1] - self.positions[-3][1])
        accel = (last_vel[0] - prev_vel[0], last_vel[1] - prev_vel[1])
        
        # Apply both velocity and acceleration
        return (
            int(self.last_position[0] + last_vel[0] + 0.5 * accel[0]),
            int(self.last_position[1] + last_vel[1] + 0.5 * accel[1])
        )
    
    def classify_motion_pattern(self, window=10):
        """Analyze motion pattern of this track."""
        if len(self.positions) < window:
            return "unknown"
            
        # Calculate directional changes
        directions = []
        for i in range(2, len(self.positions)):
            v1 = (self.positions[i-1][0] - self.positions[i-2][0], 
                 self.positions[i-1][1] - self.positions[i-2][1])
            v2 = (self.positions[i][0] - self.positions[i-1][0],
                 self.positions[i][1] - self.positions[i-1][1])
            
            # Use dot product to detect direction change
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            if dot < 0:
                directions.append("change")
            else:
                directions.append("same")
                
        # Count direction changes
        changes = directions.count("change")
        
        # Classify based on direction changes
        if changes > window * 0.3:  # More than 30% direction changes
            return "sinusoidal"
        else:
            return "linear"

def graph_based_tracking(csv_files, max_distance=20, min_area=1):
    """Track objects using a graph-based approach for the entire sequence."""
    import networkx as nx
    from scipy.spatial.distance import cdist
    
    # Initialize graph
    track_graph = nx.DiGraph()
    
    print("Building initial detection nodes...")
    # Step 1: Create nodes for each detection in each frame
    all_detections = []
    for frame_idx, csv_file in tqdm(enumerate(csv_files)):
        df = pd.read_csv(csv_file)
        frame_detections = []
        for _, row in df.iterrows():
            if row['area'] >= min_area:
                # Create unique node ID
                node_id = f"f{frame_idx}_x{int(row['x'])}_y{int(row['y'])}"
                # Store detection data
                detection = {
                    'frame': frame_idx,
                    'x': int(row['x']),
                    'y': int(row['y']),
                    'area': row['area'],
                    'node_id': node_id
                }
                # Add node to graph
                track_graph.add_node(node_id, **detection)
                frame_detections.append(detection)
        all_detections.append(frame_detections)
    
    print("Building candidate edges between frames...")
    # Step 2: Connect detections between consecutive frames
    for frame_idx in range(len(all_detections) - 1):
        current_detections = all_detections[frame_idx]
        next_detections = all_detections[frame_idx + 1]
        
        if not current_detections or not next_detections:
            continue
        
        # Calculate distance matrix between all detections in consecutive frames
        current_positions = np.array([[d['x'], d['y']] for d in current_detections])
        next_positions = np.array([[d['x'], d['y']] for d in next_detections])
        
        # Distance matrix between all points in consecutive frames
        dists = cdist(current_positions, next_positions)
        
        # Add edges for potential connections (if distance < max_distance)
        for i, current in enumerate(current_detections):
            for j, next_det in enumerate(next_detections):
                if dists[i, j] <= max_distance:
                    # Edge weight is inverse of distance (closer = stronger connection)
                    weight = 1.0 / (dists[i, j] + 0.1)  # +0.1 to avoid division by zero
                    # Add directed edge from current to next detection
                    track_graph.add_edge(
                        current['node_id'], 
                        next_det['node_id'], 
                        weight=weight, 
                        distance=dists[i, j]
                    )
    
    print("Analyzing motion patterns...")
    # Step 3: Analyze motion patterns in candidate paths
    # For each node with 2+ predecessors, look at motion patterns
    for node in track_graph.nodes():
        predecessors = list(track_graph.predecessors(node))
        if len(predecessors) >= 2:
            # Multiple incoming edges - analyze motion pattern along each path
            for pred in predecessors:
                # Try to trace back several steps (for motion pattern detection)
                path = [node, pred]
                current = pred
                # Trace back up to 5 steps if possible
                for _ in range(5):
                    pred_preds = list(track_graph.predecessors(current))
                    if not pred_preds:
                        break
                    # For simplicity, choose the strongest connection
                    best_pred = max(pred_preds, key=lambda p: track_graph[p][current]['weight'])
                    path.append(best_pred)
                    current = best_pred
                
                # If we have a long enough path, analyze motion pattern
                if len(path) >= 4:
                    # Extract positions along path
                    positions = [
                        (track_graph.nodes[n]['x'], track_graph.nodes[n]['y']) 
                        for n in reversed(path)
                    ]
                    # Check for sinusoidal pattern
                    is_sinusoidal = analyze_path_sinusoidal(positions)
                    
                    # Store path motion pattern on the edge
                    from_node = path[1]
                    to_node = path[0]
                    track_graph[from_node][to_node]['motion_pattern'] = (
                        'sinusoidal' if is_sinusoidal else 'linear'
                    )
    
    print("Resolving tracks using path optimization...")
    # Step 4: Resolve tracks by optimizing paths through the graph
    final_tracks = resolve_tracks(track_graph)
    
    return final_tracks

def analyze_path_sinusoidal(positions, window=5):
    """Analyze if a path shows sinusoidal motion pattern."""
    if len(positions) < window:
        return False
        
    # Calculate directional changes using dot product
    directions = []
    for i in range(2, len(positions)):
        v1 = (positions[i-1][0] - positions[i-2][0], positions[i-1][1] - positions[i-2][1])
        v2 = (positions[i][0] - positions[i-1][0], positions[i][1] - positions[i-1][1])
        
        # Use dot product to detect direction change
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag_v1 = (v1[0]**2 + v1[1]**2)**0.5
        mag_v2 = (v2[0]**2 + v2[1]**2)**0.5
        
        # Avoid division by zero
        if mag_v1 * mag_v2 > 0:
            # Normalized dot product (cosine of angle)
            cos_angle = dot / (mag_v1 * mag_v2)
            if cos_angle < 0:
                directions.append("change")
            else:
                directions.append("same")
    
    # Count direction changes
    changes = directions.count("change")
    
    # Classify based on direction changes (30% threshold for sinusoidal)
    return changes > (len(directions) * 0.3)

def resolve_tracks(graph):
    """Resolve final tracks from the graph, optimizing for continuous tracks."""
    import networkx as nx
    
    # Find nodes with no incoming edges (track starts)
    start_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    
    final_tracks = []
    processed_nodes = set()
    
    # Process each start node
    for start in start_nodes:
        if start in processed_nodes:
            continue
            
        # Start a new track
        current_track = SingleTrack(
            positions=[], 
            areas=[], 
            frame_indices=[]
        )
        
        # Add starting node
        node_data = graph.nodes[start]
        current_track.positions.append((node_data['x'], node_data['y']))
        current_track.areas.append(node_data['area'])
        current_track.frame_indices.append(node_data['frame'])
        processed_nodes.add(start)
        
        # Follow the path, prioritizing sinusoidal motion patterns
        current = start
        while True:
            successors = list(graph.successors(current))
            if not successors:
                break
                
            # Prioritize edges with sinusoidal pattern
            # Then by weight (proximity)
            best_successor = None
            best_score = -float('inf')
            
            for succ in successors:
                edge_data = graph[current][succ]
                # Base score is the edge weight
                score = edge_data['weight']
                
                # Bonus for sinusoidal pattern (if we've classified it)
                if edge_data.get('motion_pattern') == 'sinusoidal':
                    score *= 1.5  # 50% bonus for sinusoidal
                
                if score > best_score:
                    best_score = score
                    best_successor = succ
            
            # If we found a valid successor
            if best_successor:
                # Add to track
                node_data = graph.nodes[best_successor]
                current_track.positions.append((node_data['x'], node_data['y']))
                current_track.areas.append(node_data['area'])
                current_track.frame_indices.append(node_data['frame'])
                processed_nodes.add(best_successor)
                current = best_successor
            else:
                break
        
        # Only add tracks with sufficient length
        if len(current_track.positions) >= 3:
            # Classify the final track
            if len(current_track.positions) > 5:
                current_track.motion_pattern = current_track.classify_motion_pattern()
            final_tracks.append(current_track)
    
    # Second pass: resolve any remaining unprocessed nodes
    remaining = [n for n in graph.nodes() if n not in processed_nodes]
    
    if remaining:
        # Handle remaining nodes by finding connected components
        subgraph = graph.subgraph(remaining)
        components = list(nx.weakly_connected_components(subgraph))
        
        for component in components:
            # Find temporal ordering within component
            nodes_by_frame = {}
            for node in component:
                frame = graph.nodes[node]['frame']
                if frame not in nodes_by_frame:
                    nodes_by_frame[frame] = []
                nodes_by_frame[frame].append(node)
            
            # Sort frames
            frames = sorted(nodes_by_frame.keys())
            
            # If this component spans multiple frames
            if len(frames) > 1:
                # Start a new track from earliest frame
                start_frame = frames[0]
                # Choose node with highest weight sum of outgoing edges
                if len(nodes_by_frame[start_frame]) > 1:
                    start_node = max(nodes_by_frame[start_frame], 
                                     key=lambda n: sum(graph[n][s]['weight'] 
                                                       for s in graph.successors(n) 
                                                       if s in component))
                else:
                    start_node = nodes_by_frame[start_frame][0]
                
                # Create track similar to above process
                current_track = SingleTrack(
                    positions=[], 
                    areas=[], 
                    frame_indices=[]
                )
                
                # Add starting node
                node_data = graph.nodes[start_node]
                current_track.positions.append((node_data['x'], node_data['y']))
                current_track.areas.append(node_data['area'])
                current_track.frame_indices.append(node_data['frame'])
                processed_nodes.add(start_node)
                
                # Follow best path through remaining frames
                current = start_node
                for frame in frames[1:]:
                    candidates = [n for n in nodes_by_frame[frame] if n in component]
                    if not candidates:
                        continue
                        
                    # Find best candidate based on path from current
                    best_candidate = None
                    best_path_value = -float('inf')
                    
                    for candidate in candidates:
                        try:
                            # Try to find a path
                            path = nx.shortest_path(
                                graph, current, candidate, weight='distance'
                            )
                            # Compute path value (lower distance is better)
                            path_dist = sum(graph[path[i]][path[i+1]]['distance'] 
                                          for i in range(len(path)-1))
                            path_value = 1.0 / (path_dist + 0.1)
                            
                            if path_value > best_path_value:
                                best_path_value = path_value
                                best_candidate = candidate
                        except nx.NetworkXNoPath:
                            continue
                    
                    if best_candidate:
                        node_data = graph.nodes[best_candidate]
                        current_track.positions.append((node_data['x'], node_data['y']))
                        current_track.areas.append(node_data['area'])
                        current_track.frame_indices.append(node_data['frame'])
                        processed_nodes.add(best_candidate)
                        current = best_candidate
                
                # Only add if track is substantial
                if len(current_track.positions) >= 3:
                    if len(current_track.positions) > 5:
                        current_track.motion_pattern = current_track.classify_motion_pattern()
                    final_tracks.append(current_track)
    
    # Return all resolved tracks
    return [t for t in final_tracks if len(t.positions) >= 3]

def save_tracks(tracks: List[SingleTrack], output_csv: str):
    """Save tracks to a CSV file."""
    rows = []
    for tid, track in enumerate(tracks):
        for pos, area, frame in zip(track.positions, track.areas, track.frame_indices):
            rows.append({
                'track_id': tid,
                'frame': frame,
                'x': pos[0],
                'y': pos[1],
                'area': area
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

def visualize_tracks(
    tracks: List[SingleTrack],
    image_dir: str,
    prefix: str,
    output_dir: str,
    debug_frames: List[int] = None,  # Add frames to debug (e.g., [36])
    target_tracks: List[int] = None  # Track IDs to focus on (e.g., [14, 360])
):
    """Visualize and save annotated images with tracks."""
    print(f"Visualizing {len(tracks)} tracks...")
    os.makedirs(output_dir, exist_ok=True)
    # Assume images are named as in the detection CSVs
    image_files = sorted(glob.glob(os.path.join(image_dir, f"{prefix}_*.png")))
    # Build a mapping from frame index to image file
    frame_to_img = {i: f for i, f in enumerate(image_files)}
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(tracks))]
    
    # Get all frame indices
    all_frames = sorted(list(frame_to_img.keys()))
    
    for frame_idx, img_file in tqdm(frame_to_img.items()):
        img = cv2.imread(img_file)
        if img is None:
            continue
            
        # Draw current positions and tracks
        for tid, track in enumerate(tracks):
            # Skip if not a target track when target_tracks is specified
            if target_tracks and tid not in target_tracks:
                continue
                
            # Draw current position if present in this frame
            if frame_idx in track.frame_indices:
                idx = track.frame_indices.index(frame_idx)
                pos = track.positions[idx]
                cv2.circle(img, pos, 4, colors[tid], -1)
                cv2.putText(img, f"T{tid}", (pos[0]+5, pos[1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[tid], 1)
                
                # Draw trajectory
                for j in range(1, len(track.positions)):
                    if track.frame_indices[j] > frame_idx:
                        break
                    if track.frame_indices[j-1] <= frame_idx:
                        pt1 = track.positions[j-1]
                        pt2 = track.positions[j]
                        cv2.line(img, pt1, pt2, colors[tid], 2)
            
            # Draw prediction for next frame (if track exists in previous frame)
            if frame_idx > 0 and frame_idx-1 in track.frame_indices:
                # Find the state of the track in the previous frame
                prev_idx = track.frame_indices.index(frame_idx-1)
                
                # Create a temporary track with data up to previous frame
                temp_track = SingleTrack(
                    positions=track.positions[:prev_idx+1],
                    areas=track.areas[:prev_idx+1],
                    frame_indices=track.frame_indices[:prev_idx+1]
                )
                
                # Get the prediction
                pred_pos = temp_track.predict_next_position()
                
                # Draw the prediction as an X
                cv2.drawMarker(img, pred_pos, colors[tid], cv2.MARKER_CROSS, 8, 1)
                
                # If track appears in current frame, connect prediction to actual
                if frame_idx in track.frame_indices:
                    cur_idx = track.frame_indices.index(frame_idx)
                    actual_pos = track.positions[cur_idx]
                    # Draw a dashed line between prediction and actual
                    cv2.line(img, pred_pos, actual_pos, colors[tid], 1, cv2.LINE_AA)
                    
                    # Calculate distance for display
                    dist = np.sqrt((pred_pos[0] - actual_pos[0])**2 + 
                                  (pred_pos[1] - actual_pos[1])**2)
                    
                    # Add distance text if this is a debug frame or involves target tracks
                    is_debug = (debug_frames and frame_idx in debug_frames) or \
                              (target_tracks and tid in target_tracks)
                    if is_debug:
                        cv2.putText(img, f"d={dist:.1f}", 
                                   ((pred_pos[0] + actual_pos[0])//2, 
                                    (pred_pos[1] + actual_pos[1])//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        
        # Add frame number to the image
        cv2.putText(img, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add extra debugging for critical frames
        if debug_frames and frame_idx in debug_frames:
            cv2.putText(img, f"DEBUG FRAME", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
        out_path = os.path.join(output_dir, os.path.basename(img_file))
        cv2.imwrite(out_path, img)

if __name__ == "__main__":
    import sys
    import glob
    prefix = 'upper'
    input_dir = "/Users/vdausmann/swimming_in_salt/detection_results/20250312_21.2_01"
    csv_files = sorted(glob.glob(os.path.join(input_dir, f"{prefix}_*_objects.csv")))
    
    # Use graph-based tracking instead of sequential tracking
    tracks = graph_based_tracking(csv_files, max_distance=30)
    
    # Filter for sinusoidal tracks if desired
    sinusoidal_tracks = [t for t in tracks if t.motion_pattern == "sinusoidal"]
    
    # Save all tracks
    save_tracks(tracks, os.path.join(input_dir, f"{prefix}_graph_tracks.csv"))
    
    # Save only sinusoidal tracks (oyster larvae)
    save_tracks(sinusoidal_tracks, os.path.join(input_dir, f"{prefix}_larvae_tracks.csv"))
    
    # Visualize tracks with different colors based on motion pattern
    visualize_tracks(
        tracks,
        image_dir=input_dir,
        prefix=prefix,
        output_dir=os.path.join(input_dir, f"{prefix}_graph_tracks_vis"),
        #target_tracks= [0]
    )