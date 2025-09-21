#!/usr/bin/env python3
"""
Face Grouper (Step 3 of Workflow)
=================================
Uses FaceNet embeddings to group similar faces into 5 clusters for easier review.
Moves PNG+YAML file pairs together while preserving all original filenames.

USAGE:
------
Activate virtual environment first:
  source venv/bin/activate

Run on directories containing individual images (after character sorting):
  python scripts/03_face_grouper.py "character_group_1"
  python scripts/03_face_grouper.py "00_Asian"

WORKFLOW POSITION:
------------------
Step 1: Image Version Selection → scripts/01_web_image_selector.py
Step 2: Character Sorting → scripts/02_web_character_sorter.py
Step 3: Face Grouping → THIS SCRIPT (scripts/03_face_grouper.py)
Step 4: Final Cropping → scripts/04_batch_crop_tool.py

OUTPUT STRUCTURE:
-----------------
The script creates face group directories at the PARENT level:

Before:
  source_directory/
  └── [mixed individual images]

After:
  face_group_1/        # Cluster 1 - similar faces
  face_group_2/        # Cluster 2 - similar faces  
  face_group_3/        # Cluster 3 - similar faces
  face_group_4/        # Cluster 4 - similar faces
  face_group_5/        # Cluster 5 - similar faces
  face_group_not/      # Images without detectable faces
  source_directory/    # Now empty (all files moved out)

FEATURES:
---------
- Detects faces using OpenCV Haar Cascades
- Groups similar faces using K-means clustering (5 groups)
- Handles PNG+YAML pairs automatically
- Creates clean separation for efficient manual review
- Completely standalone - no dependencies on other scripts
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from file_tracker import FileTracker

# Required imports
try:
    import cv2
    from sklearn.cluster import KMeans
    from deepface import DeepFace
except ImportError as e:
    print(f"❌ Missing required packages. Install with:")
    print(f"pip install opencv-python scikit-learn deepface")
    print(f"Error: {e}")
    sys.exit(1)

class FaceGrouper:
    def __init__(self, embedding_model="VGG-Face", detector_backend="opencv", 
                 enforce_detection=True, normalization="base"):
        """Initialize the Face Grouper with DeepFace.
        
        Args:
            embedding_model: Model for face embeddings
                Options: "VGG-Face", "Facenet", "OpenFace", "DeepID", "ArcFace", "Dlib", "SFace"
            detector_backend: Face detection method
                Options: "opencv", "retinaface", "mtcnn", "ssd", "dlib", "mediapipe"
            enforce_detection: If True, fail when no face detected. If False, try fallback
            normalization: Embedding normalization method
                Options: "base", "raw", "Facenet", "Facenet2018", "VGGFace", "VGGFace2", "ArcFace"
        """
        print("🔄 Loading DeepFace models...")
        
        self.embedding_model = embedding_model
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        self.normalization = normalization
        
        print(f"✅ DeepFace initialized!")
        print(f"   📊 Model: {embedding_model}")
        print(f"   🔍 Detector: {detector_backend}")
        print(f"   🎯 Enforce Detection: {enforce_detection}")
        print(f"   🔧 Normalization: {normalization}")
        
    def extract_face(self, image_path: Path) -> np.ndarray:
        """Extract face embedding using DeepFace."""
        try:
            # Use DeepFace to extract embeddings
            # This automatically handles face detection and embedding generation
            embeddings = DeepFace.represent(
                img_path=str(image_path),
                model_name=self.embedding_model,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                normalization=self.normalization
            )
            
            # DeepFace returns a list of embeddings (one per detected face)
            if not embeddings or len(embeddings) == 0:
                return None

            # Use the first detected face (usually the largest/most prominent)
            first_result = embeddings[0]

            # DeepFace has changed the embedding key name a few times between releases.
            # Try the known options before giving up so we don't unnecessarily fall back
            # to the OpenCV heuristic embedding (which is much less accurate).
            embedding_vector = None
            for key in ("embedding", "face_embedding", "vector"):
                if key in first_result:
                    embedding_vector = first_result[key]
                    break

            if embedding_vector is None:
                raise KeyError("No embedding vector found in DeepFace response")

            embedding = np.asarray(embedding_vector, dtype=np.float32)

            # Flatten possible nested structures (just in case) so clustering works reliably
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
            return embedding
            
        except Exception as e:
            # If DeepFace fails, try fallback detection
            print(f"⚠️  DeepFace failed for {image_path.name}: {e}")
            return self._enhanced_opencv_detection(image_path)
    
    def _enhanced_opencv_detection(self, image_path: Path) -> np.ndarray:
        """Enhanced OpenCV-based face detection and embedding."""
        try:
            # Load OpenCV cascade if not already loaded
            if not hasattr(self, 'face_cascade'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
                
            # Convert to RGB and grayscale
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50)
            )
            
            if len(faces) == 0:
                return None
                
            # Use the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract and resize face
            face_img = rgb_img[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (160, 160))
            
            # Create simple embedding
            return self._create_simple_embedding(face_resized)
            
        except Exception as e:
            print(f"❌ Fallback detection failed for {image_path.name}: {e}")
            return None
    
    def _create_simple_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Create a simple face embedding using image statistics."""
        # This is a simplified approach - in production you'd use FaceNet
        
        # Calculate various statistical features
        features = []
        
        # Color channel means
        features.extend(np.mean(face_img, axis=(0, 1)))
        
        # Color channel standard deviations  
        features.extend(np.std(face_img, axis=(0, 1)))
        
        # Histograms for each channel (simplified)
        for channel in range(3):
            hist, _ = np.histogram(face_img[:, :, channel], bins=8, range=(0, 256))
            features.extend(hist / np.sum(hist))  # Normalize
            
        # Texture features (simplified)
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray_face, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # Convert to numpy array
        embedding = np.array(features, dtype=np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding

def scan_images(directory: Path) -> List[Path]:
    """Scan directory for PNG files."""
    png_files = sorted(directory.glob("*.png"))
    print(f"📁 Found {len(png_files)} PNG files in {directory}")
    return png_files

def group_faces_into_clusters(face_data: List[Tuple[Path, np.ndarray]], n_clusters: int = 5, 
                             clustering_method: str = "kmeans", eps: float = 0.5, 
                             min_samples: int = 2) -> Dict[int, List[Path]]:
    """Group face embeddings into clusters using specified method.
    
    Args:
        face_data: List of (image_path, embedding) tuples
        n_clusters: Number of clusters for K-means (ignored for DBSCAN)
        clustering_method: "kmeans" or "dbscan"
        eps: Maximum distance for DBSCAN clustering (smaller = stricter)
        min_samples: Minimum samples per cluster for DBSCAN
    """
    if not face_data:
        return {}
        
    if len(face_data) < n_clusters and clustering_method.lower() == "kmeans":
        # If we have fewer faces than clusters, put each in its own group
        clusters = {}
        for i, (image_path, _) in enumerate(face_data):
            clusters[i] = [image_path]
        return clusters
    
    # Extract embeddings and handle different embedding sizes
    embeddings_list = [embedding for _, embedding in face_data]
    
    # Check if all embeddings have the same shape
    if len(set(len(emb) for emb in embeddings_list)) > 1:
        print("⚠️  Mixed embedding sizes detected, normalizing...")
        # Find the most common embedding size
        sizes = [len(emb) for emb in embeddings_list]
        most_common_size = max(set(sizes), key=sizes.count)
        # Filter to only embeddings of the most common size
        filtered_data = [(path, emb) for (path, emb) in face_data if len(emb) == most_common_size]
        embeddings = np.array([embedding for _, embedding in filtered_data])
        image_paths = [image_path for image_path, _ in filtered_data]
        print(f"✅ Using {len(filtered_data)} faces with {most_common_size}-dimensional embeddings")
    else:
        embeddings = np.array(embeddings_list)
        image_paths = [image_path for image_path, _ in face_data]
    
    if clustering_method.lower() == "dbscan":
        from sklearn.cluster import DBSCAN
        # Normalize embeddings for better DBSCAN performance
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(embeddings_normalized)
        
        # Group by clusters (note: -1 is noise/outliers)
        unique_labels = set(cluster_labels)
        clusters = {}
        
        for label in unique_labels:
            if label == -1:
                # Noise/outliers go to special group
                clusters['noise'] = []
            else:
                clusters[label] = []
        
        for image_path, cluster_id in zip(image_paths, cluster_labels):
            if cluster_id == -1:
                clusters['noise'].append(image_path)
            else:
                clusters[cluster_id].append(image_path)
                
        print(f"🔍 DBSCAN found {len(unique_labels) - (1 if -1 in unique_labels else 0)} clusters")
        if -1 in unique_labels:
            print(f"   🚫 {len(clusters['noise'])} outlier faces")
            
    else:  # K-means (default)
        # Handle edge cases for K-means
        if n_clusters == 1:
            clusters = {0: [image_path for image_path, _ in face_data]}
            return clusters
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group by clusters
        clusters = {}
        for i in range(n_clusters):
            clusters[i] = []
        
        for image_path, cluster_id in zip(image_paths, cluster_labels):
            clusters[cluster_id].append(image_path)
    
    return clusters

def move_files_to_groups(clusters: Dict[int, List[Path]], no_face_files: List[Path], source_dir: Path, tracker: FileTracker = None):
    """Move PNG+YAML files to their respective group directories."""
    moved_count = 0
    group_counts = {}
    
    # Log initial state and start batch
    if tracker:
        tracker.log_directory_state(str(source_dir), "Before face grouping")
        tracker.log_batch_start("Face grouping file movement")
    
    # Create group directories at parent level (same level as normal_images)
    parent_dir = source_dir.parent
    for group_id in range(1, 6):  # face_group_1 through face_group_5
        group_dir = parent_dir / f"face_group_{group_id}"
        group_dir.mkdir(exist_ok=True)
    
    # Create face_group_not directory at parent level
    no_face_dir = parent_dir / "face_group_not"
    no_face_dir.mkdir(exist_ok=True)
    
    # Move clustered faces
    for cluster_id, image_paths in clusters.items():
        group_dir = parent_dir / f"face_group_{cluster_id + 1}"
        group_name = f"face_group_{cluster_id + 1}"
        files_moved_to_group = []
        group_counts[group_name] = 0
        
        for png_path in image_paths:
            # Move PNG file
            try:
                target_png = group_dir / png_path.name
                shutil.move(str(png_path), str(target_png))
                moved_count += 1
                group_counts[group_name] += 1
                files_moved_to_group.append(png_path.name)
                print(f"✓ Moved {png_path.name} → {group_name}")
                
                # Move corresponding YAML file
                yaml_path = png_path.parent / f"{png_path.stem}.yaml"
                if yaml_path.exists():
                    target_yaml = group_dir / yaml_path.name
                    shutil.move(str(yaml_path), str(target_yaml))
                    group_counts[group_name] += 1
                    files_moved_to_group.append(yaml_path.name)
                    print(f"✓ Moved {yaml_path.name} → {group_name}")
                    
            except Exception as e:
                print(f"❌ Error moving {png_path.name}: {e}")
        
        # Log the group movement
        if tracker and files_moved_to_group:
            tracker.log_operation(
                operation="move",
                source_dir=str(source_dir.name),
                dest_dir=group_name,
                file_count=len(files_moved_to_group),
                files=files_moved_to_group[:10],  # Only log first 10 to avoid huge logs
                notes=f"Face clustering - cluster {cluster_id + 1}"
            )
    
    # Move files with no detected faces
    group_counts["face_group_not"] = 0
    no_face_files_moved = []
    for png_path in no_face_files:
        try:
            target_png = no_face_dir / png_path.name
            shutil.move(str(png_path), str(target_png))
            moved_count += 1
            group_counts["face_group_not"] += 1
            no_face_files_moved.append(png_path.name)
            print(f"✓ Moved {png_path.name} → face_group_not")
            
            # Move corresponding YAML file
            yaml_path = png_path.parent / f"{png_path.stem}.yaml"
            if yaml_path.exists():
                target_yaml = no_face_dir / yaml_path.name
                shutil.move(str(yaml_path), str(target_yaml))
                group_counts["face_group_not"] += 1
                no_face_files_moved.append(yaml_path.name)
                print(f"✓ Moved {yaml_path.name} → face_group_not")
                
        except Exception as e:
            print(f"❌ Error moving {png_path.name}: {e}")
    
    # Log no-face files movement
    if tracker and no_face_files_moved:
        tracker.log_operation(
            operation="move",
            source_dir=str(source_dir.name),
            dest_dir="face_group_not",
            file_count=len(no_face_files_moved),
            files=no_face_files_moved[:10],
            notes="No detectable faces"
        )
    
    # Log final states and end batch
    if tracker:
        for group_id in range(1, 6):
            group_dir = parent_dir / f"face_group_{group_id}"
            if group_dir.exists():
                tracker.log_directory_state(str(group_dir), "After face grouping")
        tracker.log_directory_state(str(no_face_dir), "After face grouping")
        tracker.log_directory_state(str(source_dir), "After face grouping")
        tracker.log_batch_end(f"Moved {moved_count} files to face groups")
    
    return moved_count, group_counts

def main():
    parser = argparse.ArgumentParser(description='Group similar faces with tunable parameters')
    parser.add_argument('directory', help='Directory containing PNG files to group')
    parser.add_argument('--model', default='VGG-Face', 
                       choices=['VGG-Face', 'Facenet', 'OpenFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace'],
                       help='Face embedding model (default: VGG-Face)')
    parser.add_argument('--detector', default='opencv',
                       choices=['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe'],
                       help='Face detection backend (default: opencv)')
    parser.add_argument('--clustering', default='kmeans', choices=['kmeans', 'dbscan'],
                       help='Clustering algorithm (default: kmeans)')
    parser.add_argument('--clusters', type=int, default=5,
                       help='Number of clusters for K-means (default: 5)')
    parser.add_argument('--eps', type=float, default=0.5,
                       help='DBSCAN eps parameter - smaller = stricter grouping (default: 0.5)')
    parser.add_argument('--min-samples', type=int, default=2,
                       help='DBSCAN min_samples parameter (default: 2)')
    parser.add_argument('--strict', action='store_true',
                       help='Strict mode: enforce face detection (fail if no face)')
    parser.add_argument('--normalization', default='base',
                       choices=['base', 'raw', 'Facenet', 'Facenet2018', 'VGGFace', 'VGGFace2', 'ArcFace'],
                       help='Embedding normalization method (default: base)')
    
    args = parser.parse_args()
    
    # Initialize file tracker
    tracker = FileTracker("face_grouper")
    
    source_dir = Path(args.directory).expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"❌ Error: {source_dir} is not a directory")
        sys.exit(1)
    
    print(f"🎯 Face Grouper v3.1 - Tunable Parameters")
    print(f"📁 Source directory: {source_dir}")
    print(f"🔧 Configuration:")
    print(f"   📊 Model: {args.model}")
    print(f"   🔍 Detector: {args.detector}")
    print(f"   🎯 Clustering: {args.clustering}")
    if args.clustering == 'kmeans':
        print(f"   📈 Clusters: {args.clusters}")
    else:
        print(f"   📏 Eps: {args.eps} (smaller = stricter)")
        print(f"   👥 Min samples: {args.min_samples}")
    print(f"   ⚡ Strict detection: {args.strict}")
    print(f"   🔧 Normalization: {args.normalization}")
    
    # Initialize face grouper with custom parameters
    grouper = FaceGrouper(
        embedding_model=args.model,
        detector_backend=args.detector,
        enforce_detection=args.strict,
        normalization=args.normalization
    )
    
    # Scan for images
    png_files = scan_images(source_dir)
    if not png_files:
        print("❌ No PNG files found")
        sys.exit(1)
    
    # Extract face embeddings
    print(f"\n🔍 Extracting faces and creating embeddings...")
    face_data = []
    no_face_files = []
    
    for i, png_path in enumerate(png_files, 1):
        print(f"\rProcessing: {i}/{len(png_files)} ({(i/len(png_files)*100):.1f}%)", end="")
        
        embedding = grouper.extract_face(png_path)
        if embedding is not None:
            face_data.append((png_path, embedding))
        else:
            no_face_files.append(png_path)
    
    print(f"\n✅ Face extraction complete!")
    print(f"   • Faces detected: {len(face_data)}")
    print(f"   • No face detected: {len(no_face_files)}")
    
    if len(face_data) == 0:
        print("❌ No faces detected in any images")
        sys.exit(1)
    
    # Group faces into clusters
    if args.clustering == 'kmeans':
        print(f"\n🔄 Grouping faces into {args.clusters} clusters using K-means...")
        clusters = group_faces_into_clusters(face_data, n_clusters=args.clusters, 
                                           clustering_method='kmeans')
    else:
        print(f"\n🔄 Grouping faces using DBSCAN (eps={args.eps}, min_samples={args.min_samples})...")
        clusters = group_faces_into_clusters(face_data, clustering_method='dbscan', 
                                           eps=args.eps, min_samples=args.min_samples)
    
    # Show cluster summary
    print(f"\n📊 Clustering Results:")
    for group_id in sorted(clusters.keys()):
        count = len(clusters[group_id])
        if group_id == 'noise':
            print(f"   • face_group_outliers: {count} images")
        else:
            print(f"   • face_group_{group_id + 1}: {count} images")
    print(f"   • face_group_not: {len(no_face_files)} images")
    
    # Ask for confirmation
    total_files = len(png_files)
    while True:
        choice = input(f"\nMove {total_files} files to face groups? (y/n/q): ").lower().strip()
        if choice in ['y', 'yes']:
            break
        elif choice in ['n', 'no']:
            print("📋 Face grouping analysis complete. No files were moved.")
            sys.exit(0)
        elif choice in ['q', 'quit']:
            print("👋 Exiting without changes.")
            sys.exit(0)
        else:
            print("Please enter 'y' (yes), 'n' (no), or 'q' (quit)")
    
    # Move files to groups
    print(f"\n📦 Moving files to face groups...")
    moved_count, group_counts = move_files_to_groups(clusters, no_face_files, source_dir, tracker)
    
    print(f"\n✅ Face grouping complete!")
    print(f"📊 Summary:")
    print(f"   • Total files processed: {total_files}")
    print(f"   • Files moved: {moved_count}")
    print(f"   • Face groups created: 5")
    print(f"   • Directory: {source_dir}")
    print(f"📈 Group breakdown:")
    for group_name, count in group_counts.items():
        print(f"   • {group_name}: {count} files")

if __name__ == "__main__":
    main()
