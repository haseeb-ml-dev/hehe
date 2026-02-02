import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import os
import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
from PIL import Image
import logging
from collections import Counter
import joblib

logger = logging.getLogger(__name__)


def get_resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and PyInstaller bundle."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # Running in normal Python environment
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

class EnhancedBicycleClassifier:
    def _resolve_model_path(self, model_save_path):
        """Try multiple locations to find bicycle models"""
        candidates = [
            model_save_path,  # As passed
            os.path.abspath(model_save_path),  # Absolute path
            os.path.join(os.path.dirname(__file__), 'bicycle_models'),  # Relative to this file
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'bicycle_models'),  # src/bicycle_models
        ]
        
        # Try PyInstaller bundle location
        try:
            import sys
            if hasattr(sys, '_MEIPASS'):
                candidates.append(os.path.join(sys._MEIPASS, 'src', 'bicycle_models'))
        except:
            pass
        
        # Also try common relative paths
        candidates.extend([
            './src/bicycle_models',
            '../src/bicycle_models',
            '../../src/bicycle_models',
        ])
        
        print("[BICYCLECLF] Trying to locate models. Candidates:")
        for cand in candidates:
            exists = os.path.exists(cand)
            print(f"  {'OK' if exists else 'NO'} {cand}")
            if exists:
                return os.path.abspath(cand)
        
        # If nothing found, return the original (will create fallback)
        print("[BICYCLECLF] WARNING: No models found in any location!")
        return model_save_path
    
    def __init__(self, dataset_path='../bicycle_dataset', model_save_path='bicycle_models'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_path = dataset_path
        
        # Resolve model path - try multiple locations
        self.model_save_path = self._resolve_model_path(model_save_path)
        print(f"[BICYCLECLF] Final model path: {self.model_save_path}")
        
        # Create model directory
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Define class names - ONLY THREE CATEGORIES
        self.class_names = ['standard', 'slightly_non_standard', 'highly_non_standard']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        # Initialize ResNet50
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Remove the last two layers (avgpool and fc) to get better features
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-2])
        
        # Add adaptive pooling to get fixed size features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_extractor = nn.Sequential(
            self.feature_extractor,
            self.adaptive_pool,
            nn.Flatten()
        )
        
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Image transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Classifiers
        self.knn_model = None
        self.svm_model = None
        self.rf_model = None
        
        # Feature database
        self.feature_database = []
        self.database_labels = []
        self.class_prototypes = {}
        
        # Ensemble weights
        self.ensemble_weights = {'knn': 0.3, 'svm': 0.4, 'rf': 0.3}
        
        # Confidence threshold
        self.min_confidence = 0.3  # Lowered to be more inclusive
        
        # Load or create models
        self._load_or_create_models()
    
    def _load_or_create_models(self):
        """Load existing models or create from scratch"""
        print(f"Searching for bicycle models at: {self.model_save_path}")
        print(f"   Directory exists: {os.path.exists(self.model_save_path)}")
        
        model_files = {
            'knn': os.path.join(self.model_save_path, 'knn_model.pkl'),
            'svm': os.path.join(self.model_save_path, 'svm_model.pkl'),
            'rf': os.path.join(self.model_save_path, 'rf_model.pkl'),
            'features': os.path.join(self.model_save_path, 'features_data.pkl')
        }
        
        for key, path in model_files.items():
            exists = os.path.exists(path)
            print(f"   {key}: {'OK' if exists else 'NO'}")
        
        all_exist = all(os.path.exists(f) for f in model_files.values())
        
        if all_exist:
            try:
                # Load features
                with open(model_files['features'], 'rb') as f:
                    data = pickle.load(f)
                    self.feature_database = data['features']
                    self.database_labels = data['labels']
                    self.class_prototypes = data.get('prototypes', {})
                
                # Load models
                self.knn_model = joblib.load(model_files['knn'])
                self.svm_model = joblib.load(model_files['svm'])
                self.rf_model = joblib.load(model_files['rf'])
                
                print(f"Loaded models from {self.model_save_path}")
                print(f"   Database size: {len(self.feature_database)} samples")
                print(f"   Class prototypes: {list(self.class_prototypes.keys())}")
                return
            except Exception as e:
                print(f"Failed to load models: {e}")
                print(f"   Error: {type(e).__name__}")
        
        # Models not found - fallback to dummy models
        print(f"Models NOT FOUND at {self.model_save_path}")
        print("FALLBACK: Creating dummy classifier (bicycles will all be classified as STANDARD!)")
        print("Creating fallback bicycle classifier...")
        self._create_feature_database()
        self._train_classifiers()
    
    def _create_feature_database(self):
        """Extract enhanced features from all bicycle images"""
        print("Extracting features from bicycle dataset...")
        
        all_features = []
        all_labels = []
        
        for class_name in self.class_names:
            class_path = os.path.join(self.dataset_path, class_name)
            class_idx = self.class_to_idx[class_name]
            
            if not os.path.exists(class_path):
                print(f"Warning: Class folder not found: {class_path}")
                print(f"   Creating empty folder: {class_path}")
                os.makedirs(class_path, exist_ok=True)
                continue
            
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.jfif', '.JPG', '.JPEG', '.PNG'))]
            
            if len(image_files) == 0:
                print(f"No images found in: {class_path}")
                print(f"   Please add images to this folder for training")
                continue
            
            print(f"Processing {class_name}: {len(image_files)} images")
            
            class_features = []
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Load and preprocess image
                    image = Image.open(img_path).convert('RGB')
                    
                    # Extract features using test transform
                    image_tensor = self.test_transform(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        features = self.feature_extractor(image_tensor).cpu().numpy().flatten()
                    
                    # Normalize features
                    features = features / (np.linalg.norm(features) + 1e-10)
                    
                    class_features.append(features)
                    all_features.append(features)
                    all_labels.append(class_idx)
                    
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
            
            # Calculate prototype for this class
            if class_features:
                self.class_prototypes[class_name] = np.mean(class_features, axis=0)
                print(f"   Computed prototype for {class_name}")
        
        if all_features:
            self.feature_database = all_features
            self.database_labels = all_labels
            
            # Save features
            data = {
                'features': self.feature_database,
                'labels': self.database_labels,
                'prototypes': self.class_prototypes
            }
            
            with open(os.path.join(self.model_save_path, 'features_data.pkl'), 'wb') as f:
                pickle.dump(data, f)
            
            print(f"Created feature database with {len(self.feature_database)} samples")
            print(f"Computed prototypes for {len(self.class_prototypes)} classes")
        else:
            print("No features extracted! Creating fallback prototypes...")
            # Create dummy prototypes if no data
            feature_dim = 2048  # ResNet50 feature dimension
            for class_name in self.class_names:
                self.class_prototypes[class_name] = np.random.randn(feature_dim)
                self.class_prototypes[class_name] = self.class_prototypes[class_name] / np.linalg.norm(self.class_prototypes[class_name])
    
    def _train_classifiers(self):
        """Train multiple classifiers on the features"""
        if len(self.feature_database) == 0:
            print("No features to train on! Creating fallback models...")
            self._create_fallback_models()
            return
        
        X = np.array(self.feature_database)
        y = np.array(self.database_labels)
        
        if len(np.unique(y)) < 2:
            print("Not enough classes for training! Creating fallback models...")
            self._create_fallback_models()
            return
        
        print("Training classifiers...")
        
        try:
            # 1. KNN classifier
            print("   Training KNN...")
            self.knn_model = NearestNeighbors(n_neighbors=min(5, len(X)), metric='cosine')
            self.knn_model.fit(X)
            joblib.dump(self.knn_model, os.path.join(self.model_save_path, 'knn_model.pkl'))
            
            # 2. SVM classifier
            print("   Training SVM...")
            self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
            self.svm_model.fit(X, y)
            joblib.dump(self.svm_model, os.path.join(self.model_save_path, 'svm_model.pkl'))
            
            # 3. Random Forest classifier
            print("   Training Random Forest...")
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.rf_model.fit(X, y)
            joblib.dump(self.rf_model, os.path.join(self.model_save_path, 'rf_model.pkl'))
            
            print("All classifiers trained and saved!")
        except Exception as e:
            print(f"Error training classifiers: {e}")
            self._create_fallback_models()
    
    def _create_fallback_models(self):
        """Create fallback models when training data is insufficient"""
        print("Creating fallback models...")
        
        # Create dummy features for training fallback models
        feature_dim = 2048
        dummy_features = []
        dummy_labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            # Create 3 samples per class
            for _ in range(3):
                features = np.random.randn(feature_dim)
                features = features / (np.linalg.norm(features) + 1e-10)
                dummy_features.append(features)
                dummy_labels.append(class_idx)
        
        X = np.array(dummy_features)
        y = np.array(dummy_labels)
        
        # Create simple fallback models
        self.knn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
        self.knn_model.fit(X)
        
        self.svm_model = SVC(kernel='linear', probability=True, random_state=42)
        self.svm_model.fit(X, y)
        
        self.rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.rf_model.fit(X, y)
        
        # Save fallback models
        joblib.dump(self.knn_model, os.path.join(self.model_save_path, 'knn_model.pkl'))
        joblib.dump(self.svm_model, os.path.join(self.model_save_path, 'svm_model.pkl'))
        joblib.dump(self.rf_model, os.path.join(self.model_save_path, 'rf_model.pkl'))
        
        print("Created fallback models")
    
    def extract_enhanced_features(self, image):
        """Extract enhanced features from a single image"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Apply test transformations
            image_tensor = self.test_transform(pil_image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(image_tensor).cpu().numpy().flatten()
            
            # Normalize
            features = features / (np.linalg.norm(features) + 1e-10)
            
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return dummy features if extraction fails
            return np.random.randn(2048) / 100.0
    
    def classify_with_knn(self, features):
        """Classify using KNN similarity"""
        if self.knn_model is None:
            # Return random class if no model
            random_class = np.random.choice(self.class_names)
            return random_class, 0.33
        
        try:
            # Find k-nearest neighbors
            distances, indices = self.knn_model.kneighbors([features])
            
            # Get labels of nearest neighbors
            neighbor_labels = [self.database_labels[i] for i in indices[0]] if self.database_labels else [0]
            
            # Calculate weighted votes (closer neighbors get more weight)
            weights = 1.0 / (distances[0] + 1e-10)
            weighted_votes = {}
            
            for label, weight in zip(neighbor_labels, weights):
                weighted_votes[label] = weighted_votes.get(label, 0) + weight
            
            # Get winning class
            if not weighted_votes:
                random_class = np.random.choice(self.class_names)
                return random_class, 0.33
            
            winning_label = max(weighted_votes, key=weighted_votes.get)
            total_weight = sum(weights)
            confidence = weighted_votes[winning_label] / total_weight
            
            return self.idx_to_class[winning_label], float(confidence)
        except:
            random_class = np.random.choice(self.class_names)
            return random_class, 0.33
    
    def classify_with_svm(self, features):
        """Classify using SVM"""
        if self.svm_model is None:
            random_class = np.random.choice(self.class_names)
            return random_class, 0.33
        
        try:
            # Get probabilities for each class
            probabilities = self.svm_model.predict_proba([features])[0]
            winning_idx = np.argmax(probabilities)
            confidence = probabilities[winning_idx]
            
            return self.idx_to_class[winning_idx], float(confidence)
        except:
            random_class = np.random.choice(self.class_names)
            return random_class, 0.33
    
    def classify_with_rf(self, features):
        """Classify using Random Forest"""
        if self.rf_model is None:
            random_class = np.random.choice(self.class_names)
            return random_class, 0.33
        
        try:
            # Get probabilities for each class
            probabilities = self.rf_model.predict_proba([features])[0]
            winning_idx = np.argmax(probabilities)
            confidence = probabilities[winning_idx]
            
            return self.idx_to_class[winning_idx], float(confidence)
        except:
            random_class = np.random.choice(self.class_names)
            return random_class, 0.33
    
    def classify_with_prototype(self, features):
        """Classify by comparing to class prototypes"""
        if not self.class_prototypes:
            random_class = np.random.choice(self.class_names)
            return random_class, 0.33
        
        try:
            best_class = self.class_names[0]  # Default to first class
            best_similarity = -1
            
            for class_name, prototype in self.class_prototypes.items():
                # Calculate cosine similarity
                similarity = np.dot(features, prototype) / (
                    np.linalg.norm(features) * np.linalg.norm(prototype) + 1e-10
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_class = class_name
            
            # Convert similarity to confidence (scale from -1 to 1 into 0 to 1)
            confidence = (best_similarity + 1) / 2
            
            return best_class, float(confidence)
        except:
            random_class = np.random.choice(self.class_names)
            return random_class, 0.33
    
    def ensemble_classify(self, bicycle_image):
        """Classify using ensemble of multiple methods - ALWAYS returns one of three classes"""
        features = self.extract_enhanced_features(bicycle_image)
        
        # Debug: Check model status
        has_knn = self.knn_model is not None
        has_svm = self.svm_model is not None
        has_rf = self.rf_model is not None
        if has_knn or has_svm or has_rf:
            print(f"[CLASSIFY] Using models: KNN={has_knn}, SVM={has_svm}, RF={has_rf}")
        else:
            print("[CLASSIFY] WARNING: NO MODELS LOADED - using FALLBACK (all will be standard)")
        
        # Get predictions from all methods
        predictions = []
        confidences = []
        
        # 1. KNN
        knn_class, knn_conf = self.classify_with_knn(features)
        predictions.append(knn_class)
        confidences.append(knn_conf * self.ensemble_weights['knn'])
        
        # 2. SVM
        svm_class, svm_conf = self.classify_with_svm(features)
        predictions.append(svm_class)
        confidences.append(svm_conf * self.ensemble_weights['svm'])
        
        # 3. Random Forest
        rf_class, rf_conf = self.classify_with_rf(features)
        predictions.append(rf_class)
        confidences.append(rf_conf * self.ensemble_weights['rf'])
        
        # 4. Prototype matching
        proto_class, proto_conf = self.classify_with_prototype(features)
        predictions.append(proto_class)
        confidences.append(proto_conf * 0.1)  # Lower weight
        
        # Aggregate scores per class
        class_scores = {name: 0.0 for name in self.class_names}
        for pred, conf in zip(predictions, confidences):
            if pred in class_scores:
                class_scores[pred] += float(conf)
        
        # If all scores are zero (shouldn't happen), fallback to prototype
        if all(score == 0.0 for score in class_scores.values()):
            if proto_class in class_scores:
                return proto_class, float(proto_conf)
            else:
                return 'standard', 0.33
        
        # Pick the class with maximum aggregated score
        final_class = max(class_scores, key=class_scores.get)
        total_score = sum(class_scores.values())
        final_confidence = (class_scores[final_class] / total_score) if total_score > 0 else 0.33

        # If slightly_non_standard is close to top score, favor it to reduce under-reporting
        top_score = class_scores.get(final_class, 0.0)
        slight_score = class_scores.get('slightly_non_standard', 0.0)
        if slight_score > 0 and slight_score >= (0.85 * top_score):
            final_class = 'slightly_non_standard'
            final_confidence = (slight_score / total_score) if total_score > 0 else final_confidence
        
        # Ensure confidence is reasonable
        final_confidence = max(self.min_confidence, min(1.0, final_confidence))
        
        return final_class, float(final_confidence)
    
    def classify_bicycle(self, bicycle_image):
        """Main classification method - ALWAYS returns one of the three classes"""
        # Always use ensemble classification
        return self.ensemble_classify(bicycle_image)


# For backward compatibility
class BicycleClassifier:
    """Wrapper for backward compatibility"""
    def __init__(self, dataset_path='../bicycle_dataset', model_save_path='bicycle_models'):
        self.classifier = EnhancedBicycleClassifier(dataset_path, model_save_path)
    
    def classify_bicycle(self, bicycle_image):
        return self.classifier.classify_bicycle(bicycle_image)
    
    def extract_features(self, image):
        return self.classifier.extract_enhanced_features(image)
    
    def analyze_dataset(self):
        return self.classifier.analyze_dataset()