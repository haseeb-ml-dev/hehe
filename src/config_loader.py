"""
Configuration loader for Traffic Detector.
Loads and validates settings from config.yaml.
"""

import os
import sys
import yaml
from typing import Dict, Any, Optional


def get_resource_path(relative_path: str) -> str:
    """Resolve resource path for bundled or dev environments."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class ConfigLoader:
    """Loads and validates configuration from YAML file."""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Initialize config loader.
        
        Args:
            config_file: Path to configuration YAML file
        """
        self.config_file = config_file
        self.config = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_file}\n"
                f"Please ensure config.yaml exists in the project root directory."
            )
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                if self.config is None:
                    self.config = {}
        except yaml.YAMLError as e:
            raise ValueError(
                f"Error parsing configuration file: {self.config_file}\n"
                f"Please check the YAML syntax.\n"
                f"Error details: {e}"
            )
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration as dictionary."""
        video_cfg = self.config.get('video', {})
        model_cfg = self.config.get('model', {})
        proc_cfg = self.config.get('processing', {})
        detect_cfg = self.config.get('detection', {})
        
        return {
            'yolo_model': model_cfg.get('yolo_size', 'm'),
            'is_360': video_cfg.get('is_360_degree', True),
            'frame_skip': self._validate_int(proc_cfg.get('frame_skip', 3), 1, 10),
            'process_percentage': self._validate_int(
                proc_cfg.get('process_percentage', 100), 1, 100
            ),
            'debug_mode': detect_cfg.get('debug_mode', False),
            'confidence_threshold': self._validate_float(
                model_cfg.get('confidence_threshold', 0.5), 0.1, 0.9
            ),
            'min_hits': self._validate_int(
                detect_cfg.get('min_hits_before_counting', 3), 1, 10
            ),
            'min_object_size': self._validate_int(
                detect_cfg.get('min_object_size', 40), 20, 200
            ),
            'max_distance_ratio': self._get_max_distance_ratio(
                detect_cfg.get('distance_filter_sensitivity', 1.5)
            ),
        }
    
    def get_video_input_path(self) -> str:
        """Get video input file path."""
        video_cfg = self.config.get('video', {})
        video_file = video_cfg.get('input_file', '').strip()
        
        if video_file and os.path.exists(video_file):
            return video_file
        
        input_dir = video_cfg.get('input_directory', './data/input_videos')
        return input_dir
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        output_cfg = self.config.get('output', {})
        
        return {
            'directory': output_cfg.get('directory', './data/output'),
            'save_csv': output_cfg.get('save_csv', True),
            'save_detailed_counts': output_cfg.get('save_detailed_counts', True),
            'video_filename': output_cfg.get('video_filename', ''),
        }
    
    def get_gis_metadata_file(self) -> Optional[str]:
        """Get GIS metadata file path if configured."""
        gis_cfg = self.config.get('gis', {})
        metadata_file = gis_cfg.get('metadata_file', '').strip()
        
        if metadata_file and os.path.exists(metadata_file):
            return metadata_file
        return None
    
    def get_detection_targets(self) -> Dict[str, bool]:
        """Get detection target configuration."""
        targets_cfg = self.config.get('targets', {})
        
        return {
            'detect_vehicles': targets_cfg.get('detect_vehicles', True),
            'detect_bicycles': targets_cfg.get('detect_bicycles', True),
            'classify_bicycles': targets_cfg.get('classify_bicycles', True),
            'detect_signs': targets_cfg.get('detect_signs', True),
            'track_parking': targets_cfg.get('track_parking', True),
        }
    
    @staticmethod
    def _validate_int(value: Any, min_val: int, max_val: int) -> int:
        """Validate and clamp integer value."""
        try:
            val = int(value)
            return max(min_val, min(max_val, val))
        except (ValueError, TypeError):
            return min_val
    
    @staticmethod
    def _validate_float(value: Any, min_val: float, max_val: float) -> float:
        """Validate and clamp float value."""
        try:
            val = float(value)
            return max(min_val, min(max_val, val))
        except (ValueError, TypeError):
            return min_val
    
    @staticmethod
    def _get_max_distance_ratio(sensitivity: float) -> float:
        """
        Convert distance filter sensitivity to max distance ratio.
        
        Sensitivity mapping:
        1.0 = 0.0025 (very strict, closest only)
        1.5 = 0.0015 (balanced - default)
        2.0 = 0.0008 (loose, more distant objects)
        """
        try:
            sens = float(sensitivity)
            if sens <= 1.0:
                return 0.0025
            elif sens >= 2.0:
                return 0.0008
            else:
                # Linear interpolation between 1.0 and 2.0
                return 0.0025 - ((sens - 1.0) * 0.0017)
        except (ValueError, TypeError):
            return 0.0015  # Default


def load_config(base_dir: str = None) -> ConfigLoader:
    """
    Load configuration from config.yaml.
    
    Args:
        base_dir: Base directory to look for config.yaml (defaults to current working directory)
    
    Returns:
        ConfigLoader instance with loaded configuration
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    config_file = os.path.join(base_dir, 'config.yaml')
    
    # If not found, try PyInstaller bundle location
    if not os.path.exists(config_file):
        config_file = get_resource_path('config.yaml')
    
    return ConfigLoader(config_file)
