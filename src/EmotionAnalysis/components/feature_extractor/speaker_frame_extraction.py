from dataclasses import dataclass
import logging
from pathlib import Path
import cv2
import dlib
import face_recognition
import numpy as np
import json
import os
import shutil
import tempfile
import time
import signal
import sys
import threading
import tarfile
import fcntl
from tqdm import tqdm
from scipy.signal import find_peaks
from collections import deque

from EmotionAnalysis.config import ConfigurationManager
from EmotionAnalysis.utils.common_utils import read_from_json, save_to_json
from EmotionAnalysis.entity import VideoFeatureExtractionConfig

# Global emergency stop flag
EMERGENCY_STOP = threading.Event()



class AtomicFrameSaver:
    @staticmethod
    def save_frame(frame, path):
        """Save frame in RGB format"""
        base, ext = os.path.splitext(path)
        temp_path = f"{base}.tmp{ext}"
        # Convert to RGB before saving
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(temp_path, rgb_frame)
        os.rename(temp_path, path)

class MouthAnalyzer:
    def __init__(self):
        self.mar_history = deque(maxlen=30)
    
    def set_history_size(self, size):
        self.mar_history = deque(maxlen=size)
        
    def calculate_rhythm_score(self):
        if len(self.mar_history) < 10:
            return 0
        mar_values = np.array(self.mar_history)
        eps = 1e-8
        mar_norm = (mar_values - np.mean(mar_values)) / (np.std(mar_values) + eps)
        peaks, _ = find_peaks(mar_norm, prominence=0.5)
        valley, _ = find_peaks(-mar_norm, prominence=0.5)
        if len(peaks) > 2 and len(valley) > 2:
            peak_intervals = np.diff(peaks)
            rhythm_consistency = 1 - np.std(peak_intervals) / np.mean(peak_intervals)
            return min(max(rhythm_consistency, 0), 1)
        return 0

class AtomicCheckpointManager:
    @staticmethod
    def load(checkpoint_file):
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    return json.load(f)
            emergency_checkpoint = os.path.join(os.path.dirname(checkpoint_file), "emergency_backups", "checkpoint.json")
            if os.path.exists(emergency_checkpoint):
                shutil.copy(emergency_checkpoint, checkpoint_file)
                return AtomicCheckpointManager.load(checkpoint_file)
        except Exception as e:
            print(f"Checkpoint load failed: {str(e)}")
        return {"processed": {"train": [], "dev": [], "test": []}, "metadata": {"train": [], "dev": [], "test": []}}

    @staticmethod
    def save(checkpoint_file, data):
        temp_path = f"{checkpoint_file}.tmp"
        with open(temp_path, 'w') as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, checkpoint_file)
        # Save emergency backup
        emergency_dir = os.path.join(os.path.dirname(checkpoint_file), "emergency_backups")
        os.makedirs(emergency_dir, exist_ok=True)
        shutil.copy(checkpoint_file, os.path.join(emergency_dir, "checkpoint.json"))

class SpeakerTracker:
    def __init__(self, frame_size, config):
        self.known_faces = {}
        self.current_focus = None
        self.focus_confidence = 0
        self.consecutive_frames = 0
        self.face_analyzers = {}
        self.frame_size = frame_size
        self.config = config
        
    def update_focus(self, fid, score, position):
        x_center = position[0] + position[2]/2
        y_center = position[1] + position[3]/2
        centrality = 1 - (abs(x_center - self.frame_size[1]/2)/(self.frame_size[1]/2) +
                          abs(y_center - self.frame_size[0]/2)/(self.frame_size[0]/2))/2
        combined_score = (score * (1 - self.config.centrality_weight) + 
                        centrality * self.config.centrality_weight)
        
        if self.current_focus == fid:
            confidence_gain = combined_score * 0.1
            self.focus_confidence = min(self.focus_confidence + confidence_gain, 1)
            self.consecutive_frames += 1
        else:
            confidence_gain = combined_score * 0.05
            self.focus_confidence = max(self.focus_confidence - 0.02, 0)
            
        if (combined_score > self.focus_confidence + 0.2 or 
            (self.consecutive_frames < self.config.min_speaking_frames and 
             combined_score > self.focus_confidence)):
            self.current_focus = fid
            self.focus_confidence = combined_score
            self.consecutive_frames = 1

def get_mouth_aspect_ratio(shape):
    try:
        points = [(shape.part(i).x, shape.part(i).y) for i in range(48, 68)]
        vert_dist = (np.linalg.norm(np.array(points[2]) - np.array(points[10])) + 
                   np.linalg.norm(np.array(points[4]) - np.array(points[8]))) / 2
        horiz_dist = np.linalg.norm(np.array(points[0]) - np.array(points[6]))
        return vert_dist / horiz_dist if horiz_dist != 0 else 0
    except:
        return 0

class GracefulTerminator:
    def __init__(self, max_runtime):
        self.start_time = time.time()
        self.time_limit = self.start_time + max_runtime
        print(f"Auto-save scheduled at: {time.ctime(self.time_limit)}")

    def should_stop(self):
        return EMERGENCY_STOP.is_set() or (time.time() > self.time_limit)

    def remaining(self):
        return max(0, self.time_limit - time.time())

def get_mask_info(output_dir, num_frames):
    """Generate mask_info list based on existing frame files"""
    mask_info = []
    for idx in range(num_frames):
        frame_num = f"{idx:02d}"
        umf_path = os.path.join(output_dir, f"{frame_num}_UMF.jpg")
        mask_info.append(1 if os.path.exists(umf_path) else 0)
    return mask_info

class VideoProcessor:
    def __init__(self, config: VideoFeatureExtractionConfig):
        self.config = config
        # Initialize models
        self.face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
        self.face_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        # Create directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.emergency_dir).mkdir(parents=True, exist_ok=True)
        
        # GPU check
        print(f"DLIB CUDA Status: {dlib.DLIB_USE_CUDA}")
        print(f"Available CUDA Devices: {dlib.cuda.get_num_devices()}")

    def handle_session_end(self, signum, frame):
        lock_file = os.path.join(self.config.emergency_dir, ".save_lock")
        lock_fd = None
        
        try:
            lock_fd = os.open(lock_file, os.O_CREAT | os.O_WRONLY)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Save already in progress - skipping duplicate")
                return

            print(f"\nSession ending! Saving state (Signal {signum})")
            
            if not os.path.exists(self.config.checkpoint_file):
                print("No checkpoint to save")
                return

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_checkpoint = os.path.join(tmpdir, "checkpoint.json")
                shutil.copy(self.config.checkpoint_file, tmp_checkpoint)
                
                tmp_tar = os.path.join(tmpdir, "emergency.tar.gz")
                with tarfile.open(tmp_tar, "w:gz") as tar:
                    tar.add(self.config.output_dir, arcname="preprocessed_frames")
                    tar.add(tmp_checkpoint, arcname="checkpoint.json")
                
                shutil.move(tmp_tar, os.path.join(self.config.emergency_dir, "emergency_snapshot.tar.gz"))
                print("Emergency save completed")

        except Exception as e:
            print(f"Emergency save failed: {str(e)}")
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                except OSError:
                    pass
            try:
                os.remove(lock_file)
            except FileNotFoundError:
                pass
            EMERGENCY_STOP.set()
            sys.exit(1)

    def identify_main_speaker(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
            
        frame_size = (int(cap.get(4)), int(cap.get(3)))
        tracker = SpeakerTracker(frame_size, self.config)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dlib_faces = self.face_detector(rgb, 1)
            faces = []
            for face in dlib_faces:
                if face.confidence >= self.config.face_confidence:
                    rect = face.rect
                    faces.append((rect.top(), rect.right(), rect.bottom(), rect.left()))
            
            encodings = face_recognition.face_encodings(rgb, faces)
            current_faces = []
            
            for (top, right, bottom, left), encoding in zip(faces, encodings):
                face_id = None
                for fid, data in tracker.known_faces.items():
                    similarity = 1 - face_recognition.face_distance([data['encoding']], encoding)[0]
                    if similarity > self.config.similarity_threshold:
                        face_id = fid
                        break
                        
                if face_id is None:
                    face_id = len(tracker.known_faces)
                    tracker.known_faces[face_id] = {
                        'encoding': encoding,
                        'analyzer': MouthAnalyzer(),
                        'position': (left, top, right-left, bottom-top)
                    }
                    tracker.known_faces[face_id]['analyzer'].set_history_size(self.config.speaker_history)
                    
                shape = self.face_predictor(rgb, dlib.rectangle(left, top, right, bottom))
                mar = get_mouth_aspect_ratio(shape)
                tracker.known_faces[face_id]['analyzer'].mar_history.append(mar)
                current_faces.append(face_id)
                
            scores = {}
            for fid in current_faces:
                data = tracker.known_faces[fid]
                rhythm_score = data['analyzer'].calculate_rhythm_score()
                recent_mar = np.mean(list(data['analyzer'].mar_history)[-10:]) if data['analyzer'].mar_history else 0
                activity_score = recent_mar * (1 + rhythm_score)
                scores[fid] = activity_score
                
            if scores:
                max_fid = max(scores, key=lambda x: scores[x])
                tracker.update_focus(max_fid, scores[max_fid], 
                                    tracker.known_faces[max_fid]['position'])
                
        cap.release()
        
        if tracker.consecutive_frames >= self.config.min_speaking_frames and tracker.current_focus is not None:
            return tracker.known_faces[tracker.current_focus]['encoding']
        return None

    def process_video(self, video_path, output_dir):
        temp_dir = f"{output_dir}_processing"
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        mask_info = []
        cap = None
        
        try:
            speaker_encoding = self.identify_main_speaker(video_path)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video {video_path}")
                
            frames = []
            confirmed_focus = None
            tracker = None
            tracking_failures = 0
            last_valid_face = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current_face = None
                
                if tracker is not None:
                    tracking_quality = tracker.update(rgb)
                    if tracking_quality >= self.config.tracking_quality_threshold:
                        tracked_pos = tracker.get_position()
                        t = int(tracked_pos.top())
                        r = int(tracked_pos.right())
                        b = int(tracked_pos.bottom())
                        l = int(tracked_pos.left())
                        current_face = (t, r, b, l)
                        tracking_failures = 0
                        last_valid_face = current_face
                    else:
                        tracking_failures += 1
                        if tracking_failures > self.config.max_tracking_failures:
                            tracker = None
                            tracking_failures = 0
                
                if current_face is None:
                    dlib_faces = self.face_detector(rgb, 1)
                    valid_faces = []
                    for face in dlib_faces:
                        if face.confidence >= self.config.face_confidence:
                            rect = face.rect
                            valid_faces.append((rect.top(), rect.right(), rect.bottom(), rect.left()))
                    
                    best_face = None
                    best_similarity = -1
                    
                    if speaker_encoding is not None:
                        for (t, r, b, l) in valid_faces:
                            encoding = face_recognition.face_encodings(rgb, [(t, r, b, l)])[0]
                            similarity = 1 - face_recognition.face_distance([speaker_encoding], encoding)[0]
                            if similarity > best_similarity and similarity >= self.config.similarity_threshold:
                                best_similarity = similarity
                                best_face = (t, r, b, l)
                    
                    if best_face:
                        t, r, b, l = best_face
                        current_face = (t, r, b, l)
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(rgb, dlib.rectangle(l, t, r, b))
                        last_valid_face = current_face
                    elif last_valid_face:
                        current_face = last_valid_face
                    elif valid_faces:
                        frame_center = (rgb.shape[1]//2, rgb.shape[0]//2)
                        closest_dist = float('inf')
                        for face in valid_faces:
                            t, r, b, l = face
                            face_center = ((l + r)//2, (t + b)//2)
                            dist = (face_center[0]-frame_center[0])**2 + (face_center[1]-frame_center[1])**2
                            if dist < closest_dist:
                                closest_dist = dist
                                current_face = face
                        last_valid_face = current_face
                    else:
                        current_face = None
                
                if current_face:
                    t, r, b, l = current_face
                elif confirmed_focus:
                    x, y, w, h = confirmed_focus
                    l, t, r, b = x, y, x + w, y + h
                else:
                    t, r, b, l = 0, frame.shape[1], frame.shape[0], 0
                
                orig_height = b - t
                orig_width = r - l
                new_t = int(max(0, t - orig_height * self.config.bbox_expand_ratio))
                new_b = int(min(frame.shape[0], b + orig_height * self.config.bbox_expand_ratio))
                new_l = int(max(0, l - orig_width * self.config.bbox_expand_ratio))
                new_r = int(min(frame.shape[1], r + orig_width * self.config.bbox_expand_ratio))
                
                cropped = frame[new_t:new_b, new_l:new_r]
                resized = cv2.resize(cropped, self.config.crop_size)
                frames.append(resized)
                confirmed_focus = (new_l, new_t, new_r - new_l, new_b - new_t)
            
            cap.release()
        
            valid_frames = [f for f in frames if np.mean(f) > 10]
            for idx in range(self.config.num_frames):
                if idx < len(valid_frames):
                    frame = valid_frames[idx]
                    frame_type = "UMF"
                else:
                    frame = np.zeros((*self.config.crop_size, 3), dtype=np.uint8)
                    frame_type = "MF"
                
                output_path = os.path.join(output_dir, f"{idx:02d}_{frame_type}.jpg")
                AtomicFrameSaver.save_frame(frame, output_path)
                mask_info.append(1 if frame_type == "UMF" else 0)
            
            return mask_info
            
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            raise
        finally:
            if cap is not None:
                cap.release()
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def run(self):
        # Register signal handlers
        signal.signal(signal.SIGTERM, self.handle_session_end)
        signal.signal(signal.SIGHUP, self.handle_session_end)
        signal.signal(signal.SIGINT, self.handle_session_end)
        
        # Load input data
        video_data = read_from_json(self.config.input_json)
        
        # Initialize checkpoint
        checkpoint = AtomicCheckpointManager.load(self.config.checkpoint_file)
        
        # Initialize terminator
        terminator = GracefulTerminator(self.config.max_runtime)
        
        # Create output directories for splits
        for split in ['train', 'dev', 'test']:
            split_dir = os.path.join(self.config.output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
        
        # Count total videos
        total_videos = sum(len(video_data[split]) for split in ['train', 'dev', 'test'])
        processed_count = sum(len(checkpoint["processed"][split]) for split in ['train', 'dev', 'test'])
        main_pbar = tqdm(total=total_videos, initial=processed_count, desc="Total Progress")
        
        for split in ['train', 'dev', 'test']:
            split_dir = os.path.join(self.config.output_dir, split)
            processed_videos = set(checkpoint["processed"][split])
            total_in_split = len(video_data[split])
            
            # Filter unprocessed items
            unprocessed_items = [
                item for item in video_data[split]
                if [k for k in item.keys() if k not in ('y', 'label')][0] not in processed_videos
            ]
            
            split_pbar = tqdm(
                unprocessed_items,
                desc=f"Processing {split}",
                total=total_in_split,
                initial=len(processed_videos))
            
            for item in video_data[split]:
                if terminator.should_stop():
                    print("Emergency stop triggered - terminating")
                    AtomicCheckpointManager.save(self.config.checkpoint_file, checkpoint)
                    return
                
                video_key = [k for k in item.keys() if k not in ('y', 'label')][0]
                video_path = item[video_key]
                output_dir = os.path.join(split_dir, video_key)
                
                # Skip already processed
                if video_key in processed_videos:
                    split_pbar.update(1)
                    main_pbar.update(1)
                    continue
                
                try:
                    # Check for partial processing
                    if os.path.exists(output_dir):
                        existing_frames = len([
                            f for f in os.listdir(output_dir) 
                            if f.endswith(".jpg") and not f.startswith("_temp")
                        ])
                        if existing_frames >= self.config.frame_validation_threshold:
                            mask_info = get_mask_info(output_dir, self.config.num_frames)
                            
                            checkpoint["metadata"][split].append({
                                **item,
                                "frames_dir": output_dir,
                                "mask_info": mask_info
                            })
                            checkpoint["processed"][split].append(video_key)
                            AtomicCheckpointManager.save(self.config.checkpoint_file, checkpoint)
                            split_pbar.update(1)
                            main_pbar.update(1)
                            continue
                        shutil.rmtree(output_dir)
                    
                    # Process video
                    mask_info = self.process_video(video_path, output_dir)
                    
                    # Update checkpoint
                    checkpoint["metadata"][split].append({
                        **item,
                        "frames_dir": output_dir,
                        "mask_info": mask_info
                    })
                    checkpoint["processed"][split].append(video_key)
                    
                    # Periodically save checkpoint
                    if len(checkpoint["processed"][split]) % 5 == 0:
                        AtomicCheckpointManager.save(self.config.checkpoint_file, checkpoint)
                    
                    split_pbar.update(1)
                    main_pbar.update(1)
                    
                except Exception as e:
                    print(f"Error processing {video_key}: {str(e)}")
                    if os.path.exists(output_dir):
                        shutil.rmtree(output_dir)
            
            split_pbar.close()
            AtomicCheckpointManager.save(self.config.checkpoint_file, checkpoint)
        
        main_pbar.close()
        print("Video processing completed successfully!")
        return checkpoint

class VideoFeatureExtractor:
    def __init__(self, config: VideoFeatureExtractionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def run(self):
        self.logger.info("Starting video frame extraction")
        processor = VideoProcessor(self.config)
        result = processor.run()
        self.logger.info("Video frame extraction completed successfully!")
        return result