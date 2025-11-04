# PREMIUM Emotion Tracking Server - SINGLE FACE & ENHANCED EMOTIONS
import cv2
import asyncio
import websockets
import json
import base64
import numpy as np
import time
from datetime import datetime
import logging
from collections import deque, defaultdict

print("🚀 Starting PREMIUM Emotion Tracking Server - SINGLE FACE MODE...")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Load DeepFace
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded successfully!")
except Exception as e:
    DEEPFACE_AVAILABLE = False
    print(f"❌ DeepFace error: {e}")
    exit(1)

class EnhancedFaceValidator:
    def __init__(self):
        # Load multiple face detectors for validation
        self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Validation thresholds
        self.MIN_FACE_CONFIDENCE = 0.7
        self.MIN_EMBEDDING_QUALITY = 0.6
        self.ASPECT_RATIO_RANGE = (0.7, 1.4)  # Wider but reasonable for faces
        self.SIZE_RANGE = (60, 400)  # Pixel dimensions
        self.POSITION_MARGIN = 0.05  # 5% margin from edges
        
        print("✅ Enhanced Face Validator initialized with multiple detectors")
    
    def calculate_iou(self, boxA, boxB):
        """Calculate Intersection over Union for bounding box validation"""
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        if len(boxA) == 4 and len(boxB) == 4:
            boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
            boxB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]
        
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
        return iou
    
    def validate_with_haar_cascade(self, face_roi, original_bbox, frame_shape):
        """Validate face using Haar Cascade for additional confirmation"""
        try:
            # Convert to grayscale for Haar cascade
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the ROI
            haar_faces = self.haar_cascade.detectMultiScale(
                gray_roi, 
                scaleFactor=1.1, 
                minNeighbors=3, 
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(haar_faces) > 0:
                # Check if any Haar detection overlaps significantly with original bbox
                for (hx, hy, hw, hh) in haar_faces:
                    # Convert to original frame coordinates
                    original_x, original_y, original_w, original_h = original_bbox
                    haar_bbox = [original_x + hx, original_y + hy, hw, hh]
                    
                    iou = self.calculate_iou(original_bbox, haar_bbox)
                    if iou > 0.3:  # Significant overlap
                        return True, iou
            
            return False, 0.0
            
        except Exception as e:
            return False, 0.0
    
    def validate_face_characteristics(self, face_roi, bbox, frame_shape):
        """Comprehensive face characteristic validation"""
        validation_results = {
            'passed': True,
            'reasons': [],
            'confidence': 1.0
        }
        
        h, w = face_roi.shape[:2]
        frame_height, frame_width = frame_shape[:2]
        x, y, bbox_w, bbox_h = bbox
        
        # 1. Size validation
        if bbox_w < self.SIZE_RANGE[0] or bbox_h < self.SIZE_RANGE[0]:
            validation_results['passed'] = False
            validation_results['reasons'].append(f"Too small: {bbox_w}x{bbox_h}")
            validation_results['confidence'] *= 0.3
        
        if bbox_w > self.SIZE_RANGE[1] or bbox_h > self.SIZE_RANGE[1]:
            validation_results['passed'] = False
            validation_results['reasons'].append(f"Too large: {bbox_w}x{bbox_h}")
            validation_results['confidence'] *= 0.3
        
        # 2. Aspect ratio validation
        aspect_ratio = bbox_w / bbox_h
        if aspect_ratio < self.ASPECT_RATIO_RANGE[0] or aspect_ratio > self.ASPECT_RATIO_RANGE[1]:
            validation_results['passed'] = False
            validation_results['reasons'].append(f"Bad aspect ratio: {aspect_ratio:.2f}")
            validation_results['confidence'] *= 0.5
        
        # 3. Position validation
        margin_x = frame_width * self.POSITION_MARGIN
        margin_y = frame_height * self.POSITION_MARGIN
        
        if (x < margin_x or y < margin_y or 
            x + bbox_w > frame_width - margin_x or 
            y + bbox_h > frame_height - margin_y):
            validation_results['reasons'].append("Near frame edges")
            validation_results['confidence'] *= 0.8  # Less severe penalty
        
        # 4. Area ratio validation
        bbox_area = bbox_w * bbox_h
        frame_area = frame_width * frame_height
        area_ratio = bbox_area / frame_area
        
        if area_ratio > 0.5:  # Reject if covers more than 50% of frame
            validation_results['passed'] = False
            validation_results['reasons'].append(f"Too large area: {area_ratio:.2f}")
            validation_results['confidence'] *= 0.2
        
        if area_ratio < 0.01:  # Reject if too small
            validation_results['passed'] = False
            validation_results['reasons'].append(f"Too small area: {area_ratio:.2f}")
            validation_results['confidence'] *= 0.2
        
        # 5. Image quality checks
        if h == 0 or w == 0:
            validation_results['passed'] = False
            validation_results['reasons'].append("Invalid dimensions")
            validation_results['confidence'] = 0.0
        
        # Check for reasonable color distribution (basic image validity)
        if np.mean(face_roi) < 10 or np.mean(face_roi) > 240:
            validation_results['reasons'].append("Suspicious brightness")
            validation_results['confidence'] *= 0.7
        
        return validation_results
    
    def multi_detector_validation(self, face_roi, original_bbox, frame_shape):
        """Use multiple detection methods to validate face"""
        validation_score = 0.0
        total_checks = 0
        
        # 1. DeepFace verification (already passed to get here)
        validation_score += 1.0
        total_checks += 1
        
        # 2. Haar cascade verification
        haar_valid, haar_iou = self.validate_with_haar_cascade(face_roi, original_bbox, frame_shape)
        if haar_valid:
            validation_score += haar_iou
            total_checks += 1
        
        # 3. Face characteristics validation
        char_results = self.validate_face_characteristics(face_roi, original_bbox, frame_shape)
        validation_score += char_results['confidence']
        total_checks += 1
        
        # Calculate final confidence
        if total_checks > 0:
            final_confidence = validation_score / total_checks
        else:
            final_confidence = 0.0
        
        return final_confidence >= self.MIN_FACE_CONFIDENCE, final_confidence, char_results['reasons']

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.void)): 
            return None
        return super(NumpyEncoder, self).default(obj)

# Enhanced Premium Emotion Tracker - SINGLE FACE MODE
class SingleFaceEmotionTracker:
    def __init__(self):
        self.current_track = None
        self.max_age = 10.0
        
        # Enhanced confidence thresholds
        self.MIN_EMOTION_CONFIDENCE = 0.65  # Slightly lower for better detection
        self.HIGH_CONFIDENCE_THRESHOLD = 0.85
        self.SUPER_HIGH_CONFIDENCE = 0.92
        
        # Face embedding for single face
        self.face_embedding = None
        
        # Enhanced emotion tracking
        self.emotion_history = deque(maxlen=100)  # Longer history
        self.current_emotion = None
        self.emotion_start_time = None
        self.last_emotion_change = None
        
        # Emotion intensity tracking
        self.emotion_intensity_history = deque(maxlen=50)
        self.current_intensity = 0.0
        
        # Processing
        self.last_processed_time = 0
        self.processing_interval = 0.3  # Faster processing
        
        # Enhanced emotion categories
        self.EMOTION_CATEGORIES = {
            'positive': ['happy', 'surprise'],
            'negative': ['angry', 'sad', 'fear'],
            'neutral': ['neutral'],
            'social': ['happy', 'surprise']  # Social emotions
        }
        
        # Emotion transitions tracking
        self.emotion_transitions = []
        self.last_positive_emotion_time = None
        
        # Initialize face validator
        self.face_validator = EnhancedFaceValidator()
        
        print("🎯 ENHANCED Single Face Emotion Tracker initialized")

    def analyze_enhanced_emotion(self, face_roi):
        """Enhanced emotion analysis with better detection"""
        try:
            # Analyze both emotion and age/gender for context
            analysis = DeepFace.analyze(
                face_roi,
                actions=['emotion', 'age', 'gender'],
                detector_backend='mtcnn',
                enforce_detection=True,
                silent=True
            )
            
            if analysis and len(analysis) > 0:
                result = analysis[0]
                emotion_data = result.get('emotion', {})
                
                if emotion_data:
                    # Get all emotions with confidence
                    emotions_with_confidence = []
                    for emotion, confidence in emotion_data.items():
                        confidence_pct = confidence / 100.0
                        emotions_with_confidence.append((emotion, confidence_pct))
                    
                    # Sort by confidence
                    emotions_with_confidence.sort(key=lambda x: x[1], reverse=True)
                    
                    dominant_emotion, dominant_confidence = emotions_with_confidence[0]
                    
                    # Get secondary emotion if significant
                    secondary_emotion = None
                    secondary_confidence = 0.0
                    if len(emotions_with_confidence) > 1 and emotions_with_confidence[1][1] > 0.25:
                        secondary_emotion, secondary_confidence = emotions_with_confidence[1]
                    
                    # Enhanced emotion interpretation
                    interpreted_emotion = self.interpret_emotion_context(
                        dominant_emotion, dominant_confidence, 
                        secondary_emotion, secondary_confidence,
                        result
                    )
                    
                    if dominant_confidence >= self.MIN_EMOTION_CONFIDENCE:
                        return interpreted_emotion, dominant_confidence, {
                            'secondary_emotion': secondary_emotion,
                            'secondary_confidence': secondary_confidence,
                            'all_emotions': emotions_with_confidence[:3],  # Top 3 emotions
                            'age': result.get('age', 0),
                            'gender': result.get('dominant_gender', 'unknown'),
                            'raw_emotion': dominant_emotion
                        }
                        
        except Exception as e:
            print(f"⚠️ Enhanced emotion analysis error: {e}")
            
        return None, 0.0, {}

    def interpret_emotion_context(self, dominant_emotion, dom_confidence, 
                                secondary_emotion, sec_confidence, analysis_result):
        """Interpret emotion with contextual understanding"""
        
        # Map basic emotions to more descriptive ones
        emotion_mapping = {
            'happy': self.interpret_happiness(dom_confidence, sec_confidence),
            'sad': self.interpret_sadness(dom_confidence, sec_confidence),
            'angry': self.interpret_anger(dom_confidence, sec_confidence),
            'surprise': self.interpret_surprise(dom_confidence, sec_confidence),
            'fear': self.interpret_fear(dom_confidence, sec_confidence),
            'neutral': self.interpret_neutral(dom_confidence, sec_confidence, analysis_result)
        }
        
        return emotion_mapping.get(dominant_emotion, dominant_emotion)

    def interpret_happiness(self, confidence, sec_confidence):
        """Interpret different levels and types of happiness"""
        if confidence >= self.SUPER_HIGH_CONFIDENCE:
            return "very_happy"
        elif confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return "happy"
        else:
            return "slightly_happy"

    def interpret_sadness(self, confidence, sec_confidence):
        """Interpret different levels of sadness"""
        if confidence >= self.SUPER_HIGH_CONFIDENCE:
            return "very_sad"
        elif confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return "sad"
        else:
            return "slightly_sad"

    def interpret_anger(self, confidence, sec_confidence):
        """Interpret different levels of anger"""
        if confidence >= self.SUPER_HIGH_CONFIDENCE:
            return "very_angry"
        elif confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return "angry"
        else:
            return "slightly_annoyed"

    def interpret_surprise(self, confidence, sec_confidence):
        """Interpret surprise with context"""
        if confidence >= self.SUPER_HIGH_CONFIDENCE:
            return "very_surprised"
        elif confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return "surprised"
        else:
            return "slightly_surprised"

    def interpret_fear(self, confidence, sec_confidence):
        """Interpret fear with context"""
        if confidence >= self.SUPER_HIGH_CONFIDENCE:
            return "very_scared"
        elif confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return "scared"
        else:
            return "slightly_anxious"

    def interpret_neutral(self, confidence, sec_confidence, analysis_result):
        """Interpret neutral with context from other emotions"""
        if sec_confidence > 0.3:
            # If there's a significant secondary emotion
            return f"mostly_neutral_with_hint_of_{sec_confidence}"
        
        # Check if neutral is actually "thinking" or "concentrating"
        age = analysis_result.get('age', 0)
        if age > 0:
            # Older people might be more likely to be thinking/concentrating
            if confidence > 0.8 and age > 30:
                return "thinking"
        
        return "neutral"

    def get_face_embedding(self, face_roi):
        """Get face embedding for identity verification"""
        try:
            if (face_roi.shape[0] < 80 or face_roi.shape[1] < 80):
                return None
                
            embedding_obj = DeepFace.represent(
                face_roi,
                model_name='Facenet',
                enforce_detection=True,
                detector_backend='mtcnn',
                align=True
            )
            
            if embedding_obj and len(embedding_obj) > 0:
                return np.array(embedding_obj[0]['embedding'])
                
        except Exception as e:
            print(f"⚠️ Embedding error: {e}")
        
        return None

    def calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        try:
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)
            similarity = np.dot(emb1_norm, emb2_norm)
            return float(max(0.0, min(1.0, similarity)))
        except Exception as e:
            return 0.0

    def is_same_face(self, new_embedding):
        """Check if new face matches current track"""
        if self.face_embedding is None:
            return True  # First face detection
        
        similarity = self.calculate_similarity(self.face_embedding, new_embedding)
        return similarity >= 0.85  # High threshold for single face mode

    def validate_human_face(self, face_roi, bbox, frame_shape):
        """Enhanced human face validation using multiple methods"""
        try:
            # Use the enhanced face validator
            is_valid, confidence, reasons = self.face_validator.multi_detector_validation(
                face_roi, bbox, frame_shape
            )
            
            if is_valid:
                print(f"✅ Human face validated: {confidence:.1%} confidence")
                return True, confidence, reasons
            else:
                print(f"🚫 Non-human face rejected: {confidence:.1%} confidence - {reasons}")
                return False, confidence, reasons
                
        except Exception as e:
            print(f"⚠️ Face validation error: {e}")
            return False, 0.0, ["Validation error"]

    def track_emotion_change(self, new_emotion, confidence, emotion_context):
        """Track emotion changes with enhanced analytics"""
        current_time = time.time()
        
        if self.current_emotion is None:
            # First emotion detection
            self.current_emotion = new_emotion
            self.emotion_start_time = current_time
            self.last_emotion_change = current_time
            
            emotion_record = {
                'emotion': new_emotion,
                'confidence': float(confidence),
                'start_time': current_time,
                'timestamp': datetime.now().isoformat(),
                'context': emotion_context
            }
            
            self.emotion_history.append(new_emotion)
            self.emotion_intensity_history.append(confidence)
            return True, emotion_record
        
        previous_emotion = self.current_emotion
        emotion_changed = previous_emotion != new_emotion
        
        if emotion_changed:
            duration = current_time - self.emotion_start_time
            
            # Record transition
            transition = {
                'from': previous_emotion,
                'to': new_emotion,
                'duration': float(duration),
                'timestamp': datetime.now().isoformat()
            }
            self.emotion_transitions.append(transition)
            
            emotion_record = {
                'emotion': new_emotion,
                'confidence': float(confidence),
                'start_time': current_time,
                'previous_duration': float(duration),
                'timestamp': datetime.now().isoformat(),
                'transition_from': previous_emotion,
                'context': emotion_context
            }
            
            self.current_emotion = new_emotion
            self.emotion_start_time = current_time
            self.last_emotion_change = current_time
            self.emotion_history.append(new_emotion)
            self.emotion_intensity_history.append(confidence)
            
            print(f"🎭 Emotion changed: {previous_emotion} → {new_emotion} ({confidence:.1%})")
            return True, emotion_record
        
        else:
            # Same emotion, update intensity
            self.emotion_history.append(new_emotion)
            self.emotion_intensity_history.append(confidence)
            return False, None

    def get_enhanced_analytics(self):
        """Get comprehensive emotion analytics"""
        if self.current_emotion is None:
            return {}
        
        current_time = time.time()
        current_duration = current_time - self.emotion_start_time
        
        if not self.emotion_history:
            return {}
        
        history = list(self.emotion_history)
        intensity_history = list(self.emotion_intensity_history)
        
        # Calculate stability
        recent_emotions = history[-15:] if len(history) >= 15 else history
        same_emotion_count = sum(1 for e in recent_emotions if e == self.current_emotion)
        stability = float(same_emotion_count / len(recent_emotions)) if recent_emotions else 0.0
        
        # Calculate emotion frequency
        emotion_counts = defaultdict(int)
        for emotion in history:
            emotion_counts[emotion] += 1
        
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "unknown"
        
        # Calculate intensity metrics
        current_intensity = np.mean(intensity_history[-10:]) if intensity_history else 0.0
        intensity_trend = "stable"
        if len(intensity_history) >= 5:
            recent_avg = np.mean(intensity_history[-5:])
            previous_avg = np.mean(intensity_history[-10:-5]) if len(intensity_history) >= 10 else recent_avg
            if recent_avg > previous_avg + 0.1:
                intensity_trend = "increasing"
            elif recent_avg < previous_avg - 0.1:
                intensity_trend = "decreasing"
        
        # Categorize current emotion
        emotion_category = "neutral"
        for category, emotions in self.EMOTION_CATEGORIES.items():
            if any(emot in self.current_emotion for emot in emotions):
                emotion_category = category
                break
        
        return {
            'current_emotion': self.current_emotion,
            'current_emotion_duration': float(current_duration),
            'emotion_stability': stability,
            'dominant_emotion': dominant_emotion,
            'emotion_category': emotion_category,
            'current_intensity': float(current_intensity),
            'intensity_trend': intensity_trend,
            'total_transitions': len(self.emotion_transitions),
            'recent_emotion_trend': recent_emotions[-8:],
            'mood_stability': min(1.0, stability * 1.2),  # Enhanced stability metric
            'engagement_level': self.calculate_engagement_level(),
            'emotional_variability': self.calculate_emotional_variability()
        }

    def calculate_engagement_level(self):
        """Calculate how engaged the person appears"""
        if not self.emotion_history:
            return 0.0
        
        recent_emotions = list(self.emotion_history)[-10:]
        positive_count = sum(1 for e in recent_emotions if any(pos in e for pos in self.EMOTION_CATEGORIES['positive']))
        engagement = positive_count / len(recent_emotions)
        return float(engagement)

    def calculate_emotional_variability(self):
        """Calculate how much emotions are changing"""
        if len(self.emotion_history) < 5:
            return 0.0
        
        recent_emotions = list(self.emotion_history)[-10:]
        unique_emotions = len(set(recent_emotions))
        variability = unique_emotions / len(recent_emotions)
        return float(variability)

    def should_process_frame(self):
        """Control processing rate"""
        current_time = time.time()
        if current_time - self.last_processed_time >= self.processing_interval:
            self.last_processed_time = current_time
            return True
        return False

    def update(self, detections, face_rois, frame_shape):
        """Main update function - SINGLE FACE MODE with HUMAN VALIDATION"""
        current_time = time.time()
        
        if not self.should_process_frame():
            return self.current_track if self.current_track else {}
        
        print(f"\n🔄 FRAME UPDATE: {len(detections)} detections")
        
        # Cleanup if no face for too long
        if self.current_track and current_time - self.current_track['last_seen'] > self.max_age:
            print("🗑️ Removing old face track")
            self._cleanup_track()
        
        # Process only the first detection (single face mode)
        if detections and face_rois:
            detection = detections[0]
            face_roi = face_rois[0]
            
            print(f"🔍 Processing single face detection")
            
            # HUMAN FACE VALIDATION - NEW ENHANCED CHECK
            is_human, validation_confidence, validation_reasons = self.validate_human_face(
                face_roi, detection, frame_shape
            )
            
            if not is_human:
                print(f"🚫 Rejected: Not a valid human face - {validation_reasons}")
                return self.current_track if self.current_track else {}
            
            # Get face embedding
            new_embedding = self.get_face_embedding(face_roi)
            if new_embedding is None:
                print("🚫 No valid embedding")
                return self.current_track if self.current_track else {}
            
            # Enhanced emotion analysis
            emotion, confidence, emotion_context = self.analyze_enhanced_emotion(face_roi)
            if emotion is None:
                print("🚫 No valid emotion")
                return self.current_track if self.current_track else {}
            
            print(f"✅ Enhanced emotion: {emotion} ({confidence:.1%})")
            
            # Check if same face
            if not self.is_same_face(new_embedding):
                print("🔄 Different face detected, updating track")
                self._cleanup_track()  # Cleanup previous track
            
            # Track emotion change
            emotion_changed, emotion_record = self.track_emotion_change(
                emotion, confidence, emotion_context
            )
            
            analytics = self.get_enhanced_analytics()
            
            # Update or create track
            if self.current_track:
                # Update existing track
                self.current_track.update({
                    'bbox': detection,
                    'last_seen': current_time,
                    'hit_counter': self.current_track['hit_counter'] + 1,
                    'current_emotion': emotion,
                    'emotion_confidence': float(confidence),
                    'emotion_changed': bool(emotion_changed),
                    'emotion_record': emotion_record,
                    'emotion_analytics': analytics,
                    'emotion_context': emotion_context,
                    'is_high_confidence': bool(confidence >= self.HIGH_CONFIDENCE_THRESHOLD),
                    'is_super_confidence': bool(confidence >= self.SUPER_HIGH_CONFIDENCE),
                    'confirmed': bool(self.current_track['hit_counter'] > 2),
                    'human_validation_confidence': float(validation_confidence),
                    'human_validation_passed': True
                })
                
                # Update embedding
                alpha = 0.2  # Higher learning rate for single face
                self.face_embedding = (
                    alpha * new_embedding + (1 - alpha) * self.face_embedding
                )
                
                print(f"🔁 Updated Single Face: {emotion} ({confidence:.1%})")
                
            else:
                # Create new track
                self.current_track = {
                    'bbox': detection,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'hit_counter': 1,
                    'person_name': "Single_Face",
                    'current_emotion': emotion,
                    'emotion_confidence': float(confidence),
                    'emotion_changed': bool(emotion_changed),
                    'emotion_record': emotion_record,
                    'emotion_analytics': analytics,
                    'emotion_context': emotion_context,
                    'is_high_confidence': bool(confidence >= self.HIGH_CONFIDENCE_THRESHOLD),
                    'is_super_confidence': bool(confidence >= self.SUPER_HIGH_CONFIDENCE),
                    'confirmed': False,
                    'human_validation_confidence': float(validation_confidence),
                    'human_validation_passed': True
                }
                
                self.face_embedding = new_embedding
                print(f"👤 NEW Single Face: {emotion} ({confidence:.1%})")
        
        # Return current track if exists
        if self.current_track:
            return {'single_face': self.current_track}
        else:
            print("📊 No active face track")
            return {}

    def _cleanup_track(self):
        """Cleanup track data"""
        self.current_track = None
        self.face_embedding = None
        self.current_emotion = None
        self.emotion_start_time = None
        self.last_emotion_change = None
        # Keep history for analytics continuity

# Initialize ENHANCED tracker
emotion_tracker = SingleFaceEmotionTracker()

def detect_faces_fixed(frame):
    """Fixed face detection - single face focus with human validation"""
    if not DEEPFACE_AVAILABLE:
        return [], []
    
    try:
        detections = DeepFace.extract_faces(
            frame,
            detector_backend='mtcnn',
            enforce_detection=False,
            align=False
        )
        
        bboxes = []
        face_rois = []
        
        for detection in detections:
            if 'facial_area' in detection:
                x, y, w, h = detection['facial_area']['x'], detection['facial_area']['y'], \
                             detection['facial_area']['w'], detection['facial_area']['h']
                
                # Reasonable validation
                if (x >= 10 and y >= 10 and 
                    x + w <= frame.shape[1] - 10 and y + h <= frame.shape[0] - 10 and
                    w >= 80 and h >= 80):
                    
                    bboxes.append([int(x), int(y), int(w), int(h)])
                    
                    padding = 10
                    x1, y1 = max(0, x-padding), max(0, y-padding)
                    x2, y2 = min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding)
                    face_roi = frame[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        face_rois.append(face_roi)
        
        print(f"🔍 Detected: {len(bboxes)} faces")
        return bboxes, face_rois
        
    except Exception as e:
        print(f"⚠️ Face detection error: {e}")
        return [], []

async def process_frame_fixed(frame_data):
    """Enhanced frame processing with single face focus and human validation"""
    try:
        start_time = time.time()
        
        # Decode image
        if ',' in frame_data:
            img_data = base64.b64decode(frame_data.split(',')[1])
        else:
            img_data = base64.b64decode(frame_data)
        
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return []
        
        # Detect faces
        bboxes, face_rois = detect_faces_fixed(frame)
        
        # Update tracker - single face mode
        tracked_faces = emotion_tracker.update(bboxes, face_rois, frame.shape)
        
        # Prepare enhanced response
        detections = []
        for track_key, track_data in tracked_faces.items():
            if track_data.get('current_emotion'):
                x, y, w, h = track_data['bbox']
                analytics = track_data.get('emotion_analytics', {})
                context = track_data.get('emotion_context', {})
                
                detection_data = {
                    'face_id': 1,  # Single face ID
                    'person_name': track_data['person_name'],
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'timestamp': datetime.now().isoformat(),
                    
                    # Enhanced emotion data
                    'current_emotion': track_data['current_emotion'],
                    'emotion_confidence': float(track_data['emotion_confidence']),
                    'emotion_category': analytics.get('emotion_category', 'neutral'),
                    'is_high_confidence': bool(track_data.get('is_high_confidence', False)),
                    'is_super_confidence': bool(track_data.get('is_super_confidence', False)),
                    'emotion_changed': bool(track_data.get('emotion_changed', False)),
                    
                    # Enhanced analytics
                    'emotion_stability': float(analytics.get('emotion_stability', 0)),
                    'current_emotion_duration': float(analytics.get('current_emotion_duration', 0)),
                    'current_intensity': float(analytics.get('current_intensity', 0)),
                    'intensity_trend': analytics.get('intensity_trend', 'stable'),
                    'dominant_emotion': analytics.get('dominant_emotion', 'unknown'),
                    'engagement_level': float(analytics.get('engagement_level', 0)),
                    'mood_stability': float(analytics.get('mood_stability', 0)),
                    'emotional_variability': float(analytics.get('emotional_variability', 0)),
                    
                    # Context information
                    'secondary_emotion': context.get('secondary_emotion'),
                    'secondary_confidence': float(context.get('secondary_confidence', 0)),
                    'age': context.get('age', 0),
                    'gender': context.get('gender', 'unknown'),
                    'raw_emotion': context.get('raw_emotion', 'unknown'),
                    
                    # Human validation info
                    'human_validation_passed': bool(track_data.get('human_validation_passed', False)),
                    'human_validation_confidence': float(track_data.get('human_validation_confidence', 0)),
                    
                    # Tracking info
                    'track_count': int(track_data['hit_counter']),
                    'confirmed': bool(track_data.get('confirmed', False)),
                    
                    # Recent trend
                    'recent_emotion_trend': analytics.get('recent_emotion_trend', [])
                }
                
                # Include emotion record if changed
                if track_data.get('emotion_changed') and track_data.get('emotion_record'):
                    emotion_record = track_data['emotion_record']
                    if emotion_record:
                        converted_record = {}
                        for key, value in emotion_record.items():
                            if isinstance(value, (np.floating, np.float32, np.float64)):
                                converted_record[key] = float(value)
                            elif isinstance(value, (np.integer, np.int32, np.int64)):
                                converted_record[key] = int(value)
                            elif isinstance(value, np.bool_):
                                converted_record[key] = bool(value)
                            else:
                                converted_record[key] = value
                        detection_data['emotion_record'] = converted_record
                
                detections.append(detection_data)
        
        processing_time = (time.time() - start_time) * 1000
        
        if detections:
            track = detections[0]
            confidence_level = "HIGH" if track['is_high_confidence'] else "MEDIUM"
            if track['is_super_confidence']:
                confidence_level = "SUPER"
            
            human_valid = "✅ HUMAN" if track['human_validation_passed'] else "❌ NON-HUMAN"
            
            print(f"✅ SINGLE FACE TRACKING: {track['current_emotion']} ({confidence_level} confidence, {processing_time:.1f}ms)")
            print(f"   👤 {human_valid} - {track['human_validation_confidence']:.1%} validation")
            print(f"   📊 Analytics: {track['emotion_category']} category, {track['engagement_level']:.0%} engagement")
        else:
            print(f"🔍 No face detected this frame")
        
        return detections
        
    except Exception as e:
        print(f"❌ Frame processing error: {e}")
        return []

async def handle_client(websocket):
    """Enhanced WebSocket handler"""
    client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
    print(f"🎉 Client connected from {client_ip}")
    
    try:
        async for message in websocket:
            data = json.loads(message)
            
            if data['type'] == 'frame':
                detections = await process_frame_fixed(data['frame'])
                
                response = {
                    'type': 'premium_emotion_results',
                    'detections': detections,
                    'timestamp': datetime.now().isoformat(),
                    'total_tracks': 1 if detections else 0,
                    'active_tracks': len(detections),
                    'high_confidence_tracks': sum(1 for d in detections if d.get('is_high_confidence', False)),
                    'human_validated_tracks': sum(1 for d in detections if d.get('human_validation_passed', False)),
                    'tracker': 'single_face_emotion_tracker',
                    'confidence_threshold': float(emotion_tracker.MIN_EMOTION_CONFIDENCE),
                    'human_validation_enabled': True,
                    'mode': 'single_face'
                }
                
                await websocket.send(json.dumps(response, cls=NumpyEncoder))
                
    except websockets.exceptions.ConnectionClosed:
        print(f"📞 Client {client_ip} disconnected")
    except Exception as e:
        print(f"❌ WebSocket error for {client_ip}: {e}")

async def main():
    async with websockets.serve(handle_client, "localhost", 8765):
        print("🌈 ENHANCED WebSocket server running on ws://localhost:8765")
        print("🎯 SINGLE FACE EMOTION TRACKING - ENHANCED DETECTION")
        print("🔧 Enhanced Features:")
        print("   • Single face tracking mode")
        print("   • HUMAN-ONLY face validation")
        print("   • Multi-detector face verification")
        print("   • Enhanced emotion interpretation")
        print("   • Emotion intensity tracking")
        print("   • Contextual emotion analysis")
        print("   • Engagement level calculation")
        print("   • Mood stability metrics")
        print("   • Multiple confidence levels")
        print("   • Faster processing (0.3s intervals)")
        
        await asyncio.Future()

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 PREMIUM Emotion Tracking Server - SINGLE FACE MODE")
    print("🎯 ENHANCED EMOTION DETECTION")
    print("🎯 HUMAN-ONLY FACE VALIDATION")
    print("🎯 CONTEXTUAL INTERPRETATION")
    print("=" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
# # V10 STRICT Face Tracking Server - DEBUGGED Face Storage & Matching
# import cv2
# import asyncio
# import websockets
# import json
# import base64
# import numpy as np
# import time
# from datetime import datetime
# import logging
# from collections import deque, defaultdict

# print("🚀 Starting DEBUGGED Face Tracking Server...")

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Load DeepFace
# try:
#     from deepface import DeepFace
#     DEEPFACE_AVAILABLE = True
#     print("✅ DeepFace loaded successfully!")
# except Exception as e:
#     DEEPFACE_AVAILABLE = False
#     print(f"❌ DeepFace error: {e}")
#     exit(1)

# # Debugged Face Tracker with Enhanced Storage Analysis
# class DebuggedFaceTracker:
#     def __init__(self):
#         self.tracks = {}
#         self.next_id = 1
#         self.max_age = 5.0
#         self.max_tracks = 100
        
#         # 🎯 STRICT CONFIDENCE THRESHOLDS
#         self.MIN_SIMILARITY = 0.85
#         self.INITIAL_SIMILARITY_THRESHOLD = 0.92
#         self.MIN_EMOTION_CONFIDENCE = 0.70
        
#         # Face embedding storage - DEBUGGED
#         self.face_embeddings = {}  # track_id -> 128D numpy array
#         self.embedding_quality = {}  # track_id -> quality score
        
#         # Emotion tracking
#         self.emotion_history = defaultdict(list)
#         self.last_emotion = {}
#         self.emotion_start_time = {}
        
#         # Track processing rate
#         self.last_processed_time = 0
#         self.min_processing_interval = 1.0
        
#         # Debug counters
#         self.debug_counters = {
#             'total_detections': 0,
#             'valid_faces': 0,
#             'new_tracks': 0,
#             'updated_tracks': 0,
#             'embedding_failures': 0,
#             'similarity_checks': 0
#         }
        
#         print(f"🎯 Debugged tracking: Similarity>{self.MIN_SIMILARITY}")

#     def debug_embedding_storage(self, track_id, embedding, operation="store"):
#         """Debug embedding storage operations"""
#         if embedding is not None:
#             shape = embedding.shape
#             norm = np.linalg.norm(embedding)
#             min_val = np.min(embedding)
#             max_val = np.max(embedding)
#             print(f"🔍 EMBEDDING {operation.upper()}: Track_{track_id} | Shape: {shape} | Norm: {norm:.3f} | Range: [{min_val:.3f}, {max_val:.3f}]")
#         else:
#             print(f"❌ EMBEDDING {operation.upper()}: Track_{track_id} | FAILED - None")

#     def debug_similarity_check(self, track_id, similarity, threshold_passed, match_type):
#         """Debug similarity matching operations"""
#         status = "✅ MATCH" if threshold_passed else "❌ NO_MATCH"
#         print(f"🔍 SIMILARITY: {status} | Track_{track_id} | Score: {similarity:.3f} | Type: {match_type}")

#     def debug_track_creation(self, track_id, emotion, embedding_quality):
#         """Debug new track creation"""
#         print(f"👤 NEW TRACK: Person_{track_id} | Emotion: {emotion} | Embedding Quality: {embedding_quality:.3f}")
#         print(f"   📊 Storage: {len(self.face_embeddings)} embeddings, {len(self.tracks)} tracks")

#     def debug_track_update(self, track_id, similarity, emotion, hit_counter):
#         """Debug track updates"""
#         print(f"🔁 TRACK UPDATE: Person_{track_id} | Similarity: {similarity:.3f} | Emotion: {emotion} | Hits: {hit_counter}")

#     def get_face_embedding(self, face_roi):
#         """Extract face embedding with enhanced debugging"""
#         try:
#             # Size validation
#             if (face_roi.shape[0] < 80 or face_roi.shape[1] < 80 or 
#                 face_roi.shape[0] > 400 or face_roi.shape[1] > 400):
#                 print(f"🚫 Embedding failed: Invalid size {face_roi.shape}")
#                 return None, 0.0
                
#             print(f"🔍 Extracting embedding from ROI shape: {face_roi.shape}")
            
#             # Use Facenet with strict enforcement
#             embedding_obj = DeepFace.represent(
#                 face_roi,
#                 model_name='Facenet',
#                 enforce_detection=True,
#                 detector_backend='mtcnn',
#                 align=True
#             )
            
#             if embedding_obj and len(embedding_obj) > 0:
#                 embedding = np.array(embedding_obj[0]['embedding'])
                
#                 # Calculate embedding quality
#                 embedding_norm = np.linalg.norm(embedding)
#                 quality_score = min(embedding_norm / 10.0, 1.0)
                
#                 if quality_score >= 0.6:
#                     print(f"✅ Embedding success: Quality {quality_score:.3f}, Shape {embedding.shape}")
#                     return embedding, quality_score
#                 else:
#                     print(f"🚫 Embedding failed: Low quality {quality_score:.3f}")
#                     self.debug_counters['embedding_failures'] += 1
                    
#         except Exception as e:
#             print(f"🚫 Embedding extraction error: {e}")
#             self.debug_counters['embedding_failures'] += 1
        
#         return None, 0.0

#     def calculate_similarity(self, embedding1, embedding2):
#         """Calculate cosine similarity with debugging"""
#         if embedding1 is None or embedding2 is None:
#             print("🚫 Similarity calculation: One or both embeddings are None")
#             return 0.0
        
#         try:
#             # Normalize embeddings
#             emb1_norm = embedding1 / np.linalg.norm(embedding1)
#             emb2_norm = embedding2 / np.linalg.norm(embedding2)
            
#             # Calculate cosine similarity
#             similarity = np.dot(emb1_norm, emb2_norm)
#             similarity = max(0.0, min(1.0, similarity))
            
#             self.debug_counters['similarity_checks'] += 1
#             return similarity
            
#         except Exception as e:
#             print(f"🚫 Similarity calculation error: {e}")
#             return 0.0

#     def find_best_match(self, new_embedding):
#         """Find best matching track with detailed debugging"""
#         best_track_id = None
#         best_similarity = 0.0
#         similarity_scores = {}
        
#         print(f"🔍 MATCHING: Checking against {len(self.face_embeddings)} stored embeddings")
        
#         for track_id, existing_embedding in self.face_embeddings.items():
#             if track_id not in self.tracks:
#                 print(f"🚫 Skipping Track_{track_id}: Not in active tracks")
#                 continue
                
#             similarity = self.calculate_similarity(existing_embedding, new_embedding)
#             similarity_scores[track_id] = similarity
            
#             print(f"   🔄 Compare with Track_{track_id}: Similarity = {similarity:.3f}")
            
#             if similarity > best_similarity and similarity >= self.MIN_SIMILARITY:
#                 best_similarity = similarity
#                 best_track_id = track_id
        
#         # Debug all similarity scores
#         if similarity_scores:
#             print(f"📊 ALL SIMILARITIES: {similarity_scores}")
        
#         if best_track_id is not None:
#             self.debug_similarity_check(best_track_id, best_similarity, True, "BEST_MATCH")
#         else:
#             print(f"🔍 NO MATCH FOUND: Best similarity was {max(similarity_scores.values()) if similarity_scores else 0:.3f} (needs {self.MIN_SIMILARITY})")
        
#         return best_track_id, best_similarity

#     def create_new_track(self, detection, current_emotion, emotion_confidence, new_embedding, embedding_quality):
#         """Create a new track with proper embedding storage"""
#         new_id = self.next_id
        
#         # Track initial emotion
#         emotion_changed, emotion_record = self.track_emotion_change(
#             new_id, current_emotion, emotion_confidence
#         )
        
#         # Store track data
#         self.tracks[new_id] = {
#             'bbox': detection,
#             'first_seen': time.time(),
#             'last_seen': time.time(),
#             'hit_counter': 1,
#             'similarity': 1.0,  # Fresh track
#             'person_name': f"Person_{new_id}",
#             'current_emotion': current_emotion,
#             'emotion_confidence': emotion_confidence,
#             'emotion_changed': emotion_changed,
#             'emotion_record': emotion_record if emotion_changed else None,
#             'embedding_quality': embedding_quality
#         }
        
#         # STORE THE EMBEDDING - This is critical for future matching
#         self.face_embeddings[new_id] = new_embedding
#         self.embedding_quality[new_id] = embedding_quality
        
#         self.debug_embedding_storage(new_id, new_embedding, "STORED")
#         self.debug_track_creation(new_id, current_emotion, embedding_quality)
        
#         self.next_id += 1
#         self.debug_counters['new_tracks'] += 1
        
#         return new_id

#     def update_existing_track(self, track_id, detection, current_emotion, emotion_confidence, new_embedding, similarity):
#         """Update existing track with embedding refinement"""
#         track = self.tracks[track_id]
        
#         # Track emotion change
#         emotion_changed, emotion_record = self.track_emotion_change(
#             track_id, current_emotion, emotion_confidence
#         )
        
#         # Update track data
#         track.update({
#             'bbox': detection,
#             'last_seen': time.time(),
#             'hit_counter': track['hit_counter'] + 1,
#             'similarity': similarity,
#             'current_emotion': current_emotion,
#             'emotion_confidence': emotion_confidence,
#             'emotion_changed': emotion_changed,
#             'emotion_record': emotion_record if emotion_changed else None
#         })
        
#         # UPDATE THE EMBEDDING - Gradually refine with new data
#         alpha = 0.1  # Learning rate - small to maintain stability
#         old_embedding = self.face_embeddings[track_id]
#         self.face_embeddings[track_id] = (
#             alpha * new_embedding + (1 - alpha) * old_embedding
#         )
        
#         print(f"🔄 Embedding updated: Track_{track_id} with alpha={alpha}")
#         self.debug_track_update(track_id, similarity, current_emotion, track['hit_counter'])
        
#         self.debug_counters['updated_tracks'] += 1

#     def track_emotion_change(self, track_id, current_emotion, current_confidence):
#         """Track emotion changes - unchanged from previous"""
#         current_time = time.time()
        
#         if track_id not in self.last_emotion:
#             self.last_emotion[track_id] = current_emotion
#             self.emotion_start_time[track_id] = current_time
            
#             emotion_record = {
#                 'emotion': current_emotion,
#                 'confidence': current_confidence,
#                 'start_time': current_time,
#                 'end_time': current_time,
#                 'duration': 0,
#                 'timestamp': datetime.now().isoformat()
#             }
#             self.emotion_history[track_id].append(emotion_record)
#             return True, emotion_record
        
#         last_emotion = self.last_emotion[track_id]
        
#         if last_emotion != current_emotion:
#             duration = current_time - self.emotion_start_time[track_id]
            
#             if self.emotion_history[track_id]:
#                 previous_record = self.emotion_history[track_id][-1]
#                 previous_record['end_time'] = current_time
#                 previous_record['duration'] = duration
            
#             emotion_record = {
#                 'emotion': current_emotion,
#                 'confidence': current_confidence,
#                 'start_time': current_time,
#                 'end_time': current_time,
#                 'duration': 0,
#                 'timestamp': datetime.now().isoformat()
#             }
#             self.emotion_history[track_id].append(emotion_record)
            
#             self.last_emotion[track_id] = current_emotion
#             self.emotion_start_time[track_id] = current_time
            
#             print(f"🎭 Emotion changed: {last_emotion} → {current_emotion} (duration: {duration:.1f}s)")
#             return True, emotion_record
        
#         else:
#             if self.emotion_history[track_id]:
#                 current_record = self.emotion_history[track_id][-1]
#                 current_record['end_time'] = current_time
#                 current_record['duration'] = current_time - current_record['start_time']
            
#             return False, None

#     def analyze_emotion(self, face_roi):
#         """Analyze emotion - unchanged from previous"""
#         try:
#             analysis = DeepFace.analyze(
#                 face_roi,
#                 actions=['emotion'],
#                 detector_backend='mtcnn',
#                 enforce_detection=True,
#                 silent=True
#             )
            
#             if analysis and len(analysis) > 0:
#                 result = analysis[0]
#                 emotion_data = result.get('emotion', {})
                
#                 if emotion_data:
#                     dominant_emotion, emotion_confidence = max(emotion_data.items(), key=lambda x: x[1])
#                     emotion_confidence_pct = emotion_confidence / 100.0
                    
#                     if emotion_confidence_pct >= self.MIN_EMOTION_CONFIDENCE:
#                         return dominant_emotion, emotion_confidence_pct
                        
#         except Exception as e:
#             print(f"⚠️ Emotion analysis error: {e}")
            
#         return None, 0.0

#     def should_process_frame(self):
#         """Control processing rate"""
#         current_time = time.time()
#         if current_time - self.last_processed_time >= self.min_processing_interval:
#             self.last_processed_time = current_time
#             return True
#         return False

#     def update(self, detections, face_rois, frame_shape):
#         """Main update function with enhanced debugging"""
#         current_time = time.time()
        
#         # Rate limiting
#         if not self.should_process_frame():
#             return self.tracks
        
#         print(f"\n{'='*50}")
#         print(f"🔄 FRAME UPDATE: {len(detections)} detections, {len(self.tracks)} active tracks")
#         print(f"{'='*50}")
        
#         # Cleanup old tracks
#         for track_id in list(self.tracks.keys()):
#             if current_time - self.tracks[track_id]['last_seen'] > self.max_age:
#                 print(f"🗑️ Removing stale track: Person_{track_id}")
#                 if track_id in self.face_embeddings:
#                     del self.face_embeddings[track_id]
#                 if track_id in self.embedding_quality:
#                     del self.embedding_quality[track_id]
#                 if track_id in self.emotion_history:
#                     del self.emotion_history[track_id]
#                 if track_id in self.last_emotion:
#                     del self.last_emotion[track_id]
#                 if track_id in self.emotion_start_time:
#                     del self.emotion_start_time[track_id]
#                 del self.tracks[track_id]
        
#         self.debug_counters['total_detections'] += len(detections)
        
#         # Process each detection
#         for i, (detection, face_roi) in enumerate(zip(detections, face_rois)):
#             print(f"\n🔍 PROCESSING DETECTION {i+1}/{len(detections)}")
            
#             # Step 1: Extract face embedding
#             new_embedding, embedding_quality = self.get_face_embedding(face_roi)
            
#             if new_embedding is None:
#                 print("🚫 Skipping: No valid embedding")
#                 continue
            
#             # Step 2: Analyze emotion
#             current_emotion, emotion_confidence = self.analyze_emotion(face_roi)
            
#             if current_emotion is None:
#                 print("🚫 Skipping: No valid emotion")
#                 continue
            
#             self.debug_counters['valid_faces'] += 1
            
#             # Step 3: Find best match among existing tracks
#             best_track_id, best_similarity = self.find_best_match(new_embedding)
            
#             # Step 4: Decide whether to update existing or create new
#             if best_track_id is not None and best_similarity >= self.INITIAL_SIMILARITY_THRESHOLD:
#                 # Update existing track
#                 self.update_existing_track(
#                     best_track_id, detection, current_emotion, emotion_confidence, 
#                     new_embedding, best_similarity
#                 )
#             else:
#                 # Create new track
#                 self.create_new_track(
#                     detection, current_emotion, emotion_confidence,
#                     new_embedding, embedding_quality
#                 )
        
#         # Print summary
#         print(f"\n📊 FRAME SUMMARY:")
#         print(f"   • Total detections: {self.debug_counters['total_detections']}")
#         print(f"   • Valid faces: {self.debug_counters['valid_faces']}")
#         print(f"   • New tracks: {self.debug_counters['new_tracks']}")
#         print(f"   • Updated tracks: {self.debug_counters['updated_tracks']}")
#         print(f"   • Embedding failures: {self.debug_counters['embedding_failures']}")
#         print(f"   • Similarity checks: {self.debug_counters['similarity_checks']}")
#         print(f"   • Current storage: {len(self.face_embeddings)} embeddings, {len(self.tracks)} tracks")
        
#         return self.tracks

# # Initialize debugged tracker
# face_tracker = DebuggedFaceTracker()

# # Rest of the code (detect_human_faces_strict, process_frame, WebSocket handlers) remains the same...
# def detect_human_faces_strict(frame):
#     """Use DeepFace with strict human face detection"""
#     if not DEEPFACE_AVAILABLE:
#         return [], []
    
#     try:
#         detections = DeepFace.extract_faces(
#             frame,
#             detector_backend='mtcnn',
#             enforce_detection=False,
#             align=False
#         )
        
#         bboxes = []
#         face_rois = []
        
#         for detection in detections:
#             if 'facial_area' in detection:
#                 x, y, w, h = detection['facial_area']['x'], detection['facial_area']['y'], \
#                              detection['facial_area']['w'], detection['facial_area']['h']
                
#                 # Initial validation
#                 if (x >= 10 and y >= 10 and 
#                     x + w <= frame.shape[1] - 10 and y + h <= frame.shape[0] - 10 and
#                     w >= 80 and h >= 80 and w <= 300 and h <= 300):
                    
#                     bboxes.append([x, y, w, h])
                    
#                     padding = 8
#                     x1, y1 = max(0, x-padding), max(0, y-padding)
#                     x2, y2 = min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding)
#                     face_roi = frame[y1:y2, x1:x2]
                    
#                     if face_roi.size > 0:
#                         face_rois.append(face_roi)
        
#         print(f"🔍 Initial detection: {len(bboxes)} potential faces")
#         return bboxes, face_rois
        
#     except Exception as e:
#         print(f"⚠️ Face detection error: {e}")
#         return [], []

# async def process_frame(frame_data):
#     try:
#         start_time = time.time()
        
#         # Decode image
#         if ',' in frame_data:
#             img_data = base64.b64decode(frame_data.split(',')[1])
#         else:
#             img_data = base64.b64decode(frame_data)
        
#         nparr = np.frombuffer(img_data, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if frame is None:
#             return []
        
#         # Detect human faces
#         bboxes, face_rois = detect_human_faces_strict(frame)
        
#         # Update tracker
#         tracked_faces = face_tracker.update(bboxes, face_rois, frame.shape)
        
#         # Prepare response
#         detections = []
#         for track_id, track_data in tracked_faces.items():
#             similarity = track_data.get('similarity', 0)
#             current_emotion = track_data.get('current_emotion')
#             emotion_confidence = track_data.get('emotion_confidence', 0)
            
#             if (similarity >= face_tracker.MIN_SIMILARITY and 
#                 current_emotion is not None and 
#                 emotion_confidence >= face_tracker.MIN_EMOTION_CONFIDENCE):
                
#                 x, y, w, h = track_data['bbox']
                
#                 detection_data = {
#                     'face_id': track_id,
#                     'person_name': track_data.get('person_name', f'Person_{track_id}'),
#                     'bbox': [int(x), int(y), int(w), int(h)],
#                     'timestamp': datetime.now().isoformat(),
#                     'track_count': track_data['hit_counter'],
#                     'similarity_score': round(track_data.get('similarity', 1.0), 3),
#                     'confirmed': track_data['hit_counter'] > 2,
#                     'current_emotion': current_emotion,
#                     'emotion_confidence': round(emotion_confidence, 3),
#                     'emotion_changed': track_data.get('emotion_changed', False),
#                     'embedding_quality': round(track_data.get('embedding_quality', 0), 3)
#                 }
                
#                 if track_data.get('emotion_changed') and track_data.get('emotion_record'):
#                     detection_data['emotion_record'] = track_data['emotion_record']
                
#                 if track_id in face_tracker.emotion_history:
#                     emotion_history = face_tracker.emotion_history[track_id]
#                     detection_data['emotion_history_count'] = len(emotion_history)
#                     detection_data['recent_emotions'] = emotion_history[-3:] if len(emotion_history) >= 3 else emotion_history
                
#                 detections.append(detection_data)
        
#         processing_time = (time.time() - start_time) * 1000
        
#         if detections:
#             print(f"✅ DEBUGGED TRACKING: {len(detections)} humans in {processing_time:.1f}ms")
#         else:
#             print(f"🔍 No faces tracked this frame")
        
#         return detections
        
#     except Exception as e:
#         print(f"❌ Frame processing error: {e}")
#         return []

# # WebSocket handlers remain the same...
# async def handle_client(websocket):
#     client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
#     print(f"🎉 Client connected from {client_ip}")
    
#     try:
#         async for message in websocket:
#             data = json.loads(message)
            
#             if data['type'] == 'frame':
#                 detections = await process_frame(data['frame'])
                
#                 response = {
#                     'type': 'tracking_results', 
#                     'detections': detections,
#                     'timestamp': datetime.now().isoformat(),
#                     'total_tracks': len(face_tracker.tracks),
#                     'active_tracks': len(detections),
#                     'max_tracks': face_tracker.max_tracks,
#                     'tracker': 'debugged_face_tracker'
#                 }
                
#                 await websocket.send(json.dumps(response))
                
#     except websockets.exceptions.ConnectionClosed:
#         print(f"📞 Client {client_ip} disconnected")
#     except Exception as e:
#         print(f"❌ WebSocket error for {client_ip}: {e}")

# async def main():
#     async with websockets.serve(handle_client, "localhost", 8765):
#         print("🌈 WebSocket server running on ws://localhost:8765")
#         print("🎯 DEBUGGED FACE TRACKING - STORAGE & MATCHING ANALYSIS")
#         print("🔧 Enhanced Debugging Features:")
#         print("   • Embedding storage visualization")
#         print("   • Similarity matching details")
#         print("   • Track creation/update logging")
#         print("   • Performance counters")
#         print("   • Storage state monitoring")
        
#         await asyncio.Future()

# if __name__ == "__main__":
#     print("=" * 60)
#     print("🤖 DEBUGGED Face Tracking - Storage & Matching Analysis")
#     print("🎯 EMBEDDING STORAGE - SIMILARITY MATCHING - DEBUG LOGS")
#     print("=" * 60)
    
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print("\n🛑 Server stopped by user")
#     except Exception as e:
#         print(f"❌ Server error: {e}")
# # # # V10 STRICT Face Tracking Server - Enhanced Human Validation
# # # import cv2
# # # import asyncio
# # # import websockets
# # # import json
# # # import base64
# # # import numpy as np
# # # import time
# # # from datetime import datetime
# # # import logging
# # # from collections import deque, defaultdict

# # # print("🚀 Starting ENHANCED STRICT Face Tracking Server...")

# # # # Setup logging
# # # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# # # logger = logging.getLogger(__name__)

# # # # Load DeepFace
# # # try:
# # #     from deepface import DeepFace
# # #     DEEPFACE_AVAILABLE = True
# # #     print("✅ DeepFace loaded successfully!")
# # # except Exception as e:
# # #     DEEPFACE_AVAILABLE = False
# # #     print(f"❌ DeepFace error: {e}")
# # #     exit(1)

# # # # Enhanced Face Validator with Multiple Detection Methods
# # class EnhancedFaceValidator:
# #     def __init__(self):
# #         # Load multiple face detectors for validation
# #         self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
# #         # Validation thresholds
# #         self.MIN_FACE_CONFIDENCE = 0.7
# #         self.MIN_EMBEDDING_QUALITY = 0.6
# #         self.ASPECT_RATIO_RANGE = (0.7, 1.4)  # Wider but reasonable for faces
# #         self.SIZE_RANGE = (60, 400)  # Pixel dimensions
# #         self.POSITION_MARGIN = 0.05  # 5% margin from edges
        
# #         print("✅ Enhanced Face Validator initialized with multiple detectors")
    
# #     def calculate_iou(self, boxA, boxB):
# #         """Calculate Intersection over Union for bounding box validation"""
# #         # Convert [x, y, w, h] to [x1, y1, x2, y2]
# #         if len(boxA) == 4 and len(boxB) == 4:
# #             boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
# #             boxB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]
        
# #         xA = max(boxA[0], boxB[0])
# #         yA = max(boxA[1], boxB[1])
# #         xB = min(boxA[2], boxB[2])
# #         yB = min(boxA[3], boxB[3])
        
# #         interArea = max(0, xB - xA) * max(0, yB - yA)
# #         boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
# #         boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
# #         iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
# #         return iou
    
# #     def validate_with_haar_cascade(self, face_roi, original_bbox, frame_shape):
# #         """Validate face using Haar Cascade for additional confirmation"""
# #         try:
# #             # Convert to grayscale for Haar cascade
# #             gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
# #             # Detect faces in the ROI
# #             haar_faces = self.haar_cascade.detectMultiScale(
# #                 gray_roi, 
# #                 scaleFactor=1.1, 
# #                 minNeighbors=3, 
# #                 minSize=(30, 30),
# #                 flags=cv2.CASCADE_SCALE_IMAGE
# #             )
            
# #             if len(haar_faces) > 0:
# #                 # Check if any Haar detection overlaps significantly with original bbox
# #                 for (hx, hy, hw, hh) in haar_faces:
# #                     # Convert to original frame coordinates
# #                     original_x, original_y, original_w, original_h = original_bbox
# #                     haar_bbox = [original_x + hx, original_y + hy, hw, hh]
                    
# #                     iou = self.calculate_iou(original_bbox, haar_bbox)
# #                     if iou > 0.3:  # Significant overlap
# #                         return True, iou
            
# #             return False, 0.0
            
# #         except Exception as e:
# #             return False, 0.0
    
# #     def validate_face_characteristics(self, face_roi, bbox, frame_shape):
# #         """Comprehensive face characteristic validation"""
# #         validation_results = {
# #             'passed': True,
# #             'reasons': [],
# #             'confidence': 1.0
# #         }
        
# #         h, w = face_roi.shape[:2]
# #         frame_height, frame_width = frame_shape[:2]
# #         x, y, bbox_w, bbox_h = bbox
        
# #         # 1. Size validation
# #         if bbox_w < self.SIZE_RANGE[0] or bbox_h < self.SIZE_RANGE[0]:
# #             validation_results['passed'] = False
# #             validation_results['reasons'].append(f"Too small: {bbox_w}x{bbox_h}")
# #             validation_results['confidence'] *= 0.3
        
# #         if bbox_w > self.SIZE_RANGE[1] or bbox_h > self.SIZE_RANGE[1]:
# #             validation_results['passed'] = False
# #             validation_results['reasons'].append(f"Too large: {bbox_w}x{bbox_h}")
# #             validation_results['confidence'] *= 0.3
        
# #         # 2. Aspect ratio validation
# #         aspect_ratio = bbox_w / bbox_h
# #         if aspect_ratio < self.ASPECT_RATIO_RANGE[0] or aspect_ratio > self.ASPECT_RATIO_RANGE[1]:
# #             validation_results['passed'] = False
# #             validation_results['reasons'].append(f"Bad aspect ratio: {aspect_ratio:.2f}")
# #             validation_results['confidence'] *= 0.5
        
# #         # 3. Position validation
# #         margin_x = frame_width * self.POSITION_MARGIN
# #         margin_y = frame_height * self.POSITION_MARGIN
        
# #         if (x < margin_x or y < margin_y or 
# #             x + bbox_w > frame_width - margin_x or 
# #             y + bbox_h > frame_height - margin_y):
# #             validation_results['reasons'].append("Near frame edges")
# #             validation_results['confidence'] *= 0.8  # Less severe penalty
        
# #         # 4. Area ratio validation
# #         bbox_area = bbox_w * bbox_h
# #         frame_area = frame_width * frame_height
# #         area_ratio = bbox_area / frame_area
        
# #         if area_ratio > 0.5:  # Reject if covers more than 50% of frame
# #             validation_results['passed'] = False
# #             validation_results['reasons'].append(f"Too large area: {area_ratio:.2f}")
# #             validation_results['confidence'] *= 0.2
        
# #         if area_ratio < 0.01:  # Reject if too small
# #             validation_results['passed'] = False
# #             validation_results['reasons'].append(f"Too small area: {area_ratio:.2f}")
# #             validation_results['confidence'] *= 0.2
        
# #         # 5. Image quality checks
# #         if h == 0 or w == 0:
# #             validation_results['passed'] = False
# #             validation_results['reasons'].append("Invalid dimensions")
# #             validation_results['confidence'] = 0.0
        
# #         # Check for reasonable color distribution (basic image validity)
# #         if np.mean(face_roi) < 10 or np.mean(face_roi) > 240:
# #             validation_results['reasons'].append("Suspicious brightness")
# #             validation_results['confidence'] *= 0.7
        
# #         return validation_results
    
# #     def multi_detector_validation(self, face_roi, original_bbox, frame_shape):
# #         """Use multiple detection methods to validate face"""
# #         validation_score = 0.0
# #         total_checks = 0
        
# #         # 1. DeepFace verification (already passed to get here)
# #         validation_score += 1.0
# #         total_checks += 1
        
# #         # 2. Haar cascade verification
# #         haar_valid, haar_iou = self.validate_with_haar_cascade(face_roi, original_bbox, frame_shape)
# #         if haar_valid:
# #             validation_score += haar_iou
# #             total_checks += 1
        
# #         # 3. Face characteristics validation
# #         char_results = self.validate_face_characteristics(face_roi, original_bbox, frame_shape)
# #         validation_score += char_results['confidence']
# #         total_checks += 1
        
# #         # Calculate final confidence
# #         if total_checks > 0:
# #             final_confidence = validation_score / total_checks
# #         else:
# #             final_confidence = 0.0
        
# #         return final_confidence >= self.MIN_FACE_CONFIDENCE, final_confidence, char_results['reasons']

# # # # Strict Face Tracker with Enhanced Validation
# # # class StrictFaceTracker:
# # #     def __init__(self):
# # #         self.tracks = {}
# # #         self.next_id = 1
# # #         self.max_age = 5.0
# # #         self.max_tracks = 100
        
# # #         # 🎯 STRICT CONFIDENCE THRESHOLDS
# # #         self.MIN_SIMILARITY = 0.85
# # #         self.INITIAL_SIMILARITY_THRESHOLD = 0.92
# # #         self.MIN_EMOTION_CONFIDENCE = 0.70
        
# # #         # Enhanced validation
# # #         self.face_validator = EnhancedFaceValidator()
        
# # #         # Face embedding storage
# # #         self.face_embeddings = {}
        
# # #         # Emotion tracking
# # #         self.emotion_history = defaultdict(list)
# # #         self.last_emotion = {}
# # #         self.emotion_start_time = {}
        
# # #         # Track processing rate
# # #         self.last_processed_time = 0
# # #         self.min_processing_interval = 1.0
        
# # #         print(f"🎯 Enhanced tracking: Multi-detector validation, Similarity>{self.MIN_SIMILARITY}")
        
# # #     def cleanup_old_tracks(self):
# # #         """Remove oldest tracks when we exceed maximum"""
# # #         if len(self.tracks) <= self.max_tracks:
# # #             return
            
# # #         tracks_sorted = sorted(self.tracks.items(), key=lambda x: x[1]['last_seen'])
# # #         tracks_to_remove = len(self.tracks) - self.max_tracks
        
# # #         for i in range(tracks_to_remove):
# # #             track_id, _ = tracks_sorted[i]
# # #             self._remove_track(track_id)
# # #             print(f"🗑️  Cleaned old track: {track_id}")
    
# # #     def _remove_track(self, track_id):
# # #         """Safely remove a track and all associated data"""
# # #         if track_id in self.face_embeddings:
# # #             del self.face_embeddings[track_id]
# # #         if track_id in self.emotion_history:
# # #             del self.emotion_history[track_id]
# # #         if track_id in self.last_emotion:
# # #             del self.last_emotion[track_id]
# # #         if track_id in self.emotion_start_time:
# # #             del self.emotion_start_time[track_id]
# # #         if track_id in self.tracks:
# # #             del self.tracks[track_id]
    
# # #     def get_face_embedding(self, face_roi):
# # #         """Extract face embedding with enhanced validation"""
# # #         try:
# # #             # Enhanced size validation
# # #             if (face_roi.shape[0] < 50 or face_roi.shape[1] < 50 or 
# # #                 face_roi.shape[0] > 500 or face_roi.shape[1] > 500):
# # #                 return None, 0.0
                
# # #             # Use Facenet with strict enforcement
# # #             embedding_obj = DeepFace.represent(
# # #                 face_roi,
# # #                 model_name='Facenet',
# # #                 enforce_detection=True,
# # #                 detector_backend='mtcnn',
# # #                 align=True
# # #             )
            
# # #             if embedding_obj and len(embedding_obj) > 0:
# # #                 embedding = np.array(embedding_obj[0]['embedding'])
                
# # #                 # Calculate embedding quality (norm-based)
# # #                 embedding_norm = np.linalg.norm(embedding)
# # #                 quality_score = min(embedding_norm / 12.0, 1.0)  # Normalized quality
                
# # #                 if quality_score >= self.face_validator.MIN_EMBEDDING_QUALITY:
# # #                     return embedding, quality_score
# # #                 else:
# # #                     print(f"🚫 Low quality embedding: {quality_score:.3f}")
                    
# # #         except Exception as e:
# # #             # enforce_detection failed - not a real face
# # #             pass
        
# # #         return None, 0.0
    
# # #     def analyze_emotion(self, face_roi):
# # #         """Analyze emotion with confidence threshold"""
# # #         try:
# # #             analysis = DeepFace.analyze(
# # #                 face_roi,
# # #                 actions=['emotion'],
# # #                 detector_backend='mtcnn',
# # #                 enforce_detection=True,
# # #                 silent=True
# # #             )
            
# # #             if analysis and len(analysis) > 0:
# # #                 result = analysis[0]
# # #                 emotion_data = result.get('emotion', {})
                
# # #                 if emotion_data:
# # #                     dominant_emotion, emotion_confidence = max(emotion_data.items(), key=lambda x: x[1])
# # #                     emotion_confidence_pct = emotion_confidence / 100.0
                    
# # #                     if emotion_confidence_pct >= self.MIN_EMOTION_CONFIDENCE:
# # #                         return dominant_emotion, emotion_confidence_pct
                        
# # #         except Exception as e:
# # #             print(f"⚠️ Emotion analysis error: {e}")
            
# # #         return None, 0.0
    
# # #     def track_emotion_change(self, track_id, current_emotion, current_confidence):
# # #         """Track emotion changes and record timestamps"""
# # #         current_time = time.time()
        
# # #         # Initialize if first time
# # #         if track_id not in self.last_emotion:
# # #             self.last_emotion[track_id] = current_emotion
# # #             self.emotion_start_time[track_id] = current_time
            
# # #             emotion_record = {
# # #                 'emotion': current_emotion,
# # #                 'confidence': current_confidence,
# # #                 'start_time': current_time,
# # #                 'end_time': current_time,
# # #                 'duration': 0,
# # #                 'timestamp': datetime.now().isoformat()
# # #             }
# # #             self.emotion_history[track_id].append(emotion_record)
# # #             return True, emotion_record
        
# # #         # Check if emotion changed
# # #         last_emotion = self.last_emotion[track_id]
        
# # #         if last_emotion != current_emotion:
# # #             # Calculate duration of previous emotion
# # #             duration = current_time - self.emotion_start_time[track_id]
            
# # #             # Update previous emotion record
# # #             if self.emotion_history[track_id]:
# # #                 previous_record = self.emotion_history[track_id][-1]
# # #                 previous_record['end_time'] = current_time
# # #                 previous_record['duration'] = duration
            
# # #             # Start new emotion record
# # #             emotion_record = {
# # #                 'emotion': current_emotion,
# # #                 'confidence': current_confidence,
# # #                 'start_time': current_time,
# # #                 'end_time': current_time,
# # #                 'duration': 0,
# # #                 'timestamp': datetime.now().isoformat()
# # #             }
# # #             self.emotion_history[track_id].append(emotion_record)
            
# # #             # Update tracking
# # #             self.last_emotion[track_id] = current_emotion
# # #             self.emotion_start_time[track_id] = current_time
            
# # #             print(f"🎭 Emotion changed: {last_emotion} → {current_emotion} (duration: {duration:.1f}s)")
# # #             return True, emotion_record
        
# # #         else:
# # #             # Same emotion, update current record
# # #             if self.emotion_history[track_id]:
# # #                 current_record = self.emotion_history[track_id][-1]
# # #                 current_record['end_time'] = current_time
# # #                 current_record['duration'] = current_time - current_record['start_time']
            
# # #             return False, None
    
# # #     def calculate_similarity(self, embedding1, embedding2):
# # #         """Calculate cosine similarity between embeddings"""
# # #         if embedding1 is None or embedding2 is None:
# # #             return 0.0
        
# # #         try:
# # #             emb1_norm = embedding1 / np.linalg.norm(embedding1)
# # #             emb2_norm = embedding2 / np.linalg.norm(embedding2)
            
# # #             similarity = np.dot(emb1_norm, emb2_norm)
# # #             return max(0.0, min(1.0, similarity))
            
# # #         except Exception as e:
# # #             return 0.0
    
# # #     def should_process_frame(self):
# # #         """Control processing rate - 1 detection per second"""
# # #         current_time = time.time()
# # #         if current_time - self.last_processed_time >= self.min_processing_interval:
# # #             self.last_processed_time = current_time
# # #             return True
# # #         return False
    
# # #     def update(self, detections, face_rois, frame_shape):
# # #         """Update tracker with enhanced human face validation"""
# # #         current_time = time.time()
        
# # #         # Rate limiting
# # #         if not self.should_process_frame():
# # #             return self.tracks
        
# # #         # Remove old tracks
# # #         for track_id in list(self.tracks.keys()):
# # #             if current_time - self.tracks[track_id]['last_seen'] > self.max_age:
# # #                 self._remove_track(track_id)
        
# # #         # Enhanced validation of detections
# # #         valid_detections = []
# # #         valid_face_rois = []
# # #         validation_scores = []
        
# # #         for i, (detection, face_roi) in enumerate(zip(detections, face_rois)):
# # #             # Multi-detector validation
# # #             is_valid, confidence, reasons = self.face_validator.multi_detector_validation(
# # #                 face_roi, detection, frame_shape
# # #             )
            
# # #             if is_valid:
# # #                 valid_detections.append(detection)
# # #                 valid_face_rois.append(face_roi)
# # #                 validation_scores.append(confidence)
# # #                 print(f"✅ Face validated: confidence={confidence:.3f}")
# # #             else:
# # #                 print(f"🚫 Face rejected: {reasons}")
        
# # #         print(f"🔍 Enhanced validation: {len(valid_detections)}/{len(detections)} faces")
        
# # #         # Process valid human face detections
# # #         for i, (detection, face_roi) in enumerate(zip(valid_detections, valid_face_rois)):
# # #             # Get face embedding with quality check
# # #             new_embedding, embedding_quality = self.get_face_embedding(face_roi)
            
# # #             if new_embedding is None:
# # #                 continue
            
# # #             # Analyze emotion
# # #             current_emotion, emotion_confidence = self.analyze_emotion(face_roi)
            
# # #             if current_emotion is None:
# # #                 continue
            
# # #             # Find best match with strict similarity
# # #             best_track_id = None
# # #             best_similarity = 0.0
            
# # #             for track_id, existing_embedding in self.face_embeddings.items():
# # #                 if track_id not in self.tracks:
# # #                     continue
                    
# # #                 similarity = self.calculate_similarity(existing_embedding, new_embedding)
                
# # #                 if similarity > best_similarity and similarity >= self.MIN_SIMILARITY:
# # #                     best_similarity = similarity
# # #                     best_track_id = track_id
            
# # #             if best_track_id is not None:
# # #                 # Update existing track
# # #                 track = self.tracks[best_track_id]
                
# # #                 if best_similarity >= self.INITIAL_SIMILARITY_THRESHOLD:
# # #                     # Track emotion change
# # #                     emotion_changed, emotion_record = self.track_emotion_change(
# # #                         best_track_id, current_emotion, emotion_confidence
# # #                     )
                    
# # #                     track.update({
# # #                         'bbox': detection,
# # #                         'last_seen': current_time,
# # #                         'hit_counter': track['hit_counter'] + 1,
# # #                         'similarity': best_similarity,
# # #                         'person_name': f"Person_{best_track_id}",
# # #                         'current_emotion': current_emotion,
# # #                         'emotion_confidence': emotion_confidence,
# # #                         'emotion_changed': emotion_changed,
# # #                         'emotion_record': emotion_record if emotion_changed else None,
# # #                         'embedding_quality': embedding_quality
# # #                     })
                    
# # #                     # Update embedding slowly
# # #                     alpha = 0.1
# # #                     self.face_embeddings[best_track_id] = (
# # #                         alpha * new_embedding + (1 - alpha) * self.face_embeddings[best_track_id]
# # #                     )
                    
# # #                     print(f"🔁 Track {best_track_id} updated (similarity: {best_similarity:.3f}, emotion: {current_emotion})")
                    
# # #             else:
# # #                 # Create NEW track
# # #                 new_id = self.next_id
                
# # #                 # Track initial emotion
# # #                 emotion_changed, emotion_record = self.track_emotion_change(
# # #                     new_id, current_emotion, emotion_confidence
# # #                 )
                
# # #                 self.tracks[new_id] = {
# # #                     'bbox': detection,
# # #                     'first_seen': current_time,
# # #                     'last_seen': current_time,
# # #                     'hit_counter': 1,
# # #                     'similarity': 1.0,
# # #                     'person_name': f"Person_{new_id}",
# # #                     'current_emotion': current_emotion,
# # #                     'emotion_confidence': emotion_confidence,
# # #                     'emotion_changed': emotion_changed,
# # #                     'emotion_record': emotion_record if emotion_changed else None,
# # #                     'embedding_quality': embedding_quality
# # #                 }
                
# # #                 # Store new embedding
# # #                 self.face_embeddings[new_id] = new_embedding
# # #                 self.next_id += 1
# # #                 print(f"👤 NEW track: Person_{new_id} (emotion: {current_emotion}, quality: {embedding_quality:.3f})")
        
# # #         # Cleanup old tracks if needed
# # #         self.cleanup_old_tracks()
        
# # #         return self.tracks

# # # # Initialize enhanced tracker
# # # face_tracker = StrictFaceTracker()

# # # def detect_human_faces_strict(frame):
# # #     """Use DeepFace with enhanced face detection"""
# # #     if not DEEPFACE_AVAILABLE:
# # #         return [], []
    
# # #     try:
# # #         detections = DeepFace.extract_faces(
# # #             frame,
# # #             detector_backend='mtcnn',
# # #             enforce_detection=False,
# # #             align=False
# # #         )
        
# # #         bboxes = []
# # #         face_rois = []
        
# # #         for detection in detections:
# # #             if 'facial_area' in detection:
# # #                 x, y, w, h = detection['facial_area']['x'], detection['facial_area']['y'], \
# # #                              detection['facial_area']['w'], detection['facial_area']['h']
                
# # #                 # Initial basic validation
# # #                 if (x >= 5 and y >= 5 and 
# # #                     x + w <= frame.shape[1] - 5 and y + h <= frame.shape[0] - 5 and
# # #                     w >= 50 and h >= 50 and w <= 500 and h <= 500):
                    
# # #                     bboxes.append([x, y, w, h])
                    
# # #                     padding = 10
# # #                     x1, y1 = max(0, x-padding), max(0, y-padding)
# # #                     x2, y2 = min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding)
# # #                     face_roi = frame[y1:y2, x1:x2]
                    
# # #                     if face_roi.size > 0:
# # #                         face_rois.append(face_roi)
        
# # #         print(f"🔍 Initial detection: {len(bboxes)} potential faces")
# # #         return bboxes, face_rois
        
# # #     except Exception as e:
# # #         print(f"⚠️ Face detection error: {e}")
# # #         return [], []

# # # # Rest of the code remains the same for WebSocket handling...
# # # async def process_frame(frame_data):
# # #     try:
# # #         start_time = time.time()
        
# # #         # Decode image
# # #         if ',' in frame_data:
# # #             img_data = base64.b64decode(frame_data.split(',')[1])
# # #         else:
# # #             img_data = base64.b64decode(frame_data)
        
# # #         nparr = np.frombuffer(img_data, np.uint8)
# # #         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
# # #         if frame is None:
# # #             return []
        
# # #         # Detect human faces with enhanced validation
# # #         bboxes, face_rois = detect_human_faces_strict(frame)
        
# # #         # Update tracker (with rate limiting)
# # #         tracked_faces = face_tracker.update(bboxes, face_rois, frame.shape)
        
# # #         # Prepare response - only include tracks with valid emotions
# # #         detections = []
# # #         for track_id, track_data in tracked_faces.items():
# # #             # Final validation
# # #             similarity = track_data.get('similarity', 0)
# # #             current_emotion = track_data.get('current_emotion')
# # #             emotion_confidence = track_data.get('emotion_confidence', 0)
            
# # #             if (similarity >= face_tracker.MIN_SIMILARITY and 
# # #                 current_emotion is not None and 
# # #                 emotion_confidence >= face_tracker.MIN_EMOTION_CONFIDENCE):
                
# # #                 x, y, w, h = track_data['bbox']
                
# # #                 detection_data = {
# # #                     'face_id': track_id,
# # #                     'person_name': track_data.get('person_name', f'Person_{track_id}'),
# # #                     'bbox': [int(x), int(y), int(w), int(h)],
# # #                     'timestamp': datetime.now().isoformat(),
# # #                     'track_count': track_data['hit_counter'],
# # #                     'similarity_score': round(track_data.get('similarity', 1.0), 3),
# # #                     'confirmed': track_data['hit_counter'] > 2,
# # #                     'current_emotion': current_emotion,
# # #                     'emotion_confidence': round(emotion_confidence, 3),
# # #                     'emotion_changed': track_data.get('emotion_changed', False),
# # #                     'embedding_quality': round(track_data.get('embedding_quality', 0), 3)
# # #                 }
                
# # #                 # Include emotion history if emotion changed
# # #                 if track_data.get('emotion_changed') and track_data.get('emotion_record'):
# # #                     detection_data['emotion_record'] = track_data['emotion_record']
                
# # #                 # Include full emotion history for this track
# # #                 if track_id in face_tracker.emotion_history:
# # #                     emotion_history = face_tracker.emotion_history[track_id]
# # #                     detection_data['emotion_history_count'] = len(emotion_history)
# # #                     detection_data['recent_emotions'] = emotion_history[-3:] if len(emotion_history) >= 3 else emotion_history
                
# # #                 detections.append(detection_data)
        
# # #         processing_time = (time.time() - start_time) * 1000
        
# # #         if detections:
# # #             print(f"✅ ENHANCED TRACKING: {len(detections)} humans in {processing_time:.1f}ms")
# # #             for det in detections:
# # #                 emotion_status = "🔄" if det['emotion_changed'] else "➡️"
# # #                 print(f"   👤 {det['person_name']}: {det['current_emotion']} {emotion_status} qual={det['embedding_quality']:.3f}")
# # #         else:
# # #             print(f"🔍 No validated human faces tracked")
        
# # #         return detections
        
# # #     except Exception as e:
# # #         print(f"❌ Frame processing error: {e}")
# # #         return []

# # # async def handle_client(websocket):
# # #     client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
# # #     print(f"🎉 Client connected from {client_ip}")
    
# # #     try:
# # #         async for message in websocket:
# # #             data = json.loads(message)
            
# # #             if data['type'] == 'frame':
# # #                 detections = await process_frame(data['frame'])
                
# # #                 response = {
# # #                     'type': 'tracking_results', 
# # #                     'detections': detections,
# # #                     'timestamp': datetime.now().isoformat(),
# # #                     'total_tracks': len(face_tracker.tracks),
# # #                     'active_tracks': len(detections),
# # #                     'max_tracks': face_tracker.max_tracks,
# # #                     'tracker': 'enhanced_face_tracker_with_validation'
# # #                 }
                
# # #                 await websocket.send(json.dumps(response))
                
# # #     except websockets.exceptions.ConnectionClosed:
# # #         print(f"📞 Client {client_ip} disconnected")
# # #     except Exception as e:
# # #         print(f"❌ WebSocket error for {client_ip}: {e}")

# # # async def main():
# # #     async with websockets.serve(handle_client, "localhost", 8765):
# # #         print("🌈 WebSocket server running on ws://localhost:8765")
# # #         print("🎯 ENHANCED FACE TRACKING WITH MULTI-DETECTOR VALIDATION")
# # #         print("📊 Enhanced Validation Features:")
# # #         print(f"   • Multi-detector consensus (DeepFace + Haar Cascade)")
# # #         print(f"   • IoU-based detection validation")
# # #         print(f"   • Comprehensive face characteristic analysis")
# # #         print(f"   • Embedding quality scoring")
# # #         print(f"   • Adaptive confidence thresholds")
# # #         print("🔧 Core Tracking:")
# # #         print(f"   • Similarity Threshold: {face_tracker.MIN_SIMILARITY*100}%+")
# # #         print(f"   • Emotion Confidence: {face_tracker.MIN_EMOTION_CONFIDENCE*100}%+")
# # #         print(f"   • Max Tracks: {face_tracker.max_tracks}")
        
# # #         await asyncio.Future()

# # # if __name__ == "__main__":
# # #     print("=" * 60)
# # #     print("🤖 ENHANCED Face Tracking with Multi-Detector Validation")
# # #     print("🎯 ROBUST HUMAN VERIFICATION - EMOTION TRACKING")
# # #     print("=" * 60)
    
# # #     try:
# # #         asyncio.run(main())
# # #     except KeyboardInterrupt:
# # #         print("\n🛑 Server stopped by user")
# # #     except Exception as e:
# # #         print(f"❌ Server error: {e}")