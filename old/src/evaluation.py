"""Evaluation metrics for surveillance pipeline."""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


class EvaluationMetrics:
    """Calculate tracking and detection metrics."""
    
    @staticmethod
    def calculate_mota(ground_truth: List[Dict], predictions: List[Dict]) -> float:
        """
        Calculate Multiple Object Tracking Accuracy (MOTA).
        
        MOTA = 1 - (FN + FP + IDSW) / GT
        where:
        - FN: False Negatives
        - FP: False Positives  
        - IDSW: ID Switches
        - GT: Ground Truth detections
        """
        if len(ground_truth) == 0:
            return 0.0
        
        fn = sum(1 for gt in ground_truth if not any(
            EvaluationMetrics._iou(gt['bbox'], p['bbox']) > 0.5 
            for p in predictions
        ))
        fp = sum(1 for p in predictions if not any(
            EvaluationMetrics._iou(p['bbox'], gt['bbox']) > 0.5 
            for gt in ground_truth
        ))
        idsw = EvaluationMetrics._count_id_switches(ground_truth, predictions)
        
        mota = 1.0 - (fn + fp + idsw) / len(ground_truth)
        return max(0.0, mota)
    
    @staticmethod
    def calculate_motp(ground_truth: List[Dict], predictions: List[Dict]) -> float:
        """
        Calculate Multiple Object Tracking Precision (MOTP).
        
        MOTP = sum(IoU) / matched_detections
        """
        matched_ious = []
        
        for gt in ground_truth:
            best_iou = 0.0
            for p in predictions:
                if gt.get('track_id') == p.get('track_id'):
                    iou = EvaluationMetrics._iou(gt['bbox'], p['bbox'])
                    best_iou = max(best_iou, iou)
            
            if best_iou > 0.5:
                matched_ious.append(best_iou)
        
        if len(matched_ious) == 0:
            return 0.0
        
        return sum(matched_ious) / len(matched_ious)
    
    @staticmethod
    def _iou(box1: Tuple[float, float, float, float], 
             box2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def _count_id_switches(ground_truth: List[Dict], predictions: List[Dict]) -> int:
        """Count ID switches between consecutive frames."""
        gt_tracks = defaultdict(list)  # frame_num -> track_id
        pred_tracks = defaultdict(list)
        
        for gt in ground_truth:
            gt_tracks[gt.get('frame_number', 0)].append(gt.get('track_id'))
        
        for p in predictions:
            pred_tracks[p.get('frame_number', 0)].append(p.get('track_id'))
        
        id_switches = 0
        prev_mapping = {}
        
        for frame_num in sorted(set(list(gt_tracks.keys()) + list(pred_tracks.keys()))):
            current_mapping = {}
            
            for gt_id in gt_tracks.get(frame_num, []):
                for p_id in pred_tracks.get(frame_num, []):
                    if prev_mapping.get(gt_id) != p_id:
                        if gt_id in prev_mapping:
                            id_switches += 1
            
            prev_mapping = current_mapping
        
        return id_switches


def load_event_log(log_path: str) -> Dict:
    """Load event log from JSON file."""
    with open(log_path, 'r') as f:
        return json.load(f)


def evaluate_events(ground_truth_path: str, prediction_path: str) -> Dict:
    """
    Evaluate detected events against ground truth.
    
    Args:
        ground_truth_path: Path to ground truth event log
        prediction_path: Path to predicted event log
        
    Returns:
        Dictionary with evaluation metrics
    """
    gt_events = load_event_log(ground_truth_path)['events']
    pred_events = load_event_log(prediction_path)['events']
    
    # Count by event type
    gt_by_type = defaultdict(int)
    pred_by_type = defaultdict(int)
    tp_by_type = defaultdict(int)
    
    for event in gt_events:
        event_type = event['event_type']
        gt_by_type[event_type] += 1
    
    for event in pred_events:
        event_type = event['event_type']
        pred_by_type[event_type] += 1
        
        # Check if matched in ground truth (within 5 frames)
        for gt_event in gt_events:
            if (gt_event['event_type'] == event_type and
                gt_event['zone_name'] == event['zone_name'] and
                abs(gt_event['frame_number'] - event['frame_number']) < 5):
                tp_by_type[event_type] += 1
                break
    
    # Calculate metrics
    results = {}
    for event_type in set(list(gt_by_type.keys()) + list(pred_by_type.keys())):
        gt = gt_by_type[event_type]
        pred = pred_by_type[event_type]
        tp = tp_by_type[event_type]
        
        precision = tp / pred if pred > 0 else 0.0
        recall = tp / gt if gt > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[event_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': pred - tp,
            'fn': gt - tp
        }
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate surveillance pipeline')
    parser.add_argument('--gt', type=str, help='Ground truth event log')
    parser.add_argument('--pred', type=str, help='Predicted event log')
    
    args = parser.parse_args()
    
    if args.gt and args.pred:
        results = evaluate_events(args.gt, args.pred)
        
        print("Event Detection Metrics:")
        print("-" * 50)
        
        for event_type, metrics in results.items():
            print(f"\n{event_type}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1 Score:  {metrics['f1']:.3f}")
            print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
