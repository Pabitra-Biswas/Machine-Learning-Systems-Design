"""
Batch Prediction Endpoint with Ground Truth Evaluation
Week 6: Production Analytics & Monitoring
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import logging
import time
from datetime import datetime
import asyncio
import json

# ✅ FIXED IMPORT - Match your main.py
from src.models.bert_model import BERTNewsClassifier

logger = logging.getLogger(__name__)
router = APIRouter()

# ✅ GLOBAL MODEL INSTANCE (same pattern as main.py)
model: Optional[BERTNewsClassifier] = None

def set_model(bert_model: BERTNewsClassifier):
    """Set global model instance from main.py"""
    global model
    model = bert_model

# ============================================
# PYDANTIC MODELS
# ============================================

class BatchItem(BaseModel):
    """Single item in batch"""
    id: Optional[str] = None
    title: str = Field(..., min_length=1, max_length=512)
    ground_truth: Optional[str] = None

class BatchRequest(BaseModel):
    """Batch prediction request"""
    items: List[BatchItem] = Field(..., max_items=1000)
    include_metrics: bool = True
    return_probabilities: bool = False

class PredictionResult(BaseModel):
    """Single prediction result"""
    id: Optional[str]
    title: str
    prediction: str
    confidence: float
    ground_truth: Optional[str] = None
    is_correct: Optional[bool] = None
    latency_ms: float

class BatchResponse(BaseModel):
    """Batch response"""
    total_items: int
    successful: int
    failed: int
    predictions: List[PredictionResult]
    metrics: Optional[Dict[str, Any]] = None
    execution_time_ms: float
    timestamp: str

# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Calculate comprehensive metrics"""
    
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {"error": "Invalid label lengths"}
    
    try:
        accuracy = accuracy_score(y_true, y_pred)
        classes = sorted(set(y_true + y_pred))
        
        # Per-class metrics
        per_class = {}
        for cls in classes:
            cls_true = [1 if x == cls else 0 for x in y_true]
            cls_pred = [1 if x == cls else 0 for x in y_pred]
            
            per_class[cls] = {
                'precision': float(precision_score(cls_true, cls_pred, zero_division=0)),
                'recall': float(recall_score(cls_true, cls_pred, zero_division=0)),
                'f1': float(f1_score(cls_true, cls_pred, zero_division=0)),
                'support': sum(cls_true)
            }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        return {
            'overall': {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
                'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
                'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
                'total_samples': len(y_true),
                'correct_predictions': sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
            },
            'per_class': per_class,
            'confusion_matrix': {
                'labels': classes,
                'matrix': cm.tolist()
            }
        }
    
    except Exception as e:
        logger.error(f"❌ Metrics error: {e}")
        return {"error": str(e)}

# ============================================
# ENDPOINTS
# ============================================

@router.post("/", response_model=BatchResponse)
async def batch_predict(request: BatchRequest):
    """
    Batch prediction with optional ground truth evaluation
    
    Example:
    {
      "items": [
        {"id": "1", "title": "Apple releases iPhone", "ground_truth": "TECHNOLOGY"},
        {"id": "2", "title": "Stock market crashes", "ground_truth": "BUSINESS"},
        {"id": "3", "title": "COVID cases surge", "ground_truth": "HEALTH"}
      ],
      "include_metrics": true
    }
    """
    
    # ✅ FIXED: Check global model instance
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    predictions_list = []
    ground_truths = []
    predictions = []
    
    logger.info(f"🔄 Batch prediction: {len(request.items)} items")
    
    # Process each item
    for idx, item in enumerate(request.items):
        item_start = time.time()
        
        try:
            # ✅ FIXED: Use model.predict() from BERTNewsClassifier
            result = model.predict(item.title)
            latency = (time.time() - item_start) * 1000
            
            # Check correctness
            is_correct = None
            if item.ground_truth:
                # ✅ FIXED: result has 'prediction' and 'confidence' keys
                is_correct = result['prediction'] == item.ground_truth
                ground_truths.append(item.ground_truth)
                predictions.append(result['prediction'])
            
            # Build result
            pred_result = PredictionResult(
                id=item.id or str(idx),
                title=item.title,
                prediction=result['prediction'],
                confidence=result['confidence'],
                ground_truth=item.ground_truth,
                is_correct=is_correct,
                latency_ms=latency
            )
            
            predictions_list.append(pred_result)
        
        except Exception as e:
            logger.error(f"❌ Item {idx} error: {e}")
    
    # Calculate metrics if ground truth provided
    metrics = None
    if request.include_metrics and ground_truths and predictions:
        metrics = calculate_metrics(ground_truths, predictions)
        logger.info(f"📊 Accuracy: {metrics['overall']['accuracy']:.3f}")
    
    execution_time = (time.time() - start_time) * 1000
    
    return BatchResponse(
        total_items=len(request.items),
        successful=len(predictions_list),
        failed=len(request.items) - len(predictions_list),
        predictions=predictions_list,
        metrics=metrics,
        execution_time_ms=execution_time,
        timestamp=datetime.utcnow().isoformat()
    )

@router.post("/from-csv")
async def batch_predict_csv(file: UploadFile = File(...)):
    """Batch prediction from CSV file"""
    
    # ✅ FIXED: Check global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        logger.info(f"📁 CSV loaded: {len(df)} rows")
        
        # Convert to batch request
        items = []
        for idx, row in df.iterrows():
            items.append(BatchItem(
                id=str(row.get('id', idx)),
                title=row['title'],
                ground_truth=row.get('ground_truth', None)
            ))
        
        # Run batch prediction
        batch_request = BatchRequest(
            items=items,
            include_metrics=True
        )
        
        response = await batch_predict(batch_request)
        
        # Create output CSV
        results_data = []
        for pred in response.predictions:
            results_data.append({
                'id': pred.id,
                'title': pred.title,
                'prediction': pred.prediction,
                'confidence': f"{pred.confidence:.4f}",
                'ground_truth': pred.ground_truth or '',
                'is_correct': pred.is_correct if pred.is_correct is not None else '',
                'latency_ms': f"{pred.latency_ms:.2f}"
            })
        
        results_df = pd.DataFrame(results_data)
        output_path = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"💾 Results saved: {output_path}")
        
        return {
            "status": "success",
            "batch_response": response,
            "output_file": output_path,
            "results_preview": results_df.head(10).to_dict('records')
        }
    
    except Exception as e:
        logger.error(f"❌ CSV error: {e}")
        raise HTTPException(status_code=400, detail=str(e))