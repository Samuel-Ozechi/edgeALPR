# src/decision/temporal_voter.py

"""Temporal voting module for multi-frame plate decision making.
Collects OCR results across multiple frames and selects the most reliable plate
reading before triggering a DB lookup and access decision."""

import time
from collections import Counter, deque


class TemporalPlateVoter:
    """
    Collects OCR results across multiple frames and selects the most reliable plate.

    This helps avoid acting on a single noisy OCR prediction.

    Args: 
          window_size: Number of frames to consider in the voting window.
          min_votes: Minimum number of votes required for a plate to be considered valid.
          max_window_time_ms: Maximum time to wait for votes before making a decision.
          use_confidence_weighting: Whether to use OCR confidence scores to break ties.
    """

    def __init__(
        self,
        window_size: int = 5,
        min_votes: int = 2,
        max_window_time_ms: int = 700,
        use_confidence_weighting: bool = True,
    ):
        self.window_size = window_size
        self.min_votes = min_votes
        self.max_window_time_ms = max_window_time_ms
        self.use_confidence_weighting = use_confidence_weighting
        self.buffer = deque(maxlen=window_size)
        self.window_start_time = None

    def reset(self):
        self.buffer.clear()
        self.window_start_time = None

    def add(self, event: dict):
        """
        Add one frame-level OCR event.
        Expected event keys: plate_text, ocr_confidence, detected_vehicle_class,
        vehicle_confidence, plate_confidence, total_latency_ms.
        """
        plate_text = event.get("plate_text")
        ocr_conf = float(event.get("ocr_confidence") or 0.0)

        if not plate_text:
            return

        now = time.perf_counter()

        if self.window_start_time is None:
            self.window_start_time = now

        self.buffer.append({
            "plate_text": plate_text,
            "ocr_confidence": ocr_conf,
            "event": event,
            "timestamp": now,
        })

    def is_ready(self) -> bool:
        if not self.buffer:
            return False

        elapsed_ms = (time.perf_counter() - self.window_start_time) * 1000

        return (
            len(self.buffer) >= self.window_size
            or elapsed_ms >= self.max_window_time_ms
        )

    def get_best(self) -> dict | None:
        """
        Returns the best event based on vote count and confidence.
        Returns None if no plate text meets the min_votes threshold.
        """
        if not self.buffer:
            return None

        plate_texts = [item["plate_text"] for item in self.buffer]
        vote_counts = Counter(plate_texts)

        best_plate, best_count = vote_counts.most_common(1)[0]

        if best_count < self.min_votes:
            return None

        candidates = [
            item for item in self.buffer
            if item["plate_text"] == best_plate
        ]

        if self.use_confidence_weighting:
            best_item = max(candidates, key=lambda x: x["ocr_confidence"])
        else:
            best_item = candidates[-1]

        best_event = best_item["event"].copy()
        best_event["temporal_plate_text"] = best_plate
        best_event["temporal_vote_count"] = best_count
        best_event["temporal_window_size"] = len(self.buffer)
        best_event["temporal_avg_confidence"] = round(
            sum(item["ocr_confidence"] for item in candidates) / len(candidates),
            4,
        )

        return best_event
