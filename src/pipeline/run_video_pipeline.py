import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import cv2

# Ensure the repository root is on sys.path so direct script execution can import src
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.settings import settings
from src.vision.image_ops import crop_bbox, refine_plate_crop
from src.vision.vehicle_detector import VehicleDetector
from src.vision.plate_detector import PlateDetector
from src.vision.detector_factory import build_vehicle_detector, build_plate_detector # utilize hailo detector
from src.vision.plate_ocr import PlateOCR
from src.db.repository import VehicleRepository
from src.decision.rules_engine import AccessRulesEngine
from src.decision.temporal_voter import TemporalPlateVoter
from src.actuator.factory import build_actuator
from src.logging.event_logger import JsonlEventLogger


def draw_overlay(frame, event):
    decision = event.get("decision")
    plate = event.get("plate_text")
    latency = event.get("total_latency_ms")
    reason = event.get("reason")

    color = (0, 255, 0) if decision == "allow" else (0, 0, 255)

    text = f"{decision.upper()} | Plate: {plate} | {latency} ms | {reason}"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
    )

    return frame


def create_empty_event(reason, total_start, vehicle_latency=0, plate_latency=0, ocr_latency=0):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_id": settings.system.device_id,
        "site_id": settings.system.site_id,
        "plate_text": None,
        "ocr_confidence": 0.0,
        "detected_vehicle_class": None,
        "vehicle_confidence": 0.0,
        "plate_confidence": 0.0,
        "decision": "review",
        "reason": reason,
        "total_latency_ms": round((time.perf_counter() - total_start) * 1000, 2),
        "vehicle_latency_ms": round(vehicle_latency, 2),
        "plate_latency_ms": round(plate_latency, 2),
        "ocr_latency_ms": round(ocr_latency, 2),
    }


def process_frame_perception(
    frame,
    vehicle_detector,
    plate_detector,
    plate_ocr,
):
    """Runs perception only: vehicle detection → plate detection → OCR."""
    total_start = time.perf_counter()

    vehicle_detections, vehicle_latency = vehicle_detector.detect(frame)
    best_vehicle = vehicle_detector.select_best_vehicle(vehicle_detections)

    if best_vehicle is None:
        event = create_empty_event(
            reason="no_vehicle_detected",
            total_start=total_start,
            vehicle_latency=vehicle_latency,
        )
        return event, frame

    x1, y1, x2, y2 = best_vehicle["bbox"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"{best_vehicle['class_name']} {best_vehicle['confidence']:.2f}",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    vehicle_crop = crop_bbox(frame, best_vehicle["bbox"])

    plate_detections, plate_latency = plate_detector.detect(vehicle_crop)
    best_plate = plate_detector.select_best_plate(plate_detections)

    if best_plate is None:
        event = create_empty_event(
            reason="no_plate_detected",
            total_start=total_start,
            vehicle_latency=vehicle_latency,
            plate_latency=plate_latency,
        )
        event["detected_vehicle_class"] = best_vehicle["class_name"]
        event["vehicle_confidence"] = round(best_vehicle["confidence"], 4)
        return event, frame

    plate_crop = crop_bbox(vehicle_crop, best_plate["bbox"])
    refined_plate = refine_plate_crop(plate_crop)

    ocr_result, ocr_latency = plate_ocr.recognize(refined_plate)

    total_latency = (time.perf_counter() - total_start) * 1000

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_id": settings.system.device_id,
        "site_id": settings.system.site_id,
        "plate_text": ocr_result["plate_text"],
        "ocr_confidence": round(ocr_result["confidence"], 4),
        "detected_vehicle_class": best_vehicle["class_name"],
        "vehicle_confidence": round(best_vehicle["confidence"], 4),
        "plate_confidence": round(best_plate["confidence"], 4),
        "decision": "pending",
        "reason": "temporal_wait",
        "total_latency_ms": round(total_latency, 2),
        "vehicle_latency_ms": round(vehicle_latency, 2),
        "plate_latency_ms": round(plate_latency, 2),
        "ocr_latency_ms": round(ocr_latency, 2),
    }

    return event, frame


def finalize_temporal_decision(event, repository, rules_engine, actuator):
    """Takes the voted/best event, checks DB, applies rules, and actuates."""
    vehicle_record = repository.get_vehicle_by_plate(event["plate_text"])

    decision, reason = rules_engine.decide(
        plate_text=event["plate_text"],
        ocr_confidence=event["ocr_confidence"],
        detected_vehicle_class=event.get("detected_vehicle_class"),
        vehicle_record=vehicle_record,
    )

    event["decision"] = decision
    event["reason"] = reason
    event["decision_timestamp"] = datetime.now(timezone.utc).isoformat()

    actuator_triggered, actuator_reason = actuator.execute(
        decision=decision,
        plate_text=event.get("plate_text"),
    )

    event["actuator_triggered"] = actuator_triggered
    event["actuator_reason"] = actuator_reason

    return event


def run_video_pipeline(source=settings.video.source, output_video_path=None):
    
    vehicle_detector = build_vehicle_detector()
    plate_detector = build_plate_detector()

    plate_ocr = PlateOCR(
        lang=settings.models.ocr_lang,
        use_angle_cls=settings.models.ocr_use_angle_cls,
    )

    repository = VehicleRepository(settings.database.db_path)

    rules_engine = AccessRulesEngine(
        ocr_conf_threshold=settings.thresholds.ocr_conf,
        require_class_match=False,
    )

    actuator = build_actuator()

    event_logger = JsonlEventLogger(
        log_path=str(settings.logging.log_path.parent / "video_access_events.jsonl")
    )

    temporal_voter = TemporalPlateVoter(
        window_size=settings.temporal.window_size,
        min_votes=settings.temporal.min_votes,
        max_window_time_ms=settings.temporal.max_window_time_ms,
        use_confidence_weighting=settings.temporal.use_confidence_weighting,
    )

    review_dir = Path(settings.video.review_frame_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    # Ensure source is a string (if Path) or integer (camera index)
    if isinstance(source, Path):
        source = str(source)

    cap = cv2.VideoCapture(source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.video.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.video.frame_height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    video_writer = None
    if output_video_path is not None:
        ret_probe, probe_frame = cap.read()
        if not ret_probe:
            raise RuntimeError("Could not read from video source to determine frame size.")
        actual_height, actual_width = probe_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        for codec in ("mp4v", "avc1", "MJPG", "XVID"):
            writer = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                (actual_width, actual_height),
            )
            if writer.isOpened():
                video_writer = writer
                print(f"Video writer initialized with codec: {codec}")
                break
        
        if video_writer is None:
            print(f"Warning: Failed to initialize video writer with any codec. Video will not be saved.")

    frame_idx = 0
    last_event = None
    last_decision_time = 0.0

    print("Starting video pipeline. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("No frame received. Ending stream.")
            break

        frame_idx += 1
        should_process = frame_idx % settings.video.process_every_n_frames == 0

        if should_process:
            event, frame = process_frame_perception(
                frame=frame,
                vehicle_detector=vehicle_detector,
                plate_detector=plate_detector,
                plate_ocr=plate_ocr,
            )

            event_logger.log(event)
            last_event = event

            if event.get("plate_text"):
                temporal_voter.add(event)

            if settings.temporal.enabled and temporal_voter.is_ready():
                now = time.perf_counter()
                cooldown_elapsed_ms = (now - last_decision_time) * 1000

                if cooldown_elapsed_ms >= settings.temporal.decision_cooldown_ms:
                    best_event = temporal_voter.get_best()

                    if best_event is not None:
                        final_event = finalize_temporal_decision(
                            event=best_event,
                            repository=repository,
                            rules_engine=rules_engine,
                            actuator=actuator,
                        )

                        repository.insert_event(final_event)
                        event_logger.log(final_event)

                        print("FINAL DECISION:", final_event)

                        last_event = final_event
                        last_decision_time = now
                    else:
                        review_event = event.copy()
                        review_event["decision"] = "review"
                        review_event["reason"] = "temporal_vote_not_confident"

                        actuator_triggered, actuator_reason = actuator.execute(
                            decision="review",
                            plate_text=review_event.get("plate_text"),
                        )
                        review_event["actuator_triggered"] = actuator_triggered
                        review_event["actuator_reason"] = actuator_reason

                        repository.insert_event(review_event)
                        event_logger.log(review_event)

                        print("TEMPORAL REVIEW:", review_event)

                        last_event = review_event

                temporal_voter.reset()

            elif not settings.temporal.enabled:
                final_event = finalize_temporal_decision(
                    event=event,
                    repository=repository,
                    rules_engine=rules_engine,
                    actuator=actuator,
                )

                repository.insert_event(final_event)
                event_logger.log(final_event)

                print("FRAME DECISION:", final_event)
                last_event = final_event

        if last_event:
            frame = draw_overlay(frame, last_event)

        if video_writer is not None:
            video_writer.write(frame)

        if settings.video.display:
            cv2.imshow("BarrierVision Live Pipeline", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    if hasattr(actuator.base_actuator, "cleanup"):
        actuator.base_actuator.cleanup()


if __name__ == "__main__":
    sample_video = Path("data/pipeline_test/vehicles/input/vehicle_video_input.mp4")
    

    run_video_pipeline(
        source=sample_video,
        output_video_path="data/pipeline_test/vehicles/output/vehicle_video_output.mp4",  # set to None to skip saving
    )
