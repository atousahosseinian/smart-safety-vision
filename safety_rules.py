from collections import Counter


VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
}


def build_detection_summary(detected_items: list[dict]) -> dict:
    object_names = [item["Object"] for item in detected_items]
    counts = Counter(object_names)

    person_count = counts.get("person", 0)
    vehicle_count = sum(counts.get(vehicle, 0) for vehicle in VEHICLE_CLASSES)

    if person_count > 0 and vehicle_count > 0:
        risk_level = "Medium"
        message = "⚠️ Person and vehicle detected in the same scene. Potential safety risk."
    elif person_count > 0:
        risk_level = "Low"
        message = "Person detected. No vehicle detected in the scene."
    elif vehicle_count > 0:
        risk_level = "Low"
        message = "Vehicle detected. No person detected in the scene."
    else:
        risk_level = "None"
        message = "No person or vehicle detected."

    return {
        "person_count": person_count,
        "vehicle_count": vehicle_count,
        "risk_level": risk_level,
        "message": message,
        "object_counts": counts,
    }