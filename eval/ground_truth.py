"""Ground truth generation for place matches (TODO)."""

def extract_position(pose):
    """Return (x, y, z) position tuple from a pose-like object."""
    if pose is None:
        raise ValueError("Pose is None")
    if isinstance(pose, dict):
        if "position" in pose:
            pose = pose["position"]
        else:
            x = pose.get("x")
            y = pose.get("y")
            z = pose.get("z", 0.0)
            if x is None or y is None:
                raise ValueError(f"Pose dict missing 'x' or 'y': {pose}")
            return float(x), float(y), float(z)
        if hasattr(pose, "position"):
            pose = pose.position
        if hasattr(pose, "x") and hasattr(pose, "y"):
            x = pose.x
            y = pose.y
            z = getattr(pose, "z", 0.0)
            return float(x), float(y), float(z)
        if isinstance(pose, (list, tuple)):
            if len(pose) < 2:
                raise ValueError(f"Pose list/tuple too short: {pose}")
            x = pose[0]
            y = pose[1]
            z = pose[2] if len(pose) > 2 else 0.0
            return float(x), float(y), float(z)
        raise ValueError(f"Unsupported pose type: {type(pose)}")
    

def build_ground_truth(poses, distance_threshold):
    """Generate ground truth matches based on distance or pose rules."""
    if distance_threshold is None:
        raise ValueError("Distance threshold must be specified")
    if distance_threshold <= 0:
        return []
    
    positions = [extract_position(pose) for pose in poses]
    matches = []
    threshold_sq = distance_threshold ** 2

    for i, (xi, yi, zi) in enumerate(positions):
        for j in range(i + 1, len(positions)):
            xj, yj, zj = positions[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            if dx * dx + dy * dy + dz * dz <= threshold_sq:
                matches.append((i, j))

    return matches
