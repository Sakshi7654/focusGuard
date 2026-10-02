from context_evaluator import evaluate_context_relevance

def evaluate_behavior(row: dict) -> dict:
    """
    Takes a single 5-second observation row and produces fuzzy state probabilities:
    - focus_compatible
    - distraction_like
    - uncertain
    """
    head_away = row.get("head_deviated", 0)
    phone_present = row.get("phone_detected", 0)
    idle_sec = row.get("idle_seconds", 0)
    active_window = row.get("active_window", "")
    session_goal = row.get("session_goal", "DSA Practice")

    relevance = evaluate_context_relevance(session_goal, active_window)

    # Base state initializations
    focus_score = 0.0
    distraction_score = 0.0
    uncertain_score = 0.0

    # Rule 1: High Distraction
    # If phone is visible OR looking away on an irrelevant app
    if phone_present == 1:
        distraction_score = 0.90
        focus_score = 0.05
        uncertain_score = 0.05

    elif head_away == 1 and relevance < 0.4:
        distraction_score = 0.85
        focus_score = 0.05
        uncertain_score = 0.10

    # Rule 2: Uncertain / Reading / Thinking state
    # Low keyboard activity (idle), but user is screen-oriented and site is relevant
    elif idle_sec > 15 and head_away == 0 and relevance >= 0.7:
        uncertain_score = 0.75
        focus_score = 0.15
        distraction_score = 0.10

    # Rule 3: Active Task Focus
    # Actively typing/clicking on a relevant window and looking at the screen
    elif idle_sec <= 10 and head_away == 0 and relevance >= 0.7:
        focus_score = 0.85
        distraction_score = 0.05
        uncertain_score = 0.10

    # Rule 4: Intermediate / Unclassified
    else:
        focus_score = 0.33
        distraction_score = 0.33
        uncertain_score = 0.34

    return {
        "focus_score": round(focus_score, 2),
        "distraction_score": round(distraction_score, 2),
        "uncertain_score": round(uncertain_score, 2)
    }

# Quick test run
if __name__ == "__main__":
    # Test case 1: Thinking on LeetCode (Reading without typing)
    test_reading = {
        "head_deviated": 0,
        "phone_detected": 0,
        "idle_seconds": 25,
        "active_window": "Two Sum - LeetCode - Google Chrome",
        "session_goal": "DSA Practice"
    }
    print("Reading Scenario:", evaluate_behavior(test_reading))

    # Test case 2: Browsing Instagram on phone
    test_distraction = {
        "head_deviated": 1,
        "phone_detected": 1,
        "idle_seconds": 40,
        "active_window": "Instagram - Google Chrome",
        "session_goal": "DSA Practice"
    }
    print("Distraction Scenario:", evaluate_behavior(test_distraction))