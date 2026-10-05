def evaluate_context_relevance(session_goal: str, active_window: str) -> float:
    """
    Returns a fuzzy score between 0.0 (Irrelevant) and 1.0 (Highly Relevant)
    based on the current session goal and active window title[cite: 1, 3].
    Safely handles None, NaN, and non-string values.
    """
    # Safe type casting to prevent AttributeError on NaN/floats
    window_lower = str(active_window).lower() if active_window is not None else ""
    goal_lower = str(session_goal).lower() if session_goal is not None else ""

    # Universal distraction windows[cite: 3]
    distractions = ["instagram", "facebook", "twitter", "netflix", "prime video", "reddit", "reels", "shorts", "tiktok"]
    for d in distractions:
        if d in window_lower:
            return 0.10  # Low relevance[cite: 3]

    # Contextual matching based on technical session goal[cite: 3]
    if any(k in goal_lower for k in ["dsa", "code", "programming", "python", "developer"]):
        if any(tool in window_lower for tool in ["leetcode", "codeforces", "github", "visual studio code", "pycharm", "stackoverflow"]):
            return 0.95  # High relevance[cite: 3]
        elif "youtube" in window_lower:
            return 0.50  # YouTube could be a tutorial or music -> Medium/Uncertain[cite: 3]
        else:
            return 0.40

    # Reading / Documentation context matching[cite: 1, 3]
    elif any(k in goal_lower for k in ["reading", "study", "assignment", "paper"]):
        if any(doc in window_lower for doc in ["pdf", "docs", "word", "notion", "chatgpt", "scholar"]):
            return 0.90
        else:
            return 0.50

    return 0.50  # Default neutral score