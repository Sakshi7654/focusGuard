def evaluate_context_relevance(session_goal: str, active_window: str) -> float:
    """
    Returns a fuzzy score between 0.0 (Irrelevant) and 1.0 (Highly Relevant)
    based on the current session goal and active window title.
    """
    window_lower = active_window.lower()
    goal_lower = session_goal.lower()

    # Universal distraction windows
    distractions = ["instagram", "facebook", "twitter", "netflix", "prime video", "reddit", "reels", "shorts"]
    for d in distractions:
        if d in window_lower:
            return 0.1  # Low relevance

    # Contextual matching based on session goal
    if "dsa" in goal_lower or "code" in goal_lower or "programming" in goal_lower:
        if any(tool in window_lower for tool in ["leetcode", "codeforces", "github", "visual studio code", "pycharm", "stackoverflow"]):
            return 0.95  # High relevance
        elif "youtube" in window_lower:
            return 0.50  # YouTube could be a tutorial or music -> Uncertain/Medium
        else:
            return 0.40

    elif "reading" in goal_lower or "assignment" in goal_lower or "study" in goal_lower:
        if any(doc in window_lower for doc in ["pdf", "docs", "word", "notion", "chatgpt", "scholar"]):
            return 0.90
        else:
            return 0.50

    return 0.50  # Default neutral score