# Sieve Analytics: Methodology

This document explains the logic behind the metrics used in the dashboard.

---

## 1. Player Metric: Value Gap

The **Value Gap** measures how much a player outperforms or underperforms their contract.

**Formula:**
> Value Gap = Impact Score - Salary Score

### Calculation Steps

1.  **Normalize Salary (0-100)**
    *   Converts raw salary into a score relative to the league.
    *   Minimum Salary = 0
    *   Maximum Salary = 100

2.  **Normalize Impact (0-100)**
    *   Converts LEBRON impact metric into a score.
    *   Lowest Impact = 0
    *   MVP Level = 100

3.  **Calculate Difference**
    *   We subtract the Salary Score from the Impact Score.

### Interpretation

*   **Positive Gap (+)**: **Underpaid**. The player produces more than they are paid.
*   **Negative Gap (-)**: **Overpaid**. The player is paid more than they produce.
*   **Zero (0)**: **Fair Value**. Pay matches production exactly.

---

## 2. Team Metric: Efficiency Index

The **Efficiency Index** measures how effectively a team converts payroll into wins.

**Formula:**
> Efficiency Index = (2.0 * Win Score) - Payroll Score

### Calculation Steps

1.  **Win Score (Z-Score)**
    *   Measures how many standard deviations a team's win total is from the average.
    *   Above Average Wins = Positive Score

2.  **Payroll Score (Z-Score)**
    *   Measures how many standard deviations a team's payroll is from the average.
    *   Above Average Spending = Positive Score

3.  **Weighted Index**
    *   We weight winning twice as heavily as spending.
    *   This rewards winning teams and penalizes expensive losing teams.

### Interpretation

*   **High Score**: **Elite Efficiency**. Winning a lot, often while spending less.
*   **Low Score**: **Inefficient**. Losing a lot, or spending huge amounts for mediocre results.

---

## 3. Source Data: LEBRON Metric

We use the **LEBRON** metric (Luck-adjusted player Estimate using a Box prior Regularized ON-off) as our primary measure of impact.

*   **What it measures**: Impact per 100 possessions.
*   **Scale**:
    *   `+0.00`: League Average
    *   `+3.00`: All-Star
    *   `+5.00`: MVP Candidate
    *   `-2.00`: Replacement Level

We use LEBRON because it accounts for defense, role, and team context better than simple box score stats like Points Per Game.
